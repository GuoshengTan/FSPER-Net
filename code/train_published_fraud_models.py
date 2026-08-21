"""Train published or standard external baselines for the FSPER-Net study.

The implementations in this file are kept separate from FSPER-Net ablations:

* bert_base_chinese: standard Chinese BERT fine-tuning with a CLS head.
* roberta_wwm: standard Chinese RoBERTa-WWM fine-tuning with a CLS head.
* cnn_bigru_attention: paper-architecture reimplementation of Li et al. (2026).
* roberta_mharc: paper-architecture reimplementation of Li et al. (2024).
* pscl: single-model core reimplementation of Xiong et al. (CCL 2023).

The original PSCL-MF competition system additionally used task-adaptive
pretraining, seven-model voting, a post-classifier, FGM, and R-Drop. Those
system-level additions are deliberately not claimed by the ``pscl`` run here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

PROJECT_ROOT_LOCAL = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR_LOCAL = PROJECT_ROOT_LOCAL / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(DEFAULT_CACHE_DIR_LOCAL))
os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_CACHE_DIR_LOCAL / "hub"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from split_manifest import split_from_manifest
from train_roberta_mhag import (
    DEFAULT_CACHE_DIR,
    PROJECT_ROOT,
    FraudTextDataset,
    build_label_descriptions,
    cap_per_class,
    compute_class_weights,
    count_parameters,
    current_lrs,
    json_safe_args,
    load_checkpoint,
    load_fgrc_scd,
    load_spam_message,
    load_telecom_fraud_texts_5,
    make_group_splits,
    make_splits,
    make_weighted_sampler,
    save_checkpoint,
    save_json,
    set_seed,
    summarize_metrics,
)


MODEL_SOURCES: Dict[str, Dict[str, str]] = {
    "bert_base_chinese": {
        "display_name": "BERT-base-Chinese fine-tuning",
        "source": "Devlin et al. (2019)",
        "url": "https://aclanthology.org/N19-1423/",
        "implementation_status": "standard baseline implementation",
        "scope_note": "BERT-base-Chinese CLS representation followed by dropout and a linear classifier.",
    },
    "roberta_wwm": {
        "display_name": "Chinese RoBERTa-WWM fine-tuning",
        "source": "Liu et al. (2019); Cui et al. (2021)",
        "url": "https://github.com/ymcui/Chinese-BERT-wwm",
        "implementation_status": "standard baseline implementation",
        "scope_note": "Chinese RoBERTa-WWM CLS representation followed by dropout and a linear classifier.",
    },
    "cnn_bigru_attention": {
        "display_name": "CNN-BiGRU-Attention",
        "source": "Li, Zhu, and You (2026)",
        "url": "https://link.cnki.net/urlid/31.1260.TP.20260206.1340.002",
        "implementation_status": "paper-architecture reimplementation",
        "scope_note": (
            "Uses the reported random 128-dimensional embeddings, convolution widths 2/3/4, "
            "BiGRU, additive attention, and dropout. The paper did not release source code, so "
            "unspecified convolution channel details are implemented explicitly and recorded."
        ),
    },
    "roberta_mharc": {
        "display_name": "RoBERTa-MHARC",
        "source": "Li, Zhang, and Jiang (2024)",
        "url": "https://doi.org/10.3390/app142411628",
        "implementation_status": "paper-architecture reimplementation",
        "scope_note": (
            "Implements RoBERTa, 12-head self-attention, residual normalization, and the three "
            "reported inconsistency penalties. The authors released the dataset but no verified "
            "training code was found; unspecified loss coefficients remain explicit hyperparameters."
        ),
    },
    "pscl": {
        "display_name": "PSCL (single-model core)",
        "source": "Xiong et al. (CCL 2023)",
        "url": "https://aclanthology.org/2023.ccl-3.22/",
        "implementation_status": "single-model paper-core reimplementation",
        "scope_note": (
            "Implements the shared encoder, label-description prototypes, prototypical supervised "
            "contrastive objective, SimLDAM, and progressive weighting. It is not the complete "
            "PSCL-MF competition system, which used TAPT, seven-model voting, a post-classifier, "
            "FGM, and R-Drop."
        ),
    },
}


class StandardRoBERTaClassifier(nn.Module):
    def __init__(
        self,
        pretrained_model: str,
        num_classes: int,
        dropout: float,
        cache_dir: Path,
        local_files_only: bool,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            pretrained_model,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        hidden_dim = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        features = hidden[:, 0]
        return {"logits": self.classifier(self.dropout(features)), "features": features}


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(torch.tanh(self.proj(hidden))).squeeze(-1)
        scores = scores.masked_fill(~mask.bool(), -1e9)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)
        return pooled, weights


class CNNBiGRUAttentionClassifier(nn.Module):
    """Paper-architecture reimplementation of Li et al. (2026)."""

    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        num_classes: int,
        embedding_dim: int,
        conv_channels: int,
        gru_hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.convolutions = nn.ModuleList(
            [
                nn.Conv1d(embedding_dim, conv_channels, kernel_size=kernel_size, padding="same")
                for kernel_size in (2, 3, 4)
            ]
        )
        self.multiscale_projection = nn.Linear(conv_channels * 3, embedding_dim)
        self.bigru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=gru_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = AdditiveAttention(gru_hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(gru_hidden_dim * 2, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        embedded = self.embedding(input_ids).transpose(1, 2)
        branches = [F.relu(conv(embedded)).transpose(1, 2) for conv in self.convolutions]
        local_features = self.multiscale_projection(torch.cat(branches, dim=-1))
        local_features = local_features * attention_mask.unsqueeze(-1).float()
        sequence, _ = self.bigru(local_features)
        features, attention_weights = self.attention(sequence, attention_mask)
        return {
            "logits": self.classifier(self.dropout(features)),
            "features": features,
            "attention_weights": attention_weights,
        }


class InspectableMultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.attention_dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = hidden.shape
        qkv = self.qkv(hidden).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~attention_mask[:, None, None, :].bool(), -1e9)
        weights = torch.softmax(scores, dim=-1)
        dropped_weights = self.attention_dropout(weights)
        head_outputs = torch.matmul(dropped_weights, v)
        merged = head_outputs.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        return self.out(merged), weights, v, head_outputs


class RoBERTaMHARCClassifier(nn.Module):
    """Paper-architecture reimplementation of Li et al. (2024)."""

    def __init__(
        self,
        pretrained_model: str,
        num_classes: int,
        num_heads: int,
        dropout: float,
        cache_dir: Path,
        local_files_only: bool,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            pretrained_model,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        hidden_dim = self.encoder.config.hidden_size
        self.multihead = InspectableMultiHeadAttention(hidden_dim, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        base_hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        attended, head_attention, head_values, head_outputs = self.multihead(base_hidden, attention_mask)
        residual = self.norm(base_hidden + self.dropout(attended))
        features = residual[:, 0]
        return {
            "logits": self.classifier(self.dropout(features)),
            "features": features,
            "head_attention": head_attention,
            "head_values": head_values,
            "head_outputs": head_outputs,
        }


class PSCLClassifier(nn.Module):
    """Single-model core of Xiong et al. (CCL 2023), excluding system fusion."""

    def __init__(
        self,
        pretrained_model: str,
        num_classes: int,
        dropout: float,
        cache_dir: Path,
        local_files_only: bool,
        description_input_ids: torch.Tensor,
        description_attention_mask: torch.Tensor,
        temperature: float,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            pretrained_model,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        hidden_dim = self.encoder.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.temperature = temperature
        self.register_buffer("description_input_ids", description_input_ids)
        self.register_buffer("description_attention_mask", description_attention_mask)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return F.normalize(self.projector(hidden[:, 0]), p=2, dim=-1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.encode(input_ids, attention_mask)
        prototypes = self.encode(self.description_input_ids, self.description_attention_mask)
        prototype_logits = torch.matmul(features, prototypes.t()) / self.temperature
        return {
            "logits": self.classifier(self.dropout(features)),
            "features": features,
            "prototypes": prototypes,
            "prototype_logits": prototype_logits,
        }


def cosine_head_inconsistency(values: torch.Tensor) -> torch.Tensor:
    """Equation (5)/(7) of Li et al. (2024), including all H x H head pairs."""
    flattened = F.normalize(values.flatten(start_dim=2), p=2, dim=-1)
    similarities = torch.matmul(flattened, flattened.transpose(1, 2))
    return -similarities.mean()


def position_inconsistency(attention: torch.Tensor) -> torch.Tensor:
    """Equation (6) of Li et al. (2024), averaged over samples and head pairs."""
    pairwise_overlap = (
        attention.unsqueeze(2) * attention.unsqueeze(1)
    ).abs().sum(dim=(-1, -2))
    return -pairwise_overlap.mean()


class ExternalModelCriterion(nn.Module):
    def __init__(
        self,
        model_key: str,
        class_weights: Optional[torch.Tensor],
        class_counts: Sequence[int],
        max_epochs: int,
        mharc_subspace_weight: float,
        mharc_position_weight: float,
        mharc_representation_weight: float,
        pscl_classification_weight: float,
        pscl_similarity_gamma: float,
        ldam_max_margin: float,
    ) -> None:
        super().__init__()
        self.model_key = model_key
        self.class_weights = class_weights
        self.max_epochs = max_epochs
        self.mharc_subspace_weight = mharc_subspace_weight
        self.mharc_position_weight = mharc_position_weight
        self.mharc_representation_weight = mharc_representation_weight
        self.pscl_classification_weight = pscl_classification_weight
        self.pscl_similarity_gamma = pscl_similarity_gamma
        counts = torch.tensor(class_counts, dtype=torch.float).clamp_min(1)
        margins = counts.pow(-0.25)
        margins = margins * (ldam_max_margin / margins.max())
        self.register_buffer("ldam_margins", margins)

    def standard_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        loss = F.cross_entropy(outputs["logits"], labels, weight=self.class_weights)
        return loss, {"loss_cls": float(loss.detach().cpu()), "loss_total": float(loss.detach().cpu())}

    def mharc_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        classification = F.cross_entropy(outputs["logits"], labels, weight=self.class_weights)
        subspace = cosine_head_inconsistency(outputs["head_values"])
        position = position_inconsistency(outputs["head_attention"])
        representation = cosine_head_inconsistency(outputs["head_outputs"])
        loss = (
            classification
            + self.mharc_subspace_weight * subspace
            + self.mharc_position_weight * position
            + self.mharc_representation_weight * representation
        )
        parts = {
            "loss_cls": float(classification.detach().cpu()),
            "loss_subspace": float(subspace.detach().cpu()),
            "loss_position": float(position.detach().cpu()),
            "loss_representation": float(representation.detach().cpu()),
            "loss_total": float(loss.detach().cpu()),
        }
        return loss, parts

    def pscl_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        epoch: int,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        prototype_loss = F.cross_entropy(outputs["prototype_logits"], labels)
        logits = outputs["logits"]
        prototypes = outputs["prototypes"]
        prototype_similarity = torch.matmul(prototypes, prototypes.t())
        adjusted = logits + self.pscl_similarity_gamma * prototype_similarity[labels]
        adjusted = adjusted.clone()
        true_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        true_logits = true_logits - self.ldam_margins[labels]
        adjusted.scatter_(1, labels.unsqueeze(1), true_logits.unsqueeze(1))
        sim_ldam = F.cross_entropy(adjusted, labels)
        progress = 1.0 if self.max_epochs <= 1 else (epoch - 1) / (self.max_epochs - 1)
        alpha = 1.0 - progress
        loss = alpha * prototype_loss + (1.0 - alpha) * self.pscl_classification_weight * sim_ldam
        parts = {
            "loss_pscl": float(prototype_loss.detach().cpu()),
            "loss_sim_ldam": float(sim_ldam.detach().cpu()),
            "progressive_alpha": float(alpha),
            "loss_total": float(loss.detach().cpu()),
        }
        return loss, parts

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        epoch: int,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        if self.model_key == "roberta_mharc":
            return self.mharc_loss(outputs, labels)
        if self.model_key == "pscl":
            return self.pscl_loss(outputs, labels, epoch)
        return self.standard_loss(outputs, labels)


def build_model(
    args: argparse.Namespace,
    tokenizer,
    num_classes: int,
    label_descriptions: Sequence[str],
) -> nn.Module:
    if args.model in {"bert_base_chinese", "roberta_wwm"}:
        return StandardRoBERTaClassifier(
            args.pretrained_model,
            num_classes,
            args.dropout,
            args.cache_dir,
            not args.allow_download,
        )
    if args.model == "cnn_bigru_attention":
        return CNNBiGRUAttentionClassifier(
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id or 0,
            num_classes=num_classes,
            embedding_dim=args.embedding_dim,
            conv_channels=args.conv_channels,
            gru_hidden_dim=args.gru_hidden_dim,
            dropout=args.dropout,
        )
    if args.model == "roberta_mharc":
        return RoBERTaMHARCClassifier(
            args.pretrained_model,
            num_classes,
            args.num_heads,
            args.dropout,
            args.cache_dir,
            not args.allow_download,
        )
    if args.model == "pscl":
        encoded = tokenizer(
            list(label_descriptions),
            max_length=args.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return PSCLClassifier(
            args.pretrained_model,
            num_classes,
            args.dropout,
            args.cache_dir,
            not args.allow_download,
            encoded["input_ids"],
            encoded["attention_mask"],
            args.pscl_temperature,
        )
    raise ValueError(args.model)


def build_optimizer_and_scheduler(
    model: nn.Module,
    args: argparse.Namespace,
    total_steps: int,
) -> tuple[torch.optim.Optimizer, Any]:
    if args.model == "cnn_bigru_attention":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.head_lr, weight_decay=args.weight_decay)
        return optimizer, None

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    encoder_decay, encoder_no_decay, head_params = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("encoder."):
            if any(token in name for token in no_decay):
                encoder_no_decay.append(parameter)
            else:
                encoder_decay.append(parameter)
        else:
            head_params.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_decay, "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": encoder_no_decay, "lr": args.lr, "weight_decay": 0.0},
            {"params": head_params, "lr": args.head_lr, "weight_decay": args.weight_decay},
        ]
    )
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: ExternalModelCriterion,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    grad_clip: float,
    epoch: int,
) -> tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    total_parts: Counter[str] = Counter()
    for batch in tqdm(loader, desc=f"train {epoch}", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(input_ids, attention_mask)
        loss, parts = criterion(outputs, labels, epoch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * labels.size(0)
        for key, value in parts.items():
            total_parts[key] += value * labels.size(0)
    size = len(loader.dataset)
    return total_loss / size, {key: float(value / size) for key, value in total_parts.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: ExternalModelCriterion,
    device: torch.device,
    epoch: int,
) -> tuple[float, Dict[str, float], list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    total_parts: Counter[str] = Counter()
    y_true: list[int] = []
    y_pred: list[int] = []
    for batch in tqdm(loader, desc="eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        outputs = model(input_ids, attention_mask)
        loss, parts = criterion(outputs, labels, epoch)
        predictions = outputs["logits"].argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        for key, value in parts.items():
            total_parts[key] += value * labels.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())
    size = len(loader.dataset)
    return (
        total_loss / size,
        {key: float(value / size) for key, value in total_parts.items()},
        y_true,
        y_pred,
    )


def load_dataset(args: argparse.Namespace):
    groups = None
    if args.dataset == "telecom5":
        texts, labels = load_telecom_fraud_texts_5(args.data_dir)
    elif args.dataset == "fgrc_scd":
        texts, labels, groups = load_fgrc_scd(args.data_dir)
    elif args.dataset == "spam_message":
        texts, labels = load_spam_message(args.data_dir)
    else:
        raise ValueError(args.dataset)
    return cap_per_class(texts, labels, groups, args.max_samples_per_class, args.seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train external published fraud-text model baselines.")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_SOURCES),
        required=True,
    )
    parser.add_argument("--dataset", choices=["telecom5", "fgrc_scd", "spam_message"], default="telecom5")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "Telecom_Fraud_Texts_5",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "published_fraud_model",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--pretrained-model", default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--conv-channels", type=int, default=128)
    parser.add_argument("--gru-hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--mharc-subspace-weight", type=float, default=0.01)
    parser.add_argument("--mharc-position-weight", type=float, default=0.01)
    parser.add_argument("--mharc-representation-weight", type=float, default=0.01)
    parser.add_argument("--pscl-temperature", type=float, default=0.07)
    parser.add_argument("--pscl-classification-weight", type=float, default=1.0)
    parser.add_argument("--pscl-similarity-gamma", type=float, default=0.1)
    parser.add_argument("--ldam-max-margin", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--no-class-weight", action="store_true")
    parser.add_argument("--max-samples-per-class", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help=(
            "Optional validated train/valid/test index manifest. Used by "
            "cross-fitting experiments to prevent held-out sample leakage."
        ),
    )
    parser.add_argument(
        "--save-best-by",
        choices=["valid_macro_f1", "valid_accuracy", "valid_weighted_f1"],
        default="valid_macro_f1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    texts, raw_labels, groups = load_dataset(args)
    if not texts:
        raise RuntimeError(f"No data loaded from {args.data_dir}")
    label_names = sorted(set(raw_labels))
    label_to_id = {label: index for index, label in enumerate(label_names)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    labels = [label_to_id[label] for label in raw_labels]
    label_descriptions = build_label_descriptions(label_names)

    split_manifest = None
    if args.split_manifest is not None:
        split, split_manifest = split_from_manifest(
            args.split_manifest,
            texts,
            labels,
            groups,
        )
    elif groups is None:
        split = make_splits(texts, labels, args.seed)
    else:
        split = make_group_splits(texts, labels, groups, args.seed)
    train_texts, valid_texts, test_texts, train_labels, valid_labels, test_labels = split

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model,
        cache_dir=args.cache_dir,
        local_files_only=not args.allow_download,
    )
    train_dataset = FraudTextDataset(train_texts, train_labels, tokenizer, args.max_len)
    valid_dataset = FraudTextDataset(valid_texts, valid_labels, tokenizer, args.max_len)
    test_dataset = FraudTextDataset(test_texts, test_labels, tokenizer, args.max_len)
    sampler = make_weighted_sampler(train_labels, len(label_names)) if args.weighted_sampler else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, tokenizer, len(label_names), label_descriptions).to(device)
    parameter_info = count_parameters(model)
    total_steps = max(1, len(train_loader) * args.epochs)
    optimizer, scheduler = build_optimizer_and_scheduler(model, args, total_steps)
    class_weights = None
    if not args.no_class_weight and args.model != "pscl":
        class_weights = compute_class_weights(train_labels, len(label_names)).to(device)
    class_counts = torch.bincount(torch.tensor(train_labels), minlength=len(label_names)).tolist()
    criterion = ExternalModelCriterion(
        model_key=args.model,
        class_weights=class_weights,
        class_counts=class_counts,
        max_epochs=args.epochs,
        mharc_subspace_weight=args.mharc_subspace_weight,
        mharc_position_weight=args.mharc_position_weight,
        mharc_representation_weight=args.mharc_representation_weight,
        pscl_classification_weight=args.pscl_classification_weight,
        pscl_similarity_gamma=args.pscl_similarity_gamma,
        ldam_max_margin=args.ldam_max_margin,
    ).to(device)

    history: list[Dict[str, Any]] = []
    best_score = -1.0
    start_epoch = 1
    state = load_checkpoint(args.resume, device)
    if state:
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler is not None and state.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        history = list(state.get("history", []))
        best_score = float(state.get("best_score", -1.0))
        start_epoch = int(state["epoch"]) + 1
        print(
            json.dumps(
                {"resume_from": str(args.resume), "next_epoch": start_epoch, "best_score": best_score},
                ensure_ascii=False,
            ),
            flush=True,
        )

    source = MODEL_SOURCES[args.model]
    configuration = {
        "model_key": args.model,
        "model_name": source["display_name"],
        "source": source,
        "created_at_unix": time.time(),
        "args": json_safe_args(args),
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "label_descriptions": dict(zip(label_names, label_descriptions)),
        "label_distribution": Counter(raw_labels),
        "split_sizes": {
            "train": len(train_dataset),
            "valid": len(valid_dataset),
            "test": len(test_dataset),
        },
        "parameter_info": parameter_info,
        "selection": {"save_best_by": args.save_best_by},
        "matched_protocol": {
            "split_seed": args.seed,
            "group_aware_split": groups is not None,
            "split_manifest": (
                str(args.split_manifest.resolve())
                if args.split_manifest is not None
                else None
            ),
            "split_role": (
                split_manifest.get("test_role")
                if split_manifest is not None
                else "final_test"
            ),
            "maximum_length": args.max_len,
            "best_checkpoint_selected_on": args.save_best_by,
            "test_used_for_selection": False,
        },
    }
    save_json(args.output_dir / "experiment_config.json", configuration)
    save_json(args.output_dir / "label_mapping.json", {"label_to_id": label_to_id, "id_to_label": id_to_label})
    save_json(
        args.output_dir / "dataset_summary.json",
        {
            "dataset": args.dataset,
            "total": len(texts),
            "train": len(train_dataset),
            "valid": len(valid_dataset),
            "test": len(test_dataset),
            "labels": Counter(raw_labels),
            "model_name": source["display_name"],
            "source": source,
            **parameter_info,
        },
    )

    def checkpoint_payload(epoch: int) -> Dict[str, Any]:
        return {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_score": best_score,
            "history": history,
            "args": json_safe_args(args),
            "source": source,
            "label_to_id": label_to_id,
            "id_to_label": id_to_label,
            "parameter_info": parameter_info,
            "saved_at_unix": time.time(),
        }

    print(
        json.dumps(
            {
                "device": str(device),
                "model": source["display_name"],
                "implementation_status": source["implementation_status"],
                **parameter_info,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_parts = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            args.grad_clip,
            epoch,
        )
        valid_loss, valid_parts, valid_true, valid_pred = evaluate(
            model,
            valid_loader,
            criterion,
            device,
            epoch,
        )
        valid_metrics = summarize_metrics(valid_true, valid_pred)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_loss_parts": train_parts,
            "valid_loss": valid_loss,
            "valid_loss_parts": valid_parts,
            "valid_accuracy": valid_metrics["accuracy"],
            "valid_macro_precision": valid_metrics["macro_precision"],
            "valid_macro_recall": valid_metrics["macro_recall"],
            "valid_macro_f1": valid_metrics["macro_f1"],
            "valid_weighted_precision": valid_metrics["weighted_precision"],
            "valid_weighted_recall": valid_metrics["weighted_recall"],
            "valid_weighted_f1": valid_metrics["weighted_f1"],
            "learning_rates": current_lrs(optimizer),
            "saved_at_unix": time.time(),
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

        score = float(row[args.save_best_by])
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": json_safe_args(args),
                    "source": source,
                    "label_to_id": label_to_id,
                    "id_to_label": id_to_label,
                    "best_score": best_score,
                    "best_epoch": epoch,
                    "parameter_info": parameter_info,
                },
                args.output_dir / "best_model.pt",
            )
            save_json(
                args.output_dir / "best_metrics.json",
                {
                    "best_epoch": epoch,
                    "best_score": best_score,
                    "save_best_by": args.save_best_by,
                    "validation": row,
                    "source": source,
                    "saved_at_unix": time.time(),
                },
            )
        save_checkpoint(args.output_dir / "latest_checkpoint.pt", checkpoint_payload(epoch))
        save_json(args.output_dir / "training_history.json", history)

    best = torch.load(args.output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    best_epoch = int(best["best_epoch"])
    test_loss, test_parts, test_true, test_pred = evaluate(
        model,
        test_loader,
        criterion,
        device,
        best_epoch,
    )
    target_names = [id_to_label[index] for index in range(len(label_names))]
    report = classification_report(
        test_true,
        test_pred,
        target_names=target_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "model_key": args.model,
        "model_name": source["display_name"],
        "source": source,
        "best_epoch": best_epoch,
        "best_score": best["best_score"],
        "save_best_by": args.save_best_by,
        "test_loss": test_loss,
        "test_loss_parts": test_parts,
        "test_accuracy": float(accuracy_score(test_true, test_pred)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(test_true, test_pred).tolist(),
        "test_true": test_true,
        "test_pred": test_pred,
        "labels": target_names,
        "history": history,
    }
    save_json(args.output_dir / "metrics.json", metrics)
    macro = report["macro avg"]
    weighted = report["weighted avg"]
    final_summary = {
        "model_key": args.model,
        "model_name": source["display_name"],
        "source": source,
        "output_dir": str(args.output_dir),
        "best_epoch": best_epoch,
        "best_score": best["best_score"],
        "save_best_by": args.save_best_by,
        "test_loss": test_loss,
        "test_loss_parts": test_parts,
        "test_accuracy": float(accuracy_score(test_true, test_pred)),
        "test_macro_precision": macro["precision"],
        "test_macro_recall": macro["recall"],
        "test_macro_f1": macro["f1-score"],
        "test_weighted_precision": weighted["precision"],
        "test_weighted_recall": weighted["recall"],
        "test_weighted_f1": weighted["f1-score"],
        "args": json_safe_args(args),
        "parameter_info": parameter_info,
        "saved_at_unix": time.time(),
    }
    save_json(args.output_dir / "final_summary.json", final_summary)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
