"""Train FS-PSCL for Chinese telecom-fraud text classification.

FS-PSCL warm-starts from the single-prototype PSCL checkpoint and replaces the
single label prototype with an adaptive fraud-script prototype bank. The script
branch works on pre-projection RoBERTa features so that script-level variation
is not erased by PSCL's compact projection space.

Version 2 uses script-specific semantic anchors, backbone-dominant training, and
confidence-gated residual fusion so an uncertain prototype branch can fall back
to the Chinese RoBERTa classifier.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CACHE_DIR = PROJECT_ROOT / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(LOCAL_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(LOCAL_CACHE_DIR / "hub"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from split_manifest import split_from_manifest
from train_published_fraud_models import load_dataset
from train_roberta_mhag import (
    FraudTextDataset,
    build_label_descriptions,
    count_parameters,
    current_lrs,
    json_safe_args,
    load_checkpoint,
    make_group_splits,
    make_splits,
    make_weighted_sampler,
    save_checkpoint,
    save_json,
    set_seed,
    summarize_metrics,
)


MODEL_NAME = "FS-PSCL"
MODEL_VERSION = "3.0"
ACTIVE_STATUS_PATH: Optional[Path] = None
DEFAULT_PSCL_RUNS = {
    "telecom5": PROJECT_ROOT / "outputs" / "telecom5" / "seed_42" / "pscl_epoch20",
    "fgrc_scd": PROJECT_ROOT / "outputs" / "fgrc_scd" / "seed_42" / "pscl_epoch20",
}
DEFAULT_SCRIPT_COUNTS = {
    "telecom5": {
        "冒充电商物流客服类": 2,
        "贷款、代办信用卡类": 2,
    },
    "fgrc_scd": {
        "冒充电商物流客服类": 2,
        "虚假信用服务类": 2,
        "虚假购物、服务类": 3,
    },
}

SCRIPT_DESCRIPTION_TEMPLATES = {
    "telecom5": {
        "冒充公检法及政府机关类": [
            "冒充公检法及政府机关诈骗：冒充公安、检察院、法院或政府人员，以涉案、通缉、洗钱或资金审查相威胁，要求转账至所谓安全账户或配合远程调查。"
        ],
        "冒充电商物流客服类": [
            "电商物流理赔脚本：冒充电商、快递或物流客服，以订单异常、商品破损、快递丢失、退款理赔为由，要求提供账户信息、下载软件或转账认证。",
            "平台账户处理脚本：冒充购物平台客服，以会员、白条、金条、账户或征信异常为由，要求共享屏幕、加入会议、操作网银或转移资金。",
        ],
        "冒充领导、熟人类": [
            "冒充领导熟人诈骗：冒充领导、亲友、同学或其他熟人，以紧急借款、临时周转、代付机票、代办事项等理由要求向指定账户转账。"
        ],
        "正常文本": [
            "正常文本：普通通知、日常交流、官方安全提醒或正常业务信息，不冒充身份施压，不要求向陌生账户转账，也不索取验证码、密码或敏感信息。"
        ],
        "贷款、代办信用卡类": [
            "贷款放款脚本：以低息贷款、贷款额度或快速放款为诱饵，声称银行卡错误、贷款冻结或流水不足，要求支付解冻费、保证金或认证金。",
            "信用卡代办脚本：以代办信用卡、信用提额、征信修复或消除不良记录为由，要求预付手续费、提供验证码或提交敏感账户信息。",
        ],
    },
    "fgrc_scd": {
        "冒充公检法及政府机关类": [
            "冒充公检法及政府机关诈骗：冒充公安、检察院、法院、社保局或监管机构，以涉案、洗钱、账户冻结或资金审查相威胁，要求转移资金、共享屏幕或进入所谓安全账户。"
        ],
        "冒充军警购物类诈骗": [
            "冒充军警采购诈骗：冒充部队、武警、消防或政府采购人员，虚构大宗物资采购并指定供应商，诱导商户垫资、支付货款或向指定账户转账。"
        ],
        "冒充电商物流客服类": [
            "物流退款理赔脚本：冒充快递、物流或电商客服，以快递丢失、商品破损、订单异常、退款理赔为由，要求绑定银行卡、清空余额、提取备用金或转账认证。",
            "电商平台账户脚本：冒充京东等平台客服，以订单、会员、白条、金条或账户异常为由，要求下载会议软件、共享屏幕、操作网银或关闭相关服务。",
        ],
        "冒充领导、熟人类": [
            "冒充领导熟人诈骗：冒充领导、亲友、同学或其他熟人，以紧急借款、临时周转、代付机票、医疗事故或代办事项为由要求转账。"
        ],
        "无风险": [
            "无风险正常文本：普通业务通知、官方渠道提示、日常交流或反诈提醒，不冒充他人制造紧迫感，不要求向陌生账户转账，也不索取验证码、密码或银行卡信息。"
        ],
        "网络婚恋、交友类": [
            "网络婚恋交友诈骗：通过交友平台、恋爱关系或私密社交建立信任，以会员激活、任务充值、见面机票、生活困难或紧急借款等理由诱导付款。"
        ],
        "网黑案件": [
            "网黑及私密敲诈诈骗：通过私密视频、裸聊或社交软件获取隐私内容，以公开视频、曝光通讯录等方式威胁转账，或诱导下载私密聊天应用实施后续勒索。"
        ],
        "虚假信用服务类": [
            "信用服务关闭修复脚本：以白条、金条、学生贷、征信异常或信用记录受损为由，诱导关闭服务、信用修复、贷款过流水或向所谓安全账户转账。",
            "虚假贷款解冻脚本：以贷款额度、快速放款为诱饵，声称银行卡填写错误、贷款冻结或认证失败，要求支付解冻费、保证金、认证金或刷流水。",
        ],
        "虚假网络投资理财类": [
            "虚假网络投资理财诈骗：以刷单返利、网络兼职、荐股理财、虚拟币或高收益投资为诱饵，先给予小额回报，再诱导持续充值、垫资或追加投资。"
        ],
        "虚假购物、服务类": [
            "虚假专业服务脚本：以培训退费、证书办理、人才入库、代办业务或其他专业服务为名，要求添加私人联系方式、注册店铺、充值或预付服务费用。",
            "虚拟物品交易脚本：围绕游戏账号、虚拟商品或二手平台交易，诱导脱离平台联系、点击虚假链接、扫码操作，或支付保证金和解冻资金。",
            "虚假商品交易脚本：以商品采购、闲鱼交易、厂家供货或低价销售为由，要求支付定金、开通保障、点击站外链接或向个人账户转账。",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=sorted(DEFAULT_PSCL_RUNS),
        default="fgrc_scd",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "FGRC-SCD" / "sms" / "message",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "fs_pscl",
    )
    parser.add_argument("--cache-dir", type=Path, default=LOCAL_CACHE_DIR)
    parser.add_argument("--pretrained-model", default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--warm-start", type=Path, default=None)
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Optional validated fixed train/valid/test split manifest.",
    )
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument(
        "--compat-v2",
        action="store_true",
        help=(
            "Apply the fixed script-prototype stabilization settings used as "
            "the warm start for FSPER-Net."
        ),
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--fusion-weight", type=float, default=0.1)
    parser.add_argument("--fusion-gate-init", type=float, default=0.05)
    parser.add_argument("--fusion-gate-max", type=float, default=0.25)
    parser.add_argument("--gate-regularization-weight", type=float, default=0.01)
    parser.add_argument("--classification-weight", type=float, default=1.0)
    parser.add_argument("--script-loss-weight", type=float, default=0.2)
    parser.add_argument("--similarity-gamma", type=float, default=0.1)
    parser.add_argument("--ldam-max-margin", type=float, default=0.5)
    parser.add_argument("--rival-weight", type=float, default=0.1)
    parser.add_argument("--rival-base-margin", type=float, default=0.1)
    parser.add_argument("--rival-similarity-scale", type=float, default=0.2)
    parser.add_argument("--diversity-weight", type=float, default=0.05)
    parser.add_argument("--diversity-max-similarity", type=float, default=0.85)
    parser.add_argument("--alignment-weight", type=float, default=0.05)
    parser.add_argument("--centroid-momentum", type=float, default=0.995)
    parser.add_argument("--semantic-anchor-momentum", type=float, default=0.99)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--max-samples-per-class", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--disable-rival", action="store_true")
    parser.add_argument("--disable-diversity", action="store_true")
    parser.add_argument("--disable-alignment", action="store_true")
    parser.add_argument(
        "--save-best-by",
        choices=["valid_macro_f1", "valid_accuracy", "valid_weighted_f1"],
        default="valid_macro_f1",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> None:
    if args.split_seed is None:
        args.split_seed = args.seed
    if args.compat_v2:
        args.fusion_gate_max = 1.0
        args.gate_regularization_weight = 0.0
        args.diversity_weight = 0.02
        args.centroid_momentum = 0.95
        args.semantic_anchor_momentum = 0.0
    if args.warm_start is None:
        args.warm_start = DEFAULT_PSCL_RUNS[args.dataset] / "best_model.pt"
    if args.feature_cache is None:
        args.feature_cache = (
            PROJECT_ROOT
            / "outputs"
            / args.dataset
            / "seed_42"
            / "diagnostics"
            / args.dataset
            / "features_cache.npz"
        )


def build_script_descriptions(
    dataset: str,
    label_names: Sequence[str],
    script_counts: Sequence[int],
) -> tuple[list[list[str]], dict[str, list[str]]]:
    templates = SCRIPT_DESCRIPTION_TEMPLATES.get(dataset, {})
    fallback_descriptions = dict(
        zip(label_names, build_label_descriptions(label_names))
    )
    max_scripts = max(script_counts)
    padded_rows: list[list[str]] = []
    active_descriptions: dict[str, list[str]] = {}
    for label, count in zip(label_names, script_counts):
        descriptions = list(templates.get(label, [fallback_descriptions[label]]))
        if len(descriptions) != count:
            raise RuntimeError(
                f"Expected {count} script descriptions for {dataset}/{label}, "
                f"found {len(descriptions)}."
            )
        active_descriptions[label] = descriptions
        padded_rows.append(
            descriptions + [descriptions[-1]] * (max_scripts - count)
        )
    return padded_rows, active_descriptions


def normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def fit_cluster_assignments(
    features: np.ndarray,
    n_clusters: int,
    seed: int,
) -> np.ndarray:
    if n_clusters == 1:
        return np.zeros(len(features), dtype=np.int64)
    if len(features) > 5000:
        estimator = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=seed,
            n_init=10,
            batch_size=1024,
            max_iter=300,
        )
    else:
        estimator = KMeans(
            n_clusters=n_clusters,
            random_state=seed,
            n_init=10,
            max_iter=300,
        )
    return estimator.fit_predict(features)


def build_initial_script_prototypes(
    feature_cache: Path,
    label_names: Sequence[str],
    train_labels: Sequence[int],
    dataset: str,
    seed: int,
) -> tuple[torch.Tensor, list[int], dict[str, Any]]:
    if not feature_cache.exists():
        raise FileNotFoundError(
            f"Feature cache not found: {feature_cache}. "
            "Run diagnose_pscl_representation.py first."
        )
    cached = np.load(feature_cache)
    encoder_features = normalize_rows(
        cached["train_encoder_features"].astype(np.float32)
    )
    cached_labels = cached["train_labels"].astype(np.int64)
    expected_labels = np.asarray(train_labels, dtype=np.int64)
    if len(cached_labels) != len(expected_labels) or not np.array_equal(
        cached_labels, expected_labels
    ):
        raise RuntimeError(
            "Feature cache does not match the current training split. "
            "Use seed=42 or regenerate diagnostics for this split."
        )

    count_overrides = DEFAULT_SCRIPT_COUNTS[dataset]
    script_counts = [int(count_overrides.get(label, 1)) for label in label_names]
    max_scripts = max(script_counts)
    hidden_dim = encoder_features.shape[1]
    initial = np.zeros(
        (len(label_names), max_scripts, hidden_dim),
        dtype=np.float32,
    )
    details: dict[str, Any] = {}
    for class_id, label in enumerate(label_names):
        indices = np.flatnonzero(cached_labels == class_id)
        class_features = encoder_features[indices]
        count = script_counts[class_id]
        assignments = fit_cluster_assignments(class_features, count, seed)
        centers = []
        cluster_sizes = []
        for script_id in range(count):
            members = class_features[assignments == script_id]
            center = normalize_rows(members.mean(axis=0, keepdims=True))[0]
            initial[class_id, script_id] = center
            centers.append(center)
            cluster_sizes.append(int(len(members)))
        class_mean = normalize_rows(class_features.mean(axis=0, keepdims=True))[0]
        for script_id in range(count, max_scripts):
            initial[class_id, script_id] = class_mean
        details[label] = {
            "count": count,
            "cluster_sizes": cluster_sizes,
            "cluster_fractions": [
                size / len(class_features) for size in cluster_sizes
            ],
        }
    return torch.tensor(initial), script_counts, details


class FSPSCLClassifier(nn.Module):
    def __init__(
        self,
        pretrained_model: str,
        num_classes: int,
        dropout: float,
        cache_dir: Path,
        local_files_only: bool,
        description_input_ids: torch.Tensor,
        description_attention_mask: torch.Tensor,
        initial_centroids: torch.Tensor,
        script_counts: Sequence[int],
        temperature: float,
        fusion_weight: float,
        fusion_gate_init: float,
        fusion_gate_max: float,
        centroid_momentum: float,
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
        self.fusion_weight = fusion_weight
        self.fusion_gate_max = fusion_gate_max
        self.centroid_momentum = centroid_momentum
        self.max_scripts = initial_centroids.size(1)
        if not 0.0 < fusion_gate_init < fusion_gate_max <= 1.0:
            raise ValueError(
                "fusion_gate_init and fusion_gate_max must satisfy "
                "0 < fusion_gate_init < fusion_gate_max <= 1."
            )
        self.fusion_gate = nn.Linear(hidden_dim, num_classes)
        nn.init.zeros_(self.fusion_gate.weight)
        raw_gate_init = fusion_gate_init / fusion_gate_max
        nn.init.constant_(
            self.fusion_gate.bias,
            math.log(raw_gate_init / (1.0 - raw_gate_init)),
        )
        expected_description_shape = (num_classes, self.max_scripts)
        if description_input_ids.shape[:2] != expected_description_shape:
            raise ValueError(
                "Script description tensor must have shape "
                f"[{num_classes}, {self.max_scripts}, sequence_length], got "
                f"{tuple(description_input_ids.shape)}."
            )

        counts = torch.tensor(script_counts, dtype=torch.long)
        active_mask = (
            torch.arange(self.max_scripts).unsqueeze(0)
            < counts.unsqueeze(1)
        )
        self.register_buffer("description_input_ids", description_input_ids)
        self.register_buffer(
            "description_attention_mask",
            description_attention_mask,
        )
        self.register_buffer("script_counts", counts)
        self.register_buffer("active_script_mask", active_mask)
        self.register_buffer(
            "centroid_prototypes",
            F.normalize(initial_centroids.float(), p=2, dim=-1),
        )
        self.register_buffer(
            "initial_centroid_anchors",
            F.normalize(initial_centroids.float().clone(), p=2, dim=-1),
        )
        self.register_buffer(
            "semantic_prototypes",
            torch.zeros_like(initial_centroids, dtype=torch.float),
        )
        self.free_prototypes = nn.Parameter(
            F.normalize(initial_centroids.float().clone(), p=2, dim=-1)
        )
        gate_init = torch.tensor([0.0, 1.0, 0.5], dtype=torch.float)
        self.gate_logits = nn.Parameter(
            gate_init.view(1, 1, 3).expand(
                num_classes,
                self.max_scripts,
                3,
            ).clone()
        )

    def encode_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state[:, 0]
        script_features = F.normalize(hidden.detach(), p=2, dim=-1)
        compact_features = F.normalize(self.projector(hidden), p=2, dim=-1)
        return script_features, compact_features

    @torch.no_grad()
    def refresh_semantic_prototypes(self, momentum: float) -> None:
        if not 0.0 <= momentum < 1.0:
            raise ValueError(
                "semantic anchor momentum must satisfy 0 <= momentum < 1."
            )
        was_training = self.encoder.training
        self.encoder.eval()
        description_shape = self.description_input_ids.shape
        flat_input_ids = self.description_input_ids.reshape(
            -1,
            description_shape[-1],
        )
        flat_attention_mask = self.description_attention_mask.reshape(
            -1,
            description_shape[-1],
        )
        active_indices = self.active_script_mask.reshape(-1).nonzero(
            as_tuple=False
        ).squeeze(1)
        active_description_hidden = self.encoder(
            input_ids=flat_input_ids[active_indices],
            attention_mask=flat_attention_mask[active_indices],
        ).last_hidden_state[:, 0]
        flat_description_hidden = active_description_hidden.new_zeros(
            flat_input_ids.size(0),
            active_description_hidden.size(-1),
        ).index_copy(
            0,
            active_indices,
            active_description_hidden,
        )
        description_hidden = flat_description_hidden.reshape(
            description_shape[0],
            description_shape[1],
            -1,
        )
        fresh_semantics = F.normalize(
            description_hidden,
            p=2,
            dim=-1,
        )
        active = self.active_script_mask.unsqueeze(-1)
        updated = F.normalize(
            momentum * self.semantic_prototypes
            + (1.0 - momentum) * fresh_semantics,
            p=2,
            dim=-1,
        )
        self.semantic_prototypes.copy_(
            torch.where(active, updated, self.semantic_prototypes)
        )
        if was_training:
            self.encoder.train()

    def compose_prototypes(self) -> tuple[torch.Tensor, torch.Tensor]:
        description_sources = F.normalize(
            self.semantic_prototypes,
            p=2,
            dim=-1,
        )
        centroid_sources = F.normalize(
            self.centroid_prototypes,
            p=2,
            dim=-1,
        )
        free_sources = F.normalize(self.free_prototypes, p=2, dim=-1)
        sources = torch.stack(
            [description_sources, centroid_sources, free_sources],
            dim=2,
        )
        gates = torch.softmax(self.gate_logits, dim=-1)
        prototypes = F.normalize(
            (gates.unsqueeze(-1) * sources).sum(dim=2),
            p=2,
            dim=-1,
        )
        return prototypes, gates

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        script_features, compact_features = self.encode_hidden(
            input_ids,
            attention_mask,
        )
        prototypes, gates = self.compose_prototypes()
        slot_cosine = torch.einsum(
            "bd,ckd->bck",
            script_features,
            prototypes,
        )
        masked_slot_logits = (slot_cosine / self.temperature).masked_fill(
            ~self.active_script_mask.unsqueeze(0),
            -1e4,
        )
        log_counts = self.script_counts.float().log().unsqueeze(0)
        prototype_logits = torch.logsumexp(
            masked_slot_logits,
            dim=-1,
        ) - log_counts
        class_cosine_scores = self.temperature * prototype_logits
        classifier_logits = self.classifier(self.dropout(compact_features))
        fusion_gate = self.fusion_gate_max * torch.sigmoid(
            self.fusion_gate(compact_features)
        )
        final_logits = (
            classifier_logits
            + self.fusion_weight * fusion_gate * prototype_logits
        )
        active_weights = self.active_script_mask.float().unsqueeze(-1)
        class_prototypes = F.normalize(
            (prototypes * active_weights).sum(dim=1)
            / self.script_counts.float().unsqueeze(-1),
            p=2,
            dim=-1,
        )
        return {
            "logits": final_logits,
            "classifier_logits": classifier_logits,
            "prototype_logits": prototype_logits,
            "class_cosine_scores": class_cosine_scores,
            "slot_cosine": slot_cosine,
            "script_features": script_features,
            "compact_features": compact_features,
            "prototypes": prototypes,
            "class_prototypes": class_prototypes,
            "gates": gates,
            "fusion_gate": fusion_gate,
        }

    @torch.no_grad()
    def update_centroids(
        self,
        script_features: torch.Tensor,
        labels: torch.Tensor,
        slot_cosine: torch.Tensor,
    ) -> None:
        features = script_features.detach().float()
        similarities = slot_cosine.detach().float()
        for class_id in labels.unique().tolist():
            class_id = int(class_id)
            sample_mask = labels == class_id
            count = int(self.script_counts[class_id])
            assignments = similarities[sample_mask, class_id, :count].argmax(
                dim=1
            )
            class_features = features[sample_mask]
            for script_id in range(count):
                members = class_features[assignments == script_id]
                if len(members) == 0:
                    continue
                batch_center = F.normalize(
                    members.mean(dim=0),
                    p=2,
                    dim=0,
                )
                updated = (
                    self.centroid_momentum
                    * self.centroid_prototypes[class_id, script_id]
                    + (1.0 - self.centroid_momentum) * batch_center
                )
                self.centroid_prototypes[class_id, script_id].copy_(
                    F.normalize(updated, p=2, dim=0)
                )


class FSPSCLCriterion(nn.Module):
    def __init__(
        self,
        class_counts: Sequence[int],
        max_epochs: int,
        classification_weight: float,
        script_loss_weight: float,
        similarity_gamma: float,
        ldam_max_margin: float,
        rival_weight: float,
        rival_base_margin: float,
        rival_similarity_scale: float,
        diversity_weight: float,
        diversity_max_similarity: float,
        alignment_weight: float,
        gate_regularization_weight: float,
        active_script_mask: torch.Tensor,
    ) -> None:
        super().__init__()
        self.max_epochs = max_epochs
        self.classification_weight = classification_weight
        self.script_loss_weight = script_loss_weight
        self.similarity_gamma = similarity_gamma
        self.rival_weight = rival_weight
        self.rival_base_margin = rival_base_margin
        self.rival_similarity_scale = rival_similarity_scale
        self.diversity_weight = diversity_weight
        self.diversity_max_similarity = diversity_max_similarity
        self.alignment_weight = alignment_weight
        self.gate_regularization_weight = gate_regularization_weight
        counts = torch.tensor(class_counts, dtype=torch.float).clamp_min(1)
        margins = counts.pow(-0.25)
        margins = margins * (ldam_max_margin / margins.max())
        self.register_buffer("ldam_margins", margins)
        self.register_buffer(
            "active_script_mask",
            active_script_mask.clone(),
        )

    def sim_ldam_loss(
        self,
        classification_logits: torch.Tensor,
        class_prototypes: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        prototype_similarity = class_prototypes @ class_prototypes.t()
        adjusted = (
            classification_logits
            + self.similarity_gamma * prototype_similarity[labels]
        )
        adjusted = adjusted.clone()
        true_logits = classification_logits.gather(
            1,
            labels.unsqueeze(1),
        ).squeeze(1)
        true_logits = true_logits - self.ldam_margins[labels].to(
            dtype=classification_logits.dtype
        )
        adjusted.scatter_(
            1,
            labels.unsqueeze(1),
            true_logits.unsqueeze(1),
        )
        return F.cross_entropy(adjusted, labels)

    def rival_loss(
        self,
        class_scores: torch.Tensor,
        class_prototypes: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        competitors = class_scores.masked_fill(
            F.one_hot(
                labels,
                num_classes=class_scores.size(1),
            ).bool(),
            -1e4,
        )
        rival_scores, rival_ids = competitors.max(dim=1)
        true_scores = class_scores.gather(
            1,
            labels.unsqueeze(1),
        ).squeeze(1)
        pair_similarity = (
            class_prototypes[labels] * class_prototypes[rival_ids]
        ).sum(dim=-1).clamp_min(0.0)
        margins = (
            self.rival_base_margin
            + self.rival_similarity_scale * pair_similarity
        )
        return F.relu(margins + rival_scores - true_scores).mean()

    def diversity_loss(self, prototypes: torch.Tensor) -> torch.Tensor:
        penalties = []
        for class_id in range(prototypes.size(0)):
            active = prototypes[class_id][
                self.active_script_mask[class_id]
            ]
            if len(active) <= 1:
                continue
            similarity = active @ active.t()
            upper = torch.triu(
                torch.ones_like(similarity, dtype=torch.bool),
                diagonal=1,
            )
            penalties.append(
                F.relu(
                    similarity[upper] - self.diversity_max_similarity
                ).square().mean()
            )
        if not penalties:
            return prototypes.sum() * 0.0
        return torch.stack(penalties).mean()

    def alignment_loss(
        self,
        prototypes: torch.Tensor,
        centroids: torch.Tensor,
    ) -> torch.Tensor:
        similarities = (
            prototypes * F.normalize(centroids, p=2, dim=-1)
        ).sum(dim=-1)
        return (1.0 - similarities[self.active_script_mask]).mean()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        epoch: int,
        centroids: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        script = F.cross_entropy(outputs["prototype_logits"], labels)
        sim_ldam = self.sim_ldam_loss(
            outputs["logits"],
            outputs["class_prototypes"],
            labels,
        )
        rival = self.rival_loss(
            outputs["class_cosine_scores"],
            outputs["class_prototypes"],
            labels,
        )
        diversity = self.diversity_loss(outputs["prototypes"])
        alignment = self.alignment_loss(
            outputs["prototypes"],
            centroids,
        )
        gate_regularization = outputs["fusion_gate"].square().mean()
        loss = (
            self.classification_weight * sim_ldam
            + self.script_loss_weight * script
            + self.rival_weight * rival
            + self.diversity_weight * diversity
            + self.alignment_weight * alignment
            + self.gate_regularization_weight * gate_regularization
        )
        return loss, {
            "loss_script": float(script.detach().cpu()),
            "loss_sim_ldam": float(sim_ldam.detach().cpu()),
            "loss_rival": float(rival.detach().cpu()),
            "loss_diversity": float(diversity.detach().cpu()),
            "loss_alignment": float(alignment.detach().cpu()),
            "loss_gate": float(gate_regularization.detach().cpu()),
            "script_loss_weight": float(self.script_loss_weight),
            "fusion_gate_mean": float(
                outputs["fusion_gate"].detach().mean().cpu()
            ),
            "fusion_gate_max": float(
                outputs["fusion_gate"].detach().max().cpu()
            ),
            "loss_total": float(loss.detach().cpu()),
        }


def amp_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def build_optimizer_scheduler(
    model: nn.Module,
    args: argparse.Namespace,
    total_steps: int,
) -> tuple[torch.optim.Optimizer, Any]:
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    encoder_decay, encoder_no_decay, head_params = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("encoder."):
            target = (
                encoder_no_decay
                if any(token in name for token in no_decay)
                else encoder_decay
            )
            target.append(parameter)
        else:
            head_params.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder_decay,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
            {
                "params": encoder_no_decay,
                "lr": args.lr,
                "weight_decay": 0.0,
            },
            {
                "params": head_params,
                "lr": args.head_lr,
                "weight_decay": args.weight_decay,
            },
        ]
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=max(1, total_steps),
    )
    return optimizer, scheduler


def empty_utilization(model: FSPSCLClassifier) -> torch.Tensor:
    return torch.zeros(
        model.active_script_mask.shape,
        dtype=torch.long,
    )


def utilization_to_json(
    utilization: torch.Tensor,
    label_names: Sequence[str],
    script_counts: Sequence[int],
) -> dict[str, list[int]]:
    return {
        label: utilization[class_id, : script_counts[class_id]].tolist()
        for class_id, label in enumerate(label_names)
    }


def write_run_status(
    status_path: Path,
    status: str,
    **details: Any,
) -> None:
    save_json(
        status_path,
        {
            "status": status,
            "pid": os.getpid(),
            "updated_at_unix": time.time(),
            **details,
        },
    )


def train_one_epoch(
    model: FSPSCLClassifier,
    loader: DataLoader,
    criterion: FSPSCLCriterion,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
    grad_clip: float,
    epoch: int,
    amp_enabled: bool,
    status_path: Path,
) -> tuple[float, Dict[str, float], torch.Tensor]:
    model.train()
    total_loss = 0.0
    total_parts: Counter[str] = Counter()
    utilization = empty_utilization(model)
    total_batches = len(loader)
    for batch_index, batch in enumerate(
        tqdm(loader, desc=f"train {epoch}", leave=False),
        start=1,
    ):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(
            device,
            non_blocking=True,
        )
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device, amp_enabled):
            outputs = model(input_ids, attention_mask)
            loss, parts = criterion(
                outputs,
                labels,
                epoch,
                model.initial_centroid_anchors,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        with torch.no_grad():
            true_slot_scores = outputs["slot_cosine"][
                torch.arange(len(labels), device=device),
                labels,
            ]
            true_slots = true_slot_scores.masked_fill(
                ~model.active_script_mask[labels],
                -1e4,
            ).argmax(dim=1)
            for class_id, script_id in zip(
                labels.detach().cpu().tolist(),
                true_slots.detach().cpu().tolist(),
            ):
                utilization[class_id, script_id] += 1
            model.update_centroids(
                outputs["script_features"],
                labels,
                outputs["slot_cosine"],
            )

        total_loss += loss.item() * labels.size(0)
        for key, value in parts.items():
            total_parts[key] += value * labels.size(0)
        if batch_index % 50 == 0 or batch_index == total_batches:
            write_run_status(
                status_path,
                "running",
                phase="train",
                epoch=epoch,
                target_epochs=criterion.max_epochs,
                batch=batch_index,
                total_batches=total_batches,
                epoch_percent=100.0 * batch_index / total_batches,
                running_train_loss=total_loss
                / min(batch_index * loader.batch_size, len(loader.dataset)),
            )
    size = len(loader.dataset)
    return (
        total_loss / size,
        {key: float(value / size) for key, value in total_parts.items()},
        utilization,
    )


@torch.no_grad()
def evaluate(
    model: FSPSCLClassifier,
    loader: DataLoader,
    criterion: FSPSCLCriterion,
    device: torch.device,
    epoch: int,
    amp_enabled: bool,
    status_path: Path,
    phase: str,
) -> tuple[float, Dict[str, float], list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    total_parts: Counter[str] = Counter()
    y_true: list[int] = []
    y_pred: list[int] = []
    total_batches = len(loader)
    for batch_index, batch in enumerate(
        tqdm(loader, desc=phase, leave=False),
        start=1,
    ):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(
            device,
            non_blocking=True,
        )
        labels = batch["label"].to(device, non_blocking=True)
        with amp_context(device, amp_enabled):
            outputs = model(input_ids, attention_mask)
            loss, parts = criterion(
                outputs,
                labels,
                epoch,
                model.initial_centroid_anchors,
            )
        predictions = outputs["logits"].argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        for key, value in parts.items():
            total_parts[key] += value * labels.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())
        if batch_index % 50 == 0 or batch_index == total_batches:
            write_run_status(
                status_path,
                "running",
                phase=phase,
                epoch=epoch,
                target_epochs=criterion.max_epochs,
                batch=batch_index,
                total_batches=total_batches,
                phase_percent=100.0 * batch_index / total_batches,
            )
    size = len(loader.dataset)
    return (
        total_loss / size,
        {key: float(value / size) for key, value in total_parts.items()},
        y_true,
        y_pred,
    )


def prototype_diagnostics(
    model: FSPSCLClassifier,
    label_names: Sequence[str],
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        prototypes, gates = model.compose_prototypes()
    result: dict[str, Any] = {}
    for class_id, label in enumerate(label_names):
        count = int(model.script_counts[class_id])
        active = prototypes[class_id, :count]
        pairwise = active @ active.t()
        off_diagonal = pairwise[
            torch.triu(
                torch.ones_like(pairwise, dtype=torch.bool),
                diagonal=1,
            )
        ]
        result[label] = {
            "script_count": count,
            "gate_weights": gates[class_id, :count].cpu().tolist(),
            "pairwise_similarity": pairwise.cpu().tolist(),
            "mean_off_diagonal_similarity": (
                float(off_diagonal.mean().cpu())
                if len(off_diagonal)
                else None
            ),
        }
    return result


def main() -> None:
    global ACTIVE_STATUS_PATH
    args = parse_args()
    resolve_paths(args)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    ACTIVE_STATUS_PATH = args.output_dir / "run_status.json"
    write_run_status(
        ACTIVE_STATUS_PATH,
        "starting",
        phase="initialization",
        target_epochs=args.epochs,
        output_dir=str(args.output_dir),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError(
            "CUDA is unavailable. Use a CUDA-enabled PyTorch installation or "
            "pass --allow-cpu explicitly."
        )
    amp_enabled = bool(args.amp and device.type == "cuda")
    print(
        json.dumps(
            {
                "model": MODEL_NAME,
                "device": str(device),
                "gpu": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else None
                ),
                "amp": amp_enabled,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    texts, raw_labels, groups = load_dataset(args)
    if not texts:
        raise RuntimeError(f"No data loaded from {args.data_dir}")
    label_names = sorted(set(raw_labels))
    label_to_id = {
        label: index for index, label in enumerate(label_names)
    }
    id_to_label = {
        index: label for label, index in label_to_id.items()
    }
    labels = [label_to_id[label] for label in raw_labels]
    split_manifest = None
    if args.split_manifest is not None:
        split, split_manifest = split_from_manifest(
            args.split_manifest,
            texts,
            labels,
            groups,
            expected_dataset=args.dataset,
        )
    elif groups is None:
        split = make_splits(texts, labels, args.split_seed)
    else:
        split = make_group_splits(
            texts,
            labels,
            groups,
            args.split_seed,
        )
    (
        train_texts,
        valid_texts,
        test_texts,
        train_labels,
        valid_labels,
        test_labels,
    ) = split

    initial_centroids, script_counts, init_details = (
        build_initial_script_prototypes(
            args.feature_cache,
            label_names,
            train_labels,
            args.dataset,
            args.seed,
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model,
        cache_dir=args.cache_dir,
        local_files_only=not args.allow_download,
    )
    description_rows, script_descriptions = build_script_descriptions(
        args.dataset,
        label_names,
        script_counts,
    )
    flat_descriptions = [
        description
        for row in description_rows
        for description in row
    ]
    encoded_descriptions = tokenizer(
        flat_descriptions,
        max_length=args.max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    description_input_ids = encoded_descriptions["input_ids"].reshape(
        len(label_names),
        initial_centroids.size(1),
        -1,
    )
    description_attention_mask = encoded_descriptions[
        "attention_mask"
    ].reshape(
        len(label_names),
        initial_centroids.size(1),
        -1,
    )
    model = FSPSCLClassifier(
        pretrained_model=args.pretrained_model,
        num_classes=len(label_names),
        dropout=args.dropout,
        cache_dir=args.cache_dir,
        local_files_only=not args.allow_download,
        description_input_ids=description_input_ids,
        description_attention_mask=description_attention_mask,
        initial_centroids=initial_centroids,
        script_counts=script_counts,
        temperature=args.temperature,
        fusion_weight=args.fusion_weight,
        fusion_gate_init=args.fusion_gate_init,
        fusion_gate_max=args.fusion_gate_max,
        centroid_momentum=args.centroid_momentum,
    ).to(device)

    if not args.warm_start.exists():
        raise FileNotFoundError(f"Warm-start checkpoint not found: {args.warm_start}")
    warm_start = torch.load(
        args.warm_start,
        map_location=device,
        weights_only=False,
    )
    if warm_start.get("label_to_id") != label_to_id:
        raise RuntimeError(
            "Warm-start label mapping differs from the current dataset split."
        )
    warm_start_state = {
        key: value
        for key, value in warm_start["model_state_dict"].items()
        if key not in {
            "description_input_ids",
            "description_attention_mask",
        }
    }
    incompatible = model.load_state_dict(warm_start_state, strict=False)
    allowed_missing_prefixes = (
        "description_input_ids",
        "description_attention_mask",
        "script_counts",
        "active_script_mask",
        "centroid_prototypes",
        "initial_centroid_anchors",
        "semantic_prototypes",
        "free_prototypes",
        "gate_logits",
        "fusion_gate",
    )
    unexpected_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(allowed_missing_prefixes)
    ]
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected warm-start mismatch: missing={unexpected_missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    # Loading the PSCL state must not replace the data-driven script initialization.
    model.centroid_prototypes.copy_(initial_centroids.to(device))
    model.initial_centroid_anchors.copy_(initial_centroids.to(device))
    model.free_prototypes.data.copy_(initial_centroids.to(device))
    model.refresh_semantic_prototypes(momentum=0.0)

    train_dataset = FraudTextDataset(
        train_texts,
        train_labels,
        tokenizer,
        args.max_len,
    )
    valid_dataset = FraudTextDataset(
        valid_texts,
        valid_labels,
        tokenizer,
        args.max_len,
    )
    test_dataset = FraudTextDataset(
        test_texts,
        test_labels,
        tokenizer,
        args.max_len,
    )
    sampler = (
        make_weighted_sampler(train_labels, len(label_names))
        if args.weighted_sampler
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    class_counts = torch.bincount(
        torch.tensor(train_labels),
        minlength=len(label_names),
    ).tolist()
    criterion = FSPSCLCriterion(
        class_counts=class_counts,
        max_epochs=args.epochs,
        classification_weight=args.classification_weight,
        script_loss_weight=args.script_loss_weight,
        similarity_gamma=args.similarity_gamma,
        ldam_max_margin=args.ldam_max_margin,
        rival_weight=0.0 if args.disable_rival else args.rival_weight,
        rival_base_margin=args.rival_base_margin,
        rival_similarity_scale=args.rival_similarity_scale,
        diversity_weight=(
            0.0 if args.disable_diversity else args.diversity_weight
        ),
        diversity_max_similarity=args.diversity_max_similarity,
        alignment_weight=(
            0.0 if args.disable_alignment else args.alignment_weight
        ),
        gate_regularization_weight=args.gate_regularization_weight,
        active_script_mask=model.active_script_mask,
    ).to(device)
    total_steps = max(1, len(train_loader) * args.epochs)
    optimizer, scheduler = build_optimizer_scheduler(
        model,
        args,
        total_steps,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    start_epoch = 1
    history: list[dict[str, Any]] = []
    best_score = -1.0
    epochs_without_improvement = 0
    state = load_checkpoint(args.resume, device)
    if state is not None:
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        if state.get("scaler_state_dict"):
            scaler.load_state_dict(state["scaler_state_dict"])
        history = list(state.get("history", []))
        best_score = float(state.get("best_score", -1.0))
        epochs_without_improvement = int(
            state.get("epochs_without_improvement", 0)
        )
        start_epoch = int(state["epoch"]) + 1
        print(
            json.dumps(
                {
                    "resume_from": str(args.resume),
                    "next_epoch": start_epoch,
                    "best_score": best_score,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    parameter_info = count_parameters(model)
    configuration = {
        "model_name": MODEL_NAME,
        "model_version": (
            "2.0-compatible-reconstruction"
            if args.compat_v2
            else MODEL_VERSION
        ),
        "created_at_unix": time.time(),
        "args": json_safe_args(args),
        "warm_start_best_epoch": warm_start.get("best_epoch"),
        "warm_start_best_score": warm_start.get("best_score"),
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "script_descriptions": script_descriptions,
        "script_counts": dict(zip(label_names, script_counts)),
        "script_initialization": init_details,
        "split_sizes": {
            "train": len(train_dataset),
            "valid": len(valid_dataset),
            "test": len(test_dataset),
        },
        "group_aware_split": groups is not None,
        "matched_protocol": {
            "split_seed": args.split_seed,
            "training_seed": args.seed,
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
            "compat_v2": args.compat_v2,
            "best_checkpoint_selected_on": args.save_best_by,
            "test_used_for_model_selection": False,
        },
        "test_used_for_model_selection": False,
        **parameter_info,
    }
    save_json(args.output_dir / "experiment_config.json", configuration)
    save_json(
        args.output_dir / "label_mapping.json",
        {"label_to_id": label_to_id, "id_to_label": id_to_label},
    )
    save_json(
        args.output_dir / "script_prototype_config.json",
        {
            "script_counts": dict(zip(label_names, script_counts)),
            "script_descriptions": script_descriptions,
            "initialization": init_details,
        },
    )

    def checkpoint_payload(epoch: int) -> Dict[str, Any]:
        return {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "args": json_safe_args(args),
            "label_to_id": label_to_id,
            "id_to_label": id_to_label,
            "script_counts": dict(zip(label_names, script_counts)),
            "saved_at_unix": time.time(),
        }

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "split_sizes": configuration["split_sizes"],
                "script_counts": configuration["script_counts"],
                **parameter_info,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        model.refresh_semantic_prototypes(
            momentum=args.semantic_anchor_momentum
        )
        write_run_status(
            ACTIVE_STATUS_PATH,
            "running",
            phase="train",
            epoch=epoch,
            target_epochs=args.epochs,
            batch=0,
            total_batches=len(train_loader),
            epoch_percent=0.0,
        )
        train_loss, train_parts, utilization = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            scaler,
            device,
            args.grad_clip,
            epoch,
            amp_enabled,
            ACTIVE_STATUS_PATH,
        )
        valid_loss, valid_parts, valid_true, valid_pred = evaluate(
            model,
            valid_loader,
            criterion,
            device,
            epoch,
            amp_enabled,
            ACTIVE_STATUS_PATH,
            "validation",
        )
        valid_metrics = summarize_metrics(valid_true, valid_pred)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_loss_parts": train_parts,
            "train_script_utilization": utilization_to_json(
                utilization,
                label_names,
                script_counts,
            ),
            "valid_loss": valid_loss,
            "valid_loss_parts": valid_parts,
            "valid_accuracy": valid_metrics["accuracy"],
            "valid_macro_precision": valid_metrics["macro_precision"],
            "valid_macro_recall": valid_metrics["macro_recall"],
            "valid_macro_f1": valid_metrics["macro_f1"],
            "valid_weighted_precision": valid_metrics[
                "weighted_precision"
            ],
            "valid_weighted_recall": valid_metrics["weighted_recall"],
            "valid_weighted_f1": valid_metrics["weighted_f1"],
            "learning_rates": current_lrs(optimizer),
            "prototype_diagnostics": prototype_diagnostics(
                model,
                label_names,
            ),
            "saved_at_unix": time.time(),
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        score = float(row[args.save_best_by])
        if score > best_score:
            best_score = score
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": json_safe_args(args),
                    "label_to_id": label_to_id,
                    "id_to_label": id_to_label,
                    "script_counts": dict(
                        zip(label_names, script_counts)
                    ),
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
                },
            )
        else:
            epochs_without_improvement += 1
        save_checkpoint(
            args.output_dir / "latest_checkpoint.pt",
            checkpoint_payload(epoch),
        )
        save_json(args.output_dir / "training_history.json", history)
        write_run_status(
            ACTIVE_STATUS_PATH,
            "running",
            phase="epoch_saved",
            epoch=epoch,
            target_epochs=args.epochs,
            valid_macro_f1=valid_metrics["macro_f1"],
            best_score=best_score,
            latest_checkpoint=str(
                args.output_dir / "latest_checkpoint.pt"
            ),
            epochs_without_improvement=epochs_without_improvement,
        )
        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            write_run_status(
                ACTIVE_STATUS_PATH,
                "running",
                phase="early_stopping",
                epoch=epoch,
                target_epochs=args.epochs,
                best_score=best_score,
                patience=args.early_stopping_patience,
            )
            break

    best = torch.load(
        args.output_dir / "best_model.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best["model_state_dict"])
    best_epoch = int(best["best_epoch"])
    test_loss, test_parts, test_true, test_pred = evaluate(
        model,
        test_loader,
        criterion,
        device,
        best_epoch,
        amp_enabled,
        ACTIVE_STATUS_PATH,
        "test",
    )
    target_names = [
        id_to_label[index] for index in range(len(label_names))
    ]
    report = classification_report(
        test_true,
        test_pred,
        target_names=target_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "dataset": args.dataset,
        "best_epoch": best_epoch,
        "best_score": best["best_score"],
        "test_loss": test_loss,
        "test_loss_parts": test_parts,
        **summarize_metrics(test_true, test_pred),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            test_true,
            test_pred,
        ).tolist(),
        "script_counts": dict(zip(label_names, script_counts)),
        "prototype_diagnostics": prototype_diagnostics(
            model,
            label_names,
        ),
        **parameter_info,
    }
    save_json(args.output_dir / "metrics.json", metrics)
    save_json(
        args.output_dir / "final_summary.json",
        {
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "dataset": args.dataset,
            "completed_epochs": len(history),
            "early_stopped": len(history) < args.epochs,
            "best_epoch": best_epoch,
            "validation_best_score": best["best_score"],
            "test_accuracy": metrics["accuracy"],
            "test_macro_f1": metrics["macro_f1"],
            "test_weighted_f1": metrics["weighted_f1"],
            "script_counts": metrics["script_counts"],
            **parameter_info,
        },
    )
    write_run_status(
        ACTIVE_STATUS_PATH,
        "completed",
        phase="done",
        completed_epochs=len(history),
        early_stopped=len(history) < args.epochs,
        best_epoch=best_epoch,
        best_validation_score=best["best_score"],
        test_accuracy=metrics["accuracy"],
        test_macro_f1=metrics["macro_f1"],
        output_dir=str(args.output_dir),
    )
    print(
        json.dumps(
            {
                "done": True,
                "best_epoch": best_epoch,
                "best_validation_score": best["best_score"],
                "test_accuracy": metrics["accuracy"],
                "test_macro_f1": metrics["macro_f1"],
                "test_weighted_f1": metrics["weighted_f1"],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        if ACTIVE_STATUS_PATH is not None:
            write_run_status(
                ACTIVE_STATUS_PATH,
                "interrupted",
                phase="terminated_by_user",
                error="KeyboardInterrupt",
            )
        raise
    except Exception as exc:
        if ACTIVE_STATUS_PATH is not None:
            write_run_status(
                ACTIVE_STATUS_PATH,
                "failed",
                phase="exception",
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
        raise
