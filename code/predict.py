"""Offline inference from the released, trained FGRC-SCD checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
from transformers import AutoConfig, AutoTokenizer

from train_sparse_routed_pscl import SparseRoutedFSPSCLClassifier

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / 'models' / 'fgrc_scd_seed42'


def verify_bundle(directory):
    directory = Path(directory).resolve()
    manifest = directory / 'SHA256SUMS.txt'
    if not manifest.is_file():
        raise FileNotFoundError(f'Missing checksum manifest: {manifest}')
    names = set()
    for line in manifest.read_text(encoding='ascii').splitlines():
        expected, name = line.split('  ', 1)
        path = (directory / name).resolve()
        if not path.is_relative_to(directory):
            raise ValueError('Checksum path is outside the model directory.')
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected.lower():
            raise ValueError(f'Checksum mismatch: {name}. Run git lfs pull if the weight is an LFS pointer.')
        names.add(name)
    required = {'weights.index.json', 'inference_config.json', 'tokenizer/config.json',
                'tokenizer/tokenizer_config.json', 'tokenizer/tokenizer.json',
                'tokenizer/vocab.txt', 'tokenizer/special_tokens_map.json',
                'tokenizer/added_tokens.json'}
    if not required.issubset(names):
        raise ValueError('Checksum manifest is incomplete.')
    index = json.loads((directory / 'weights.index.json').read_text(encoding='utf-8'))
    if not set(index['weight_map'].values()).issubset(names):
        raise ValueError('One or more weight shards are missing from the checksum manifest.')


def load_released_state(directory):
    index = json.loads((directory / 'weights.index.json').read_text(encoding='utf-8'))
    state = {}
    for name in sorted(set(index['weight_map'].values())):
        shard = torch.load(directory / name, map_location='cpu', weights_only=True)
        overlap = set(state).intersection(shard)
        if overlap:
            raise ValueError(f'Duplicate tensors across weight shards: {sorted(overlap)[:3]}')
        state.update(shard)
    if set(state) != set(index['weight_map']):
        raise ValueError('Weight index and loaded tensors do not match.')
    return state


class Predictor:
    def __init__(self, model_dir=DEFAULT_MODEL_DIR, device='auto'):
        self.directory = Path(model_dir).resolve()
        verify_bundle(self.directory)
        self.config = json.loads((self.directory / 'inference_config.json').read_text(encoding='utf-8'))
        if self.config['schema_version'] != 1:
            raise ValueError('Unsupported inference configuration.')
        mapping = self.config['label_to_id']
        if sorted(mapping.values()) != list(range(len(mapping))):
            raise ValueError('Label IDs must be contiguous.')
        self.labels = sorted(mapping, key=mapping.get)
        self.device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device)
        if self.device.type == 'cpu':
            torch.set_num_threads(min(4, os.cpu_count() or 1))
        local = self.directory / 'tokenizer'
        self.tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True)
        encoder_config = AutoConfig.from_pretrained(local, local_files_only=True)
        state = load_released_state(self.directory)
        self.model = SparseRoutedFSPSCLClassifier(
            pretrained_model=str(local), num_classes=len(self.labels),
            cache_dir=local, local_files_only=True, encoder_config=encoder_config,
            description_input_ids=state['description_input_ids'],
            description_attention_mask=state['description_attention_mask'],
            initial_centroids=state['centroid_prototypes'],
            script_counts=state['script_counts'].tolist(),
            use_router=True, fixed_fusion=False, **self.config['model_parameters'],
        )
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, texts, batch_size=16, truncate=False):
        if batch_size < 1:
            raise ValueError('batch_size must be positive.')
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError('Provide nonempty text strings.')
        texts = [' '.join(text.strip().split()) for text in texts]
        limit = self.config['max_length']
        lengths = [len(self.tokenizer.encode(text, truncation=False)) for text in texts]
        if not truncate and any(length > limit for length in lengths):
            raise ValueError(f'Input exceeds {limit} tokens. Shorten it or explicitly use --truncate.')
        result = []
        for start in range(0, len(texts), batch_size):
            encoded = self.tokenizer(texts[start:start + batch_size], max_length=limit,
                                     padding='max_length', truncation=True, return_tensors='pt')
            with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                enabled=self.device.type == 'cuda'):
                logits = self.model(encoded['input_ids'].to(self.device),
                                    encoded['attention_mask'].to(self.device))['logits']
            for offset, scores in enumerate(logits.float().softmax(-1).cpu().tolist()):
                index = max(range(len(scores)), key=scores.__getitem__)
                result.append({'label_id': index, 'label': self.labels[index],
                               'score': scores[index], 'scores': dict(zip(self.labels, scores)),
                               'truncated': lengths[start + offset] > limit})
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--batch-size', type=int, default=16)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--text')
    source.add_argument('--text-file', type=Path, help='UTF-8 file, one text per line')
    parser.add_argument('--truncate', action='store_true')
    args = parser.parse_args()
    texts = [args.text] if args.text is not None else args.text_file.read_text(encoding='utf-8-sig').splitlines()
    predictor = Predictor(args.model_dir, args.device)
    print(json.dumps(predictor.predict(texts, args.batch_size, args.truncate), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
