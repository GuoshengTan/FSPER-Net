"""Evaluate the released checkpoint on the original FGRC-SCD test split, without training."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from sklearn.metrics import accuracy_score, f1_score

from predict import DEFAULT_MODEL_DIR, Predictor
import run_fgrc_scd as protocol
from split_manifest import split_from_manifest
from train_published_fraud_models import load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--output', type=Path, default=Path('outputs/checkpoint_evaluation.json'))
    args = parser.parse_args()
    predictor = Predictor(args.model_dir, args.device)
    reference = predictor.config['reference_test']
    data_args = argparse.Namespace(dataset='fgrc_scd', data_dir=args.data_dir,
                                   max_samples_per_class=0, seed=reference['split_seed'])
    texts, raw_labels, groups = load_dataset(data_args)
    mapping = predictor.config['label_to_id']
    labels = [mapping[label] for label in raw_labels]
    protocol.DATA_DIR = args.data_dir
    with tempfile.TemporaryDirectory() as directory:
        manifest_path = protocol.build_fixed_split_manifest(Path(directory), reference['split_seed'])
        split, manifest = split_from_manifest(manifest_path, texts, labels, groups, 'fgrc_scd')
    index_hash = hashlib.sha256(json.dumps(manifest['indices']['test'], separators=(',', ':')).encode()).hexdigest()
    if (manifest['dataset_fingerprint'] != reference['dataset_fingerprint'] or
            index_hash != reference['test_indices_sha256']):
        raise ValueError('Dataset or test split differs from the released checkpoint protocol.')
    test_texts, test_labels = split[2], split[5]
    if len(test_texts) != reference['samples']:
        raise ValueError('Test sample count mismatch.')
    predictions = []
    batch_size = reference['batch_size']
    for start in range(0, len(test_texts), batch_size):
        rows = predictor.predict(test_texts[start:start + batch_size], batch_size, truncate=True)
        predictions.extend(row['label_id'] for row in rows)
        if start % (batch_size * 32) == 0:
            print(f'Evaluated {len(predictions)}/{len(test_texts)}', flush=True)
    result = {
        'dataset': 'FGRC-SCD', 'training_seed': 42, 'test_samples': len(test_texts),
        'device': str(predictor.device),
        'precision': 'cuda_float16_autocast' if predictor.device.type == 'cuda' else 'float32',
        'accuracy': accuracy_score(test_labels, predictions),
        'macro_f1': f1_score(test_labels, predictions, average='macro'),
        'dataset_fingerprint': manifest['dataset_fingerprint'], 'test_indices_sha256': index_hash,
        'source_checkpoint_sha256': predictor.config['source_checkpoint_sha256'],
        'scope': 'single_seed_checkpoint_not_multi_seed_mean',
    }
    result['reference_accuracy'] = reference['accuracy']
    result['reference_macro_f1'] = reference['macro_f1']
    result['matches_reference'] = all(abs(result[key] - reference[key]) < 1e-12 for key in ('accuracy', 'macro_f1'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    if not result['matches_reference']:
        raise SystemExit('Metrics differ from the recorded checkpoint result; inspect precision/environment.')


if __name__ == '__main__':
    main()
