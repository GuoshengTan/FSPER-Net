"""Validated custom dataset splits shared by cross-fitting experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


ROLE_NAMES = ("train", "valid", "test")


def dataset_fingerprint(
    texts: Sequence[str],
    labels: Sequence[int],
    groups: Sequence[str] | None = None,
) -> str:
    if len(texts) != len(labels):
        raise ValueError("Texts and labels must have identical lengths.")
    if groups is not None and len(groups) != len(texts):
        raise ValueError("Groups and texts must have identical lengths.")
    digest = hashlib.sha256()
    for index, (text, label) in enumerate(zip(texts, labels)):
        group = groups[index] if groups is not None else ""
        digest.update(str(index).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(label).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(group).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(text).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported split manifest schema in {path}: "
            f"{payload.get('schema_version')}"
        )
    return payload


def validate_manifest(
    manifest: dict[str, Any],
    texts: Sequence[str],
    labels: Sequence[int],
    groups: Sequence[str] | None = None,
) -> dict[str, list[int]]:
    expected_size = int(manifest.get("sample_count", -1))
    if expected_size != len(texts):
        raise ValueError(
            f"Split manifest sample count mismatch: {expected_size} != "
            f"{len(texts)}"
        )
    expected_fingerprint = manifest.get("dataset_fingerprint")
    actual_fingerprint = dataset_fingerprint(texts, labels, groups)
    if expected_fingerprint != actual_fingerprint:
        raise ValueError(
            "Split manifest fingerprint mismatch. The dataset contents or "
            "ordering changed after the manifest was generated."
        )

    role_indices: dict[str, list[int]] = {}
    seen: set[int] = set()
    for role in ROLE_NAMES:
        indices = [int(index) for index in manifest["indices"][role]]
        if not indices:
            raise ValueError(f"Split role {role!r} is empty.")
        if len(indices) != len(set(indices)):
            raise ValueError(f"Split role {role!r} contains duplicate indices.")
        invalid = [
            index for index in indices if index < 0 or index >= len(texts)
        ]
        if invalid:
            raise ValueError(
                f"Split role {role!r} contains out-of-range indices: "
                f"{invalid[:5]}"
            )
        overlap = seen.intersection(indices)
        if overlap:
            raise ValueError(
                f"Split role {role!r} overlaps an earlier role at indices "
                f"{sorted(overlap)[:5]}"
            )
        seen.update(indices)
        role_indices[role] = indices

    if groups is not None:
        role_groups = {
            role: {groups[index] for index in indices}
            for role, indices in role_indices.items()
        }
        for left_index, left in enumerate(ROLE_NAMES):
            for right in ROLE_NAMES[left_index + 1 :]:
                overlap = role_groups[left].intersection(role_groups[right])
                if overlap:
                    raise ValueError(
                        f"Group leakage between {left!r} and {right!r}: "
                        f"{sorted(overlap)[:5]}"
                    )
    return role_indices


def split_from_manifest(
    path: Path,
    texts: Sequence[str],
    labels: Sequence[int],
    groups: Sequence[str] | None = None,
    expected_dataset: str | None = None,
) -> tuple[
    tuple[
        list[str],
        list[str],
        list[str],
        list[int],
        list[int],
        list[int],
    ],
    dict[str, Any],
]:
    manifest = load_manifest(path)
    if (
        expected_dataset is not None
        and manifest.get("dataset") != expected_dataset
    ):
        raise ValueError(
            "Split manifest dataset mismatch: "
            f"{manifest.get('dataset')!r} != {expected_dataset!r}"
        )
    role_indices = validate_manifest(manifest, texts, labels, groups)

    def take(values: Sequence[Any], indices: Sequence[int]) -> list[Any]:
        return [values[index] for index in indices]

    split = (
        take(texts, role_indices["train"]),
        take(texts, role_indices["valid"]),
        take(texts, role_indices["test"]),
        take(labels, role_indices["train"]),
        take(labels, role_indices["valid"]),
        take(labels, role_indices["test"]),
    )
    return split, manifest
