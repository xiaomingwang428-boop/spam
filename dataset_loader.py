"""Dataset loading helpers for email classification experiments.

This module expects a local ``archive.zip`` containing the six source CSV files.
The dataset itself is not included in this repository.
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SOURCE_FILES = (
    "CEAS_08.csv",
    "Enron.csv",
    "Ling.csv",
    "Nazario.csv",
    "Nigerian_Fraud.csv",
    "SpamAssasin.csv",
)
TEXT_FIELDS = ("sender", "date", "subject", "body")


def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).replace("\x00", " ").split())


def combine_email(row: pd.Series) -> str:
    labels = {"sender": "Sender", "date": "Date", "subject": "Subject", "body": "Body"}
    parts = []
    for field in TEXT_FIELDS:
        value = clean(row.get(field, ""))
        if value:
            parts.append(f"{labels[field]}: {value}")
    return "\n".join(parts)


def load_sources(archive: Path, max_chars: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        missing = set(SOURCE_FILES) - names
        if missing:
            raise ValueError(f"Archive is missing source files: {sorted(missing)}")
        for name in SOURCE_FILES:
            with zf.open(name) as stream:
                frame = pd.read_csv(stream, low_memory=False, on_bad_lines="skip")
            if "label" not in frame.columns:
                raise ValueError(f"{name} has no label column")
            frame["text_combined"] = frame.apply(combine_email, axis=1).str.slice(0, max_chars)
            frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
            frame = frame.loc[frame["label"].isin([0, 1]), ["text_combined", "label"]]
            frame = frame.loc[frame["text_combined"].str.len() > 0].copy()
            frame["source"] = name
            frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data["label"] = data["label"].astype(np.int8)
    data["text_hash"] = data["text_combined"].map(
        lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
    )
    conflicts = data.groupby("text_hash")["label"].nunique()
    conflict_hashes = set(conflicts[conflicts > 1].index)
    if conflict_hashes:
        data = data.loc[~data["text_hash"].isin(conflict_hashes)]
    return data.drop_duplicates("text_hash", keep="first").reset_index(drop=True)


def sample_data(data: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    if limit is None or limit >= len(data):
        return data
    sampled, _ = train_test_split(
        data, train_size=limit, stratify=data["label"], random_state=seed
    )
    return sampled.reset_index(drop=True)
