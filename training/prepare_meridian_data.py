"""
training/prepare_meridian_data.py
===================================
Generates the placeholder train/val parquet files verl's data loader
expects.

This looks strange the first time you read it, so it's worth explaining
plainly: verl's core trainer was built around a fixed dataset of prompts
(classic RLHF style). Every agent environment in verl-agent -- including
this one -- doesn't actually use that dataset's CONTENT. The real prompt
for each step comes from MeridianEnvironmentManager, built fresh from
live environment state. All these parquet files need to provide is the
right NUMBER OF ROWS, so verl's batching produces batches of the size
data.train_batch_size / data.val_batch_size expects. Confirmed directly
from verl-agent's own data_preprocess/prepare.py: in text mode, its
"prompt" field is a literal empty string -- content is thrown away too.

Usage:
    python training/prepare_meridian_data.py --train-size 4 --val-size 4
"""

from __future__ import annotations

import argparse
import os

import pandas as pd


def build_split(split: str, size: int) -> pd.DataFrame:
    rows = []
    for idx in range(size):
        rows.append({
            "data_source": "text",
            "prompt": [{"role": "user", "content": ""}],  # discarded; real prompt comes from the env manager
            "ability": "agent",
            "extra_info": {"split": split, "index": idx},
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=4)
    parser.add_argument("--val-size", type=int, default=4)
    parser.add_argument("--out-dir", default=os.path.expanduser("~/data/meridian"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_df = build_split("train", args.train_size)
    val_df = build_split("val", args.val_size)

    train_path = os.path.join(args.out_dir, "train.parquet")
    val_path = os.path.join(args.out_dir, "val.parquet")
    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)

    print(f"Wrote {len(train_df)} placeholder rows to {train_path}")
    print(f"Wrote {len(val_df)} placeholder rows to {val_path}")