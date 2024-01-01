#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import json
from typing import Generator

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def load_jsonl(path: Path) -> Generator[dict, None, None]:
    """Load a JSONL file."""
    with path.open("r") as fin:
        yield from (json.loads(ln) for ln in fin.readlines())


def shuffle_and_measure(
    errors: list[float], fixes: list[float], k: int = 250
) -> list[float]:
    """Shuffle uncorrected and corrected sentence embeddings and compare their
    embeddings.

    Parameters
    ----------
    errors
        Uncorrected sentence embeddings
    fixes
        Corrected sentence embeddings
    k
        Number of iterations to run

    Returns
    -------
    similarities
        Mean cosine similarities for the shuffled sentence pairs
    """
    num_obs = len(errors)
    similarities = []
    for _ in range(k):
        rand_sims = []
        for err in errors:
            idx = np.random.choice(range(num_obs))
            sim = cosine_similarity([err], [fixes[idx]]).item()
            rand_sims.append(sim)
        similarities.append(rand_sims)

    return [
        np.mean([shuf[i] for shuf in similarities]) for i in range(num_obs)
    ]


def main(args: argparse.Namespace) -> None:
    """Run the script."""
    results = []
    for path in args.paths:
        print(f"Running test for {path.stem}")
        data = list(load_jsonl(path))
        errors = [entry["emb_err"] for entry in data]
        fixes = [entry["emb_fix"] for entry in data]

        similarities = [
            cosine_similarity([err], [fix]).item()
            for err, fix in zip(errors, fixes)
        ]
        shuffled = shuffle_and_measure(errors, fixes, args.k)

        df = pd.DataFrame({"original": similarities, "shuffled": shuffled})
        df = (
            df.stack()
            .to_frame(name="cosine_similarity")
            .droplevel(0)
            .reset_index(names="label")
        )
        df["embedding_type"] = path.stem
        results.append(df)

    results = pd.concat([df for df in results])
    results.to_csv(args.outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paths", type=Path, nargs="+", help="JSONL embeddings"
    )
    parser.add_argument("--outfile", type=Path, help="Output CSV")
    parser.add_argument("--k", type=int, default=250, help="Num. iterations")
    args = parser.parse_args()
    main(args)
