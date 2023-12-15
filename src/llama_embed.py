#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import json

from llama_cpp import Llama


def main(args: argparse.Namespace) -> None:
    """Run the script."""
    with args.uncorrected.open("r") as fin:
        uncorrected = [ln.strip() for ln in fin.readlines()]
    with args.corrected.open("r") as fin:
        corrected = [ln.strip() for ln in fin.readlines()]

    assert len(uncorrected) == len(corrected), "Misaglined sentences!"

    with args.embeddings.open("w") as fout:
        for err, fix in zip(uncorrected, corrected):
            emb_err, emb_fix = LLM.embed(err), LLM.embed(fix)
            output = {
                "err": err, "fix": fix, "emb_err": emb_err, "emb_fix": emb_fix
            }
            print(json.dumps(output), file=fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--uncorrected", type=Path, help="Uncorrected sentences"
    )
    parser.add_argument("--corrected", type=Path, help="Corrected sentences")
    parser.add_argument(
        "--embeddings", type=Path, help="Output embeddings (JSONL)"
    )
    parser.add_argument("--model_path", type=Path, help="Path to LLM")
    parser.add_argument(
        "--n_gpu_layers", type=int, default=0, help="Model layers on the GPU"
    )
    args = parser.parse_args()

    LLM = Llama(
        model_path=args.model_path.as_posix(),
        embedding=True,
        n_gpu_layers=args.n_gpu_layers,
        verbose=True,
    )
    main(args)
