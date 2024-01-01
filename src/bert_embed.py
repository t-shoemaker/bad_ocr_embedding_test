#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import json

from sentence_transformers import SentenceTransformer, models


def main(args: argparse.Namespace) -> None:
    """Run the script."""
    with args.uncorrected.open("r") as fin:
        uncorrected = [ln.strip() for ln in fin.readlines()]
    with args.corrected.open("r") as fin:
        corrected = [ln.strip() for ln in fin.readlines()]

    assert len(uncorrected) == len(corrected), "Misaglined sentences!"

    uncorrected_emb = MODEL.encode(uncorrected)
    corrected_emb = MODEL.encode(corrected)

    with args.embeddings.open("w") as fout:
        for err, fix, emb_err, emb_fix in zip(
            uncorrected, corrected, uncorrected_emb, corrected_emb
        ):
            output = {
                "err": err,
                "fix": fix,
                "emb_err": emb_err.tolist(),
                "emb_fix": emb_fix.tolist(),
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
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="bert-base-uncased",
        help="Model name",
    )
    parser.add_argument(
        "--tokenizer_name", type=str, default="", help="Tokenizer name"
    )
    args = parser.parse_args()

    tokenizer = (
        args.model_checkpoint
        if args.tokenizer_name == ""
        else args.tokenizer_name
    )
    base = models.Transformer(
        model_name_or_path=args.model_checkpoint,
        tokenizer_name_or_path=tokenizer,
    )
    pooling = models.Pooling(base.get_word_embedding_dimension())
    MODEL = SentenceTransformer(modules=[base, pooling])
    main(args)
