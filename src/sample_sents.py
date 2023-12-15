#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import json
import re
import random
from typing import Generator

from nltk.tokenize import sent_tokenize, word_tokenize


def corpus_load(path: Path, pattern: str = "*.json") -> Generator:
    """Load corpus files."""
    files = list(path.glob(pattern))
    for fname in files:
        with fname.open("r") as fin:
            yield json.loads(fin.read())


def to_sentences(doc: str, min_length: int = 10) -> list[str]:
    """Convert a document to sentences."""
    sents = sent_tokenize(doc)
    sents = [re.sub(r"\s", " ", sent) for sent in sents]
    sents = [sent for sent in sents if len(word_tokenize(sent)) >= min_length]

    return sents


def main(args: argparse.Namespace) -> None:
    """Run the script."""
    sampled = []
    for doc in corpus_load(args.indir, args.pattern):
        sents = to_sentences(doc["text"], args.min_length)
        choices = random.choices(sents, k=args.k)
        sampled.extend(choices)

    with args.outfile.open("w") as fout:
        for sent in sampled:
            print(sent, file=fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=Path, help="Input directory")
    parser.add_argument("--outfile", type=Path, help="Output file")
    parser.add_argument(
        "--k", type=int, default=3, help="Sentences/document to sample"
    )
    parser.add_argument(
        "--min_length", type=int, default=10, help="Minimum sentence length"
    )
    parser.add_argument(
        "--pattern", type=str, default="*.json", help="Glob pattern for input"
    )
    args = parser.parse_args()
    main(args)
