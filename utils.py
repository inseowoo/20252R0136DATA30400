import matplotlib.pyplot as plt
import json
import warnings
import numpy as np
from collections import defaultdict
import itertools
import torch
from typing import Dict, List, Tuple
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore")

# Suppress HuggingFace transformers logs
from transformers import logging
logging.set_verbosity_error()

def load_review_corpus(path: Path, corpus_type: str) -> Dict[str, str]:
    rid2text = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            review_idx, review_text = line.strip().split('\t', 1)
            unique_id = f"{corpus_type}_{review_idx}"
            rid2text[unique_id] = review_text.strip()
    return rid2text

def load_merged_corpus(train_path: Path, test_path: Path) -> Tuple[List[str], List[str]]:
    test_corpus = load_review_corpus(test_path, "TEST")
    train_corpus = load_review_corpus(train_path, "TRAIN")

    merged = {**train_corpus, **test_corpus}

    ids = list(merged.keys())
    texts = list(merged.values())
    return ids, texts


def load_class_corpus(path: Path) -> Tuple[List[int], List[str]]:
    id2text = {}
    def preprocess(raw_text):
        parts = raw_text.split(":", 1)
        class_name = parts[0].replace("_", " ")

        if len(parts) > 1:
            keywords = parts[1]
            kw_list = [kw.replace("_", " ") for kw in keywords.split(",")]
            return f"{class_name}: {', '.join(kw_list)}."
        else:
            return f"{class_name}."

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            class_idx, raw = line.split("\t", 1)
            id2text[int(class_idx)] = preprocess(raw)

    class_ids = sorted(id2text.keys())
    class_texts = [id2text[i] for i in class_ids]
    return class_ids, class_texts