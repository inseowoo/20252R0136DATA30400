import os
import csv
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from utils import *

import torch
from sentence_transformers import SentenceTransformer

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = Path("./Amazon_products")
TRAIN_PATH = ROOT / "train"
TEST_PATH = ROOT / "test"
CLASS_KEYWORDS_ID_PATH = ROOT / "class_keywords_with_id.txt"

TRAIN_CORPUS_PATH = TRAIN_PATH / "train_corpus.txt"
TEST_CORPUS_PATH = TEST_PATH / "test_corpus.txt"

MODEL_NAME = "all-mpnet-base-v2"
model = SentenceTransformer(MODEL_NAME).to(device)

# Functions for loading txt files
def encode_texts(texts: List[str], batch_size=256):
    """
    SBERT handles everything: tokenization, padding, pooling.
    Returns a Tensor (N, 768).
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=True
    )

    return embeddings.cpu()

review_ids, review_texts = load_merged_corpus(TRAIN_CORPUS_PATH, TEST_CORPUS_PATH)
class_ids, class_texts = load_class_corpus(CLASS_KEYWORDS_ID_PATH)

review_embeddings = encode_texts(review_texts, batch_size=256)
torch.save(
    {"review_ids": review_ids, "embeddings": review_embeddings},
    ROOT / "merged_review_embeddings.pt"
)

class_embeddings = encode_texts(class_texts, batch_size=256)
torch.save(
    {"class_ids": class_ids, "embeddings": class_embeddings},
    ROOT / "class_embeddings.pt"
)

print("\nEmbeddings saved successfully.")
