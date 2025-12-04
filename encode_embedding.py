import os
import csv
from tqdm import tqdm
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path configurations
ROOT = Path("./Amazon_products")
TRAIN_PATH = ROOT / "train"
TEST_PATH = ROOT / "test"
CLASS_PATH = ROOT / "classes.txt" #class_id, class_name
CLASS_KEYWORDS_PATH = ROOT / "class_related_keywords.txt" #class_id, keyword
CLASS_KEYWORDS_ID_PATH = ROOT / "class_keywords_with_id.txt"
TRAIN_CORPUS_PATH = TRAIN_PATH / "train_corpus.txt" #review_id, review_text
TEST_CORPUS_PATH = TEST_PATH / "test_corpus.txt" #review_id, review_text

MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME).eval().to(device)

# Data loading functions
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
    merged_corpus = {**train_corpus, **test_corpus}
    merged_rid = list(merged_corpus.keys())
    merged_text = list(merged_corpus.values())
    return merged_rid, merged_text

def load_class_corpus(path: Path) -> Tuple[List[int], List[str]]:
    id2class = {}
    
    def preprocess_class_text(raw_text):
        parts = raw_text.split(":", 1)
        class_name = parts[0].replace("_", " ")
        if len(parts) > 1:
            keywords = parts[1]
            keyword_list = [kw.replace("_", " ") for kw in keywords.split(",")]
            keywords_str = ", ".join(keyword_list)
            return f"{class_name}: {keywords_str}."
        else:
            return f"{class_name}."
    
    with open(path, 'r', encoding='utf-8') as f:
        for lin in f:
            line = line.strip()
            class_idx, raw = line.split("\t", 1)
            processed_text = preprocess_class_text(raw)
            id2class[int(class_idx)] = processed_text
    
    class_ids = sorted(id2class.keys())
    class_texts = [id2class[i] for i in class_ids]
    return class_ids, class_texts

# BERT encoding function
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask

def encode_texts(texts, batch_size=64, max_len=512):
    all_embeddings = []

    # Process texts in mini-batches
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]

        # Tokenize and move to model device
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        ).to(model.device)

        # Forward pass through BERT
        with torch.no_grad():
            output = model(**encoded)

        # Mean pooling (exclude padding tokens)
        embeddings = mean_pooling(output, encoded["attention_mask"])
        all_embeddings.append(embeddings.cpu())

    # Concatenate all batch embeddings
    return torch.cat(all_embeddings, dim=0)

# Encoding product review document and hierarchical classes
review_ids, review_texts = load_merged_corpus(TRAIN_CORPUS_PATH, TEST_CORPUS_PATH)
class_ids, class_texts = load_class_corpus(CLASS_KEYWORDS_ID_PATH)

review_embeddings = encode_texts(review_texts, batch_size=64, max_len=512)
torch.save({'review_ids': review_ids, 'embeddings': review_embeddings}, ROOT / "merged_review_embeddings.pt")

class_embeddings = encode_texts(class_texts, batch_size=64, max_len=128)
torch.save({'class_ids': class_ids, 'embeddings': class_embeddings}, ROOT / "class_embeddings.pt")