import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from utils import *

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = Path("./Amazon_products")

# Paths
REVIEW_EMB_PATH = ROOT / "merged_review_embeddings.pt"
CLASS_EMB_PATH = ROOT / "class_embeddings.pt"
TRAIN_CORPUS_PATH = ROOT / "train/train_corpus.txt"
TEST_CORPUS_PATH = ROOT / "test/test_corpus.txt"
CLASS_KEYWORDS_ID_PATH = ROOT / "class_keywords_with_id.txt"
SAVE_SIM_PATH = ROOT / "similarity_matrix.pt"


review_ids, review_texts = load_merged_corpus(TRAIN_CORPUS_PATH, TEST_CORPUS_PATH)
class_ids, class_texts = load_class_corpus(CLASS_KEYWORDS_ID_PATH)

num_reviews = len(review_texts)
num_classes = len(class_texts)

print(f"Loaded {num_reviews} reviews and {num_classes} classes")

# Loading SBERT embeddings
review_data = torch.load(REVIEW_EMB_PATH)
class_data = torch.load(CLASS_EMB_PATH)

review_emb = review_data["embeddings"]   # shape (N, 768)
class_emb = class_data["embeddings"]     # shape (C, 768)

# Normalize for cosine similarity
review_emb = F.normalize(review_emb, p=2, dim=1)
class_emb = F.normalize(class_emb, p=2, dim=1)

sim_bert = (review_emb @ class_emb.T).numpy()

# Calculate TF-IDF similarity
tfidf = TfidfVectorizer(max_features=50000)

tfidf_review = tfidf.fit_transform(review_texts)
tfidf_class = tfidf.transform(class_texts)

sim_tfidf = cosine_similarity(tfidf_review, tfidf_class)

def normalize(sim):
    sim_min, sim_max = sim.min(), sim.max()
    return (sim - sim_min) / (sim_max - sim_min + 1e-8)

sim_bert = normalize(sim_bert)
sim_tfidf = normalize(sim_tfidf)

# Calculating the weighted similarity
alpha = 0.7
beta = 0.3

sim_total = alpha * sim_bert + beta * sim_tfidf
sim_total = sim_total.astype(np.float32)

torch.save({"review_ids": review_ids,"class_ids": class_ids, "similarity_matrix": sim_total}, SAVE_SIM_PATH)

print(f"\nSaved similarity matrices to: {SAVE_SIM_PATH}")
print("Done!")
