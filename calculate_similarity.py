import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
# Vectorizing review corpus and classes using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=20000)
tfidf_review = vectorizer.fit_transform(review_texts)
tfidf_class = vectorizer.transform(class_texts)
sim_tfidf = cosine_similarity(tfidf_review, tfidf_class)
# Loading SBERT embeddings
review_data = torch.load(REVIEW_EMB_PATH)
class_data = torch.load(CLASS_EMB_PATH)

review_emb = review_data["embeddings"]   # shape (N, 768)
class_emb = class_data["embeddings"]     # shape (C, 768)

# Normalize for cosine similarity
review_emb = F.normalize(review_emb, p=2, dim=1)
class_emb = F.normalize(class_emb, p=2, dim=1)

sim_bert = (review_emb @ class_emb.T).cpu().numpy()
def normalize(sim):
    sim_min, sim_max = sim.min(), sim.max()
    return (sim - sim_min) / (sim_max - sim_min + 1e-8)

sim_tfidf = normalize(sim_tfidf)
sim_bert = normalize(sim_bert)

alpha, beta = 0.7, 0.3
ensemble_sim = alpha * sim_bert + beta * sim_tfidf
torch.save({"review_ids": review_ids,"class_ids": class_ids, "similarity_matrix": ensemble_sim}, SAVE_SIM_PATH)

print(f"\nSimilarity matrix saved successfully")
