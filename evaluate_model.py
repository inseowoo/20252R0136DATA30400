import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import defaultdict
import random
import numpy as np
from pathlib import Path

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

ROOT = Path("./Amazon_products")
BEST_MODEL_PATH = ROOT / "best_model.pt"
TEST_EMB_PATH   = ROOT / "test_review_embeddings.pt"
CLASS_EMB_PATH  = ROOT / "class_embeddings.pt"
HIER_PATH       = ROOT / "class_hierarchy.txt"

BATCH_SIZE = 256
TOP_K = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


parent_map = defaultdict(list)
children_map = defaultdict(list)
def load_hierarchy(path):
    with open(path, "r") as f:
        for line in f:
            p, c = map(int, line.strip().split())
            parent_map[c].append(p)
            children_map[p].append(c)

load_hierarchy(HIER_PATH)

class ClassGCN(nn.Module):
    def __init__(self, emb_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.empty(emb_dim, emb_dim)) for _ in range(num_layers)]
        )
        for W in self.weights:
            nn.init.xavier_uniform_(W)

    def forward(self, H, A_hat):
        for i, W in enumerate(self.weights):
            H = torch.sparse.mm(A_hat, H)
            H = torch.matmul(H, W)
            if i < self.num_layers - 1:
                H = F.relu(H)
                H = F.dropout(H, p=self.dropout, training=self.training)
        return H


def build_adj_norm(cid_list, parent_map, children_map, device):
    cid2idx = {cid: i for i, cid in enumerate(cid_list)}
    C = len(cid_list)
    rows, cols = [], []

    def add_undirected(a, b):
        if a in cid2idx and b in cid2idx:
            ia, ib = cid2idx[a], cid2idx[b]
            rows.append(ia); cols.append(ib)
            rows.append(ib); cols.append(ia)

    for child, parents in parent_map.items():
        for p in parents:
            add_undirected(child, p)

    for i in range(C):
        rows.append(i); cols.append(i)

    indices = torch.tensor([rows, cols], dtype=torch.long, device=device)
    values  = torch.ones(indices.size(1), dtype=torch.float32, device=device)

    A = torch.sparse_coo_tensor(indices, values, size=(C, C), device=device).coalesce()

    deg = torch.sparse.sum(A, dim=1).to_dense()
    deg_inv_sqrt = torch.pow(deg.clamp(min=1e-12), -0.5)

    idx = A.indices()
    val = A.values()
    norm_val = val * deg_inv_sqrt[idx[0]] * deg_inv_sqrt[idx[1]]

    A_hat = torch.sparse_coo_tensor(idx, norm_val, size=(C, C), device=device).coalesce()
    return A_hat
class GCNEnhancedClassifier(nn.Module):
    def __init__(self, doc_dim, class_init_emb, cid_list, A_hat, num_layers=2):
        super().__init__()
        self.cid_list = cid_list
        self.class_emb = nn.Parameter(class_init_emb.clone())
        emb_dim = self.class_emb.size(1)

        self.gcn = ClassGCN(emb_dim=emb_dim, num_layers=num_layers)

        self.B = nn.Parameter(torch.empty(emb_dim, doc_dim))
        nn.init.xavier_uniform_(self.B)

        self.register_buffer("A_hat", A_hat)

    def forward(self, docs):
        class_mat = self.gcn(self.class_emb, self.A_hat)
        proj = class_mat @ self.B

        proj_norm = F.normalize(proj, dim=1)
        docs_norm = F.normalize(docs, dim=1)

        logits = (proj_norm @ docs_norm.T).T
        return logits

class_data = torch.load(CLASS_EMB_PATH, map_location=device)
cid_list = class_data["class_ids"]
class_emb = class_data["embeddings"].to(device)

A_hat = build_adj_norm(cid_list, parent_map, children_map, device)

test_data = torch.load(TEST_EMB_PATH, map_location=device)
test_ids = test_data["ids"]
test_emb = test_data["embeddings"]

class TestDataset(Dataset):
    def __init__(self, ids, emb):
        self.ids = ids
        self.emb = emb
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, i):
        return {"doc_id": self.ids[i], "embedding": self.emb[i]}

test_loader = DataLoader(
    TestDataset(test_ids, test_emb),
    batch_size=BATCH_SIZE,
    shuffle=False
)


model = GCNEnhancedClassifier(
    doc_dim=test_emb.size(1),
    class_init_emb=class_emb,
    cid_list=cid_list,
    A_hat=A_hat,
    num_layers=2
).to(device)

state = torch.load(BEST_MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.eval()

import csv
import numpy as np

TAU23 = 0.005
MIN_S3 = None

TOP_POOL = 30
SUBMISSION_PATH = ROOT / "submission.csv"

cid2idx = {c: i for i, c in enumerate(cid_list)}

print("Running inference + building submission...")

submission_rows = []
num_2_labels = 0
num_3_labels = 0

model.eval()
with torch.no_grad():
    for batch in test_loader:
        docs = batch["embedding"].to(device)
        ids  = batch["doc_id"]

        logits = model(docs).cpu().numpy()  # (B, C)

        for bi, doc_id in enumerate(ids):
            scores = logits[bi]  # (C,)

            sorted_idx = np.argsort(-scores)[:TOP_POOL]
            top_scores = scores[sorted_idx]
            top_cids = [cid_list[j] for j in sorted_idx]

            selected = top_cids[:3]
            '''
            if len(top_cids) >= 3:
                s1, s2, s3 = top_scores[:3]
                gap23 = s2 - s3
                take3 = (gap23 >= TAU23)
                if MIN_S3 is not None:
                    take3 = take3 and (s3 <= MIN_S3)

                if take3:
                    selected = top_cids[:3]

            if len(selected) == 2:
                num_2_labels += 1
            else:
                num_3_labels += 1
            '''
            labels_str = ",".join(str(c) for c in selected)
            numeric_id = doc_id.replace("test_", "")
            submission_rows.append([numeric_id, labels_str])

with open(SUBMISSION_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "labels"])
    writer.writerows(submission_rows)

print("\nSanity check (first 20 rows):")
for row in submission_rows[:20]:
    print(row)
