import math
import random
import copy
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple

from utils import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

ROOT = Path("./Amazon_products")
HIER_PATH       = ROOT / "class_hierarchy.txt"
SILVER_PATH     = ROOT / "confident_core_classes.pt"
REVIEW_EMB_PATH = ROOT / "bert_review_embeddings.pt"
CLASS_EMB_PATH  = ROOT / "bert_class_embeddings.pt"

MAX_NEG_PER_DOC   = 50
POS_WEIGHT_CAP    = 30.0
EPOCHS            = 200
PATIENCE          = 6
LR                = 4e-3

parent_map = defaultdict(list)
children_map = defaultdict(list)
all_classes = set()

def load_hierarchy(path):
    global parent_map, children_map, all_classes
    parent_map.clear()
    children_map.clear()
    all_classes.clear()

    nodes = set()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p, c = line.strip().split()
            p = int(p); c = int(c)
            parent_map[c].append(p)
            children_map[p].append(c)
            nodes.add(p); nodes.add(c)
    all_classes = nodes

def get_children(parent_id: int) -> List[int]:
    return children_map.get(parent_id, [])

def get_parents(child_id: int) -> List[int]:
    return parent_map.get(child_id, [])

def get_descendants(class_id: int) -> Set[int]:
    descendants = set()
    queue = deque([class_id])
    visited = set()
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        children = get_children(current)
        for ch in children:
            descendants.add(ch)
            queue.append(ch)
    return descendants

def get_ancestors(class_id: int) -> Set[int]:
    ancestors = set()
    queue = deque([class_id])
    visited = set()
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        parents = get_parents(current)
        for p in parents:
            ancestors.add(p)
            queue.append(p)
    return ancestors

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

class EmbDataset(Dataset):
    def __init__(self, doc_ids, pid2idx, review_embedding):
        self.doc_ids = doc_ids
        self.pid2idx = pid2idx
        self.review_embedding = review_embedding

    def __len__(self):
        return len(self.doc_ids)

    def __getitem__(self, idx):
        doc_id = self.doc_ids[idx]
        emb_idx = self.pid2idx[doc_id]
        emb = self.review_embedding[emb_idx]
        return {"doc_id": doc_id, "embedding": emb}

def build_pos_neg(core_dict, numeric_classes):
    pos = {}
    neg = {}
    for doc, core in core_dict.items():
        if not core:
            pos[doc] = set()
            neg[doc] = set()
            continue

        C = set(core.keys())
        P = set(C)
        for x in list(C):
            P.update(get_ancestors(x))

        descendants = set()
        for x in C:
            descendants.update(get_descendants(x))
        N = numeric_classes - P - descendants

        pos[doc] = P
        neg[doc] = N

    return pos, neg

def build_targets_and_mask(doc_ids_batch, pos, neg, cid_to_col, C, device, max_neg_per_doc=50):
    B = len(doc_ids_batch)
    targets = torch.zeros((B, C), device=device)
    mask    = torch.zeros((B, C), device=device)

    for i, d in enumerate(doc_ids_batch):
        pos_set = pos.get(d, set())
        neg_set = neg.get(d, set())

        for c in pos_set:
            col = cid_to_col.get(c, None)
            if col is not None:
                targets[i, col] = 1.0
                mask[i, col] = 1.0

        if neg_set:
            neg_list = list(neg_set)
            if len(neg_list) > max_neg_per_doc:
                neg_list = random.sample(neg_list, max_neg_per_doc)
            for c in neg_list:
                col = cid_to_col.get(c, None)
                if col is not None:
                    mask[i, col] = 1.0

    return targets, mask

def compute_pos_weight_vector(targets, mask, cap=100.0):
    pos_counts = (targets * mask).sum(dim=0)
    neg_counts = ((1.0 - targets) * mask).sum(dim=0)
    pos_weight = neg_counts / pos_counts.clamp(min=1.0)
    pos_weight[pos_counts == 0] = 1.0
    return pos_weight.clamp(max=cap)

@torch.no_grad()
def validate_epoch(model, loader, device, pos, neg, cid_list, max_neg_per_doc=50):
    cid_to_col = {cid: i for i, cid in enumerate(cid_list)}
    C = len(cid_list)

    model.eval()
    total_loss = 0.0
    num_batches = 0

    pos_sum = 0.0; pos_n = 0
    neg_sum = 0.0; neg_n = 0

    for batch in loader:
        docs = batch["embedding"].to(device)
        ids  = batch["doc_id"]

        targets, mask = build_targets_and_mask(
            ids, pos, neg, cid_to_col, C, device, max_neg_per_doc=max_neg_per_doc
        )
        if mask.sum().item() == 0:
            continue

        logits = model(docs)

        pos_weight = compute_pos_weight_vector(targets, mask, cap=POS_WEIGHT_CAP).to(device)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=pos_weight
        )
        loss = (bce * mask).sum() / mask.sum().clamp(min=1e-8)

        total_loss += loss.item()
        num_batches += 1

        pos_mask = (targets == 1.0) & (mask == 1.0)
        neg_mask = (targets == 0.0) & (mask == 1.0)
        if pos_mask.any():
            pos_sum += logits[pos_mask].sum().item()
            pos_n += int(pos_mask.sum().item())
        if neg_mask.any():
            neg_sum += logits[neg_mask].sum().item()
            neg_n += int(neg_mask.sum().item())

    avg_loss = (total_loss / num_batches) if num_batches > 0 else 0.0
    pos_mean = (pos_sum / pos_n) if pos_n > 0 else float("nan")
    neg_mean = (neg_sum / neg_n) if neg_n > 0 else float("nan")
    margin   = (pos_mean - neg_mean) if (pos_n > 0 and neg_n > 0) else float("nan")
    return avg_loss, pos_mean, neg_mean, margin

def train_epoch(model, train_loader, optimizer, device, pos, neg, cid_list):
    cid_to_col = {cid: i for i, cid in enumerate(cid_list)}
    C = len(cid_list)

    model.train()
    total_loss = 0.0
    num_batches = 0

    pos_sum = 0.0; pos_n = 0
    neg_sum = 0.0; neg_n = 0

    for batch in tqdm(train_loader, desc="Training"):
        docs = batch["embedding"].to(device)
        ids  = batch["doc_id"]

        targets, mask = build_targets_and_mask(
            ids, pos, neg, cid_to_col, C, device, max_neg_per_doc=MAX_NEG_PER_DOC
        )
        if mask.sum().item() == 0:
            continue

        logits = model(docs)

        pos_weight = compute_pos_weight_vector(targets, mask, cap=POS_WEIGHT_CAP).to(device)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=pos_weight
        )
        loss = (bce * mask).sum() / mask.sum().clamp(min=1e-8)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pos_mask = (targets == 1.0) & (mask == 1.0)
        neg_mask = (targets == 0.0) & (mask == 1.0)
        if pos_mask.any():
            pos_sum += logits[pos_mask].sum().item()
            pos_n += int(pos_mask.sum().item())
        if neg_mask.any():
            neg_sum += logits[neg_mask].sum().item()
            neg_n += int(neg_mask.sum().item())

    train_loss = (total_loss / num_batches) if num_batches > 0 else 0.0
    train_pos_mean = (pos_sum / pos_n) if pos_n > 0 else float("nan")
    train_neg_mean = (neg_sum / neg_n) if neg_n > 0 else float("nan")
    train_margin   = (train_pos_mean - train_neg_mean) if (pos_n > 0 and neg_n > 0) else float("nan")

    return train_loss, train_pos_mean, train_neg_mean, train_margin

def train(model, train_loader, val_loader, optimizer, device, pos, neg, cid_list):
    best_val_margin = -float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_pos_mean, train_neg_mean, train_margin = train_epoch(
            model, train_loader, optimizer, device, pos, neg, cid_list
        )

        val_loss, val_pos_mean, val_neg_mean, val_margin = validate_epoch(
            model, val_loader, device, pos, neg, cid_list, max_neg_per_doc=MAX_NEG_PER_DOC
        )

        print(
            f"[Epoch {epoch}] "
            f"loss(train/val)={train_loss:.4f}/{val_loss:.4f} | "
            f"pos_logit(train/val)={train_pos_mean:.4f}/{val_pos_mean:.4f} | "
            f"neg_logit(train/val)={train_neg_mean:.4f}/{val_neg_mean:.4f} | "
            f"margin(train/val)={train_margin:.4f}/{val_margin:.4f}"
        )

        improved = (val_margin == val_margin) and (val_margin > best_val_margin)
        if improved:
            best_val_margin = val_margin
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[Early Stopping] val margin did not improve for {PATIENCE} epochs.")
                break

    return best_model_state, best_val_margin

print("Loading hierarchy…")
load_hierarchy(HIER_PATH)

print("Loading silver labels…")
silver = torch.load(SILVER_PATH)
core = silver["confident_cores"]
doc_ids = silver["doc_ids"]
class_ids = silver["class_ids"]

print("Loading review embeddings…")
review_data = torch.load(REVIEW_EMB_PATH)
pid_list = review_data["ids"]
pid2idx = {pid: i for i, pid in enumerate(pid_list)}
review_embedding = review_data["embeddings"]

print("Loading class embeddings…")
class_data = torch.load(CLASS_EMB_PATH)
cid_list = [int(c) for c in class_data["ids"]]
class_embedding = class_data["embeddings"]

numeric_classes = set(cid_list)

print("Building initial pos/neg sets…")
pos, neg = build_pos_neg(core, numeric_classes)

labeled_pid_list = []
for pid in pid_list:
    if pid in pos and pid in neg and (len(pos[pid]) + len(neg[pid]) > 0):
        labeled_pid_list.append(pid)

print(f"Labeled docs for training/val: {len(labeled_pid_list)} / {len(pid_list)}")

val_size = int(0.1 * len(labeled_pid_list))
val_idx = set(random.sample(range(len(labeled_pid_list)), val_size))
train_ids = [labeled_pid_list[i] for i in range(len(labeled_pid_list)) if i not in val_idx]
val_ids   = [labeled_pid_list[i] for i in range(len(labeled_pid_list)) if i in val_idx]

train_loader = DataLoader(EmbDataset(train_ids, pid2idx, review_embedding), batch_size=32, shuffle=True)
val_loader   = DataLoader(EmbDataset(val_ids, pid2idx, review_embedding), batch_size=64)

A_hat = build_adj_norm(cid_list, parent_map, children_map, device=device)
print("A_hat nnz:", A_hat._nnz(), "shape:", A_hat.shape)

model = GCNEnhancedClassifier(
    doc_dim=review_embedding.size(1),
    class_init_emb=class_embedding.to(device),
    cid_list=cid_list,
    A_hat=A_hat,
    num_layers=2
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=LR)

best_state, best_margin = train(
    model, train_loader, val_loader, optimizer, device, pos, neg, cid_list
)

print("Training complete. Best val margin:", best_margin)

model_path = ROOT / "best_model.pt"
torch.save(best_state, model_path)
print(f"Best model saved to {model_path}")
