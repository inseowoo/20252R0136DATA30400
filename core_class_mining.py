import math
import random
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

ROOT = Path("./Amazon_products")

SIMILARITY_PATH          = ROOT / "similarity_matrix.pt"
HIERARCHY_PATH           = ROOT / "class_hierarchy.txt"
CORE_CANDIDATES_PATH     = ROOT / "core_class_candidates.pt"
CONFIDENT_CORES_PATH     = ROOT / "confident_core_classes.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global hierarchy maps
parent_map = defaultdict(list)
children_map = defaultdict(list)

# Loading class hierarchy and adding a synthetic root if needed
# This is in case the class hierarchy is a forest (multiple roots).
parent_map.clear()
children_map.clear()

def load_hierarchy(path):
    global parent_map, children_map

    nodes = set()
    parent_count = defaultdict(int)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                parent, child = line.split()
            except ValueError:
                continue

            nodes.add(parent)
            nodes.add(child)

            children_map[parent].append(child)
            parent_map[child].append(parent)
            parent_count[child] += 1

    # Identifying roots. Upon inspection, ['0', '3', '10', '23', '40', '169'] are roots.
    # Since I want to explore the whole hierarchy, I will add a synthetic root connecting all roots.
    roots = [n for n in nodes if parent_count[n] == 0]

    print(f"Found {len(roots)} & zero indegree nodes: {sorted(roots)}")

    if len(roots) == 0:
        raise RuntimeError("No root found")

    if len(roots) == 1:
        return roots[0]

    SYN_ROOT = "SYN_ROOT"
    for r in roots:
        parent_map[r].append(SYN_ROOT)
    children_map[SYN_ROOT] = roots
    nodes.add(SYN_ROOT)
    return SYN_ROOT

def get_childrens(parent_id):
    return children_map.get(parent_id, [])


def get_parents(child_id):
    return parent_map.get(child_id, [])


def get_siblings(class_id):
    siblings = set()
    for parent in get_parents(class_id):
        for child in get_childrens(parent):
            if child != class_id:
                siblings.add(child)
    return siblings

# Core class candidate selection implementation.
# Following the methodology of TaxoClass
# Perform top-down candidate selection for each document.
# Start at root, pick top-2 children by similarity
# For each level pick l+2 children and keep (l+1)^2 best by path score.
def candidate_selection(doc_idx, sim_matrix, class_id_to_idx, root_id):
    candidate_set = set()
    path_score = {} # path scores
    path_score[root_id] = 1.0 # ps(Root) = 1

    # Level 0: root -> pick top 2 children by similarity
    root_children = get_childrens(root_id)
    if not root_children:
        return candidate_set

    child_infos = []
    row = sim_matrix[doc_idx]

    for child in root_children:
        if not child.isdigit():
            continue
        child_int = int(child)
        col_idx = class_id_to_idx.get(child_int)
        if col_idx is None:
            continue
        sim = float(row[col_idx])
        ps_child = path_score[root_id] * sim
        child_infos.append((child, sim, ps_child))

    if not child_infos:
        return candidate_set

    # Top-2 by similarity
    child_infos.sort(key=lambda x: x[1], reverse=True)
    top2 = child_infos[:2]

    current_level_nodes = []
    for cid, _, ps_child in top2:
        candidate_set.add(cid)
        path_score[cid] = ps_child
        current_level_nodes.append(cid)
        
    level = 1

    while current_level_nodes:
        pooled_children_ps = {}

        for parent_id in current_level_nodes:
            children = get_childrens(parent_id)
            if not children:
                continue

            child_sims = []
            for child in children:
                if not child.isdigit():
                    continue
                child_int = int(child)
                col_idx = class_id_to_idx.get(child_int)
                if col_idx is None:
                    continue
                sim = float(row[col_idx])
                child_sims.append((child, sim))

            if not child_sims:
                continue

            k = min(level + 2, len(child_sims))
            child_sims.sort(key=lambda x: x[1], reverse=True)
            top_k = child_sims[:k]

            for child, sim in top_k:
                best_parent_ps = 0.0
                for p in get_parents(child):
                    if p in path_score:
                        best_parent_ps = max(best_parent_ps, path_score[p])
                if best_parent_ps == 0.0:
                    continue
                ps_candidate = best_parent_ps * sim

                if (child not in pooled_children_ps) or (ps_candidate > pooled_children_ps[child]):
                    pooled_children_ps[child] = ps_candidate

        if not pooled_children_ps:
            break

        keep = (level + 1) ** 2
        sorted_children = sorted(pooled_children_ps.items(), key=lambda x: x[1], reverse=True)
        selected = sorted_children[:keep]

        next_level_nodes = []
        for child, ps_val in selected:
            if child not in path_score or ps_val > path_score[child]:
                path_score[child] = ps_val
            candidate_set.add(child)
            next_level_nodes.append(child)

        current_level_nodes = next_level_nodes
        level += 1
        
    # Remove root from candidate set if present
    if root_id in candidate_set:
        candidate_set.remove(root_id)

    return candidate_set

# Confident Core Class Identification
# I reduced the possible candidates set above.
# This part computes the confidence score of cadidates and selects confident cores.
# The confidence score is compare to the median confidence score per class.

def compute_conf_scores(sim_matrix, class_id_to_idx, candidate_sets):
    num_docs = sim_matrix.shape[0]

    conf_per_doc = [dict() for _ in range(num_docs)]
    per_class_scores = defaultdict(list)

    for i in tqdm(range(num_docs), desc="Computing conf(D,c)"):
        row = sim_matrix[i]
        cand_set = candidate_sets[i]
        if not cand_set:
            continue

        for c_str in cand_set:
            if not c_str.isdigit():
                continue
            c_int = int(c_str)
            col_idx = class_id_to_idx.get(c_int)
            if col_idx is None:
                continue

            sim_c = float(row[col_idx])

            # Parents and siblings
            parents = get_parents(c_str)
            siblings = get_siblings(c_str)
            competitors = set(parents) | siblings

            competitor_sims = []
            for comp in competitors:
                if not comp.isdigit():
                    continue
                comp_int = int(comp)
                comp_idx = class_id_to_idx.get(comp_int)
                if comp_idx is None:
                    continue
                competitor_sims.append(float(row[comp_idx]))

            if competitor_sims:
                max_comp = max(competitor_sims)
            else:
                max_comp = 0.0

            conf = sim_c - max_comp
            conf_per_doc[i][c_int] = conf
            per_class_scores[c_int].append(conf)

    # median per class
    median_conf_per_class = {}
    for c_int, scores in per_class_scores.items():
        median_conf_per_class[c_int] = float(np.median(scores))

    return conf_per_doc, median_conf_per_class


def select_confident_cores(candidate_sets, conf_per_doc, median_conf_per_class):
    num_docs = len(candidate_sets)
    core_classes_per_doc = [dict() for _ in range(num_docs)]

    for i in range(num_docs):
        cand_set = candidate_sets[i]
        if not cand_set:
            continue

        for c_str in cand_set:
            if not c_str.isdigit():
                continue
            c_int = int(c_str)
            conf = conf_per_doc[i].get(c_int, None)
            if conf is None:
                continue
            med = median_conf_per_class.get(c_int, None)
            if med is None:
                continue
            threshold = max(med, 0.04)
            if conf >= threshold:
                core_classes_per_doc[i][c_int] = float(conf)

    return core_classes_per_doc

# The similarity scores I got from SBERT embeddings and tf-idf vectors are too flat.
# This introduced a problem where the median confidence was too low and many noisy classes were selected.
# This function explicitly reduces number of core classes to 3 when greater than 3
# Add the next most confident neighbor when core class == 1
# Keep as is when core class is 3, 2, or 0.
def refine_core_classes(core_classes_per_doc, sim_matrix, class_id_to_idx, review_ids):
    refined = {}
    
    for doc_idx, doc_id in enumerate(review_ids):
        cls_conf_dict = core_classes_per_doc[doc_idx]
        items = sorted(cls_conf_dict.items(), key=lambda x: x[1], reverse=True)

        if len(items) == 0 or len(items) == 3 or len(items) == 2:
            refined[doc_id] = cls_conf_dict
            continue

        if len(items) > 3:
            refined[doc_id] = dict(items[:3])
            continue
        
        if len(items) == 1:
            main_c, main_conf = items[0]

            # Collect neighbors: parents, siblings, and children
            parents = get_parents(str(main_c))
            siblings = get_siblings(str(main_c))
            children = get_childrens(str(main_c))

            neighbor_ids = set()

            for nb in list(parents) + list(siblings) + list(children):
                if nb.isdigit():
                    neighbor_ids.add(int(nb))

            candidates = []

            row = sim_matrix[doc_idx]

            for nb_id in neighbor_ids:
                col = class_id_to_idx.get(nb_id)
                if col is None:
                    continue
                sim_val = float(row[col])
                candidates.append((nb_id, sim_val))

            candidates.sort(key=lambda x: x[1], reverse=True)

            refined_list = {main_c: main_conf}
            if candidates:
                nb_id, nb_sim = candidates[0]
                refined_list[nb_id] = nb_sim

            refined[doc_id] = refined_list
            continue
    
    return refined

# Load similarity matrix and perform core class mining
data = torch.load(SIMILARITY_PATH, weights_only=False)
review_ids = data["review_ids"]
class_ids = data["class_ids"]
sim_mat = data["similarity_matrix"]

class_ids = [int(c) for c in class_ids]

if isinstance(sim_mat, torch.Tensor):
    sim_matrix = sim_mat.cpu().numpy().astype(np.float32)
else:
    sim_matrix = np.asarray(sim_mat, dtype=np.float32)

num_docs, num_classes = sim_matrix.shape
    
class_id_to_idx = {cid: i for i, cid in enumerate(class_ids)}

root_id = load_hierarchy(HIERARCHY_PATH)

candidate_sets = []

for i in tqdm(range(num_docs), desc="Core candidate mining"):
    cand_set = candidate_selection(
        doc_idx=i,
        sim_matrix=sim_matrix,
        class_id_to_idx=class_id_to_idx,
        root_id=root_id
    )
    candidate_sets.append(cand_set)

cand_dict = {}
for i, doc_id in enumerate(review_ids):
    cand_ints = [int(c) for c in candidate_sets[i] if c.isdigit()]
    cand_dict[doc_id] = sorted(cand_ints)

torch.save(
    {
        "doc_ids": review_ids,
        "class_ids": class_ids,
        "candidates": cand_dict,
    },
    CORE_CANDIDATES_PATH,
)
print(f"Core candidate sets saved")

conf_per_doc, median_conf_per_class = compute_conf_scores(
    sim_matrix=sim_matrix,
    class_id_to_idx=class_id_to_idx,
    candidate_sets=candidate_sets,
)

core_classes_per_doc = select_confident_cores(
    candidate_sets=candidate_sets,
    conf_per_doc=conf_per_doc,
    median_conf_per_class=median_conf_per_class,
)

core_classes_per_doc_refined = refine_core_classes(
    core_classes_per_doc=core_classes_per_doc,
    sim_matrix=sim_matrix,
    class_id_to_idx=class_id_to_idx,
    review_ids=review_ids,
)

core_dict = {}
num_non_empty = 0
label_counts = []
confidence_scores = []

for i, doc_id in enumerate(review_ids):
    # {class_id: confidence_score}
    class_dict = core_classes_per_doc_refined[doc_id]
    core_dict[doc_id] = class_dict
    
    if class_dict:
        num_non_empty += 1
        label_counts.append(len(class_dict))
        confidence_scores.extend(class_dict.values())

print(f"Docs with non-empty core set: {num_non_empty} / {num_docs}")
print(f"Coverage: {100.0 * num_non_empty / num_docs:.2f}%")
if label_counts:
    print(f"Avg labels per doc (non-empty): {sum(label_counts) / len(label_counts):.2f}")
    print(f"Min labels per doc: {min(label_counts)}, Max: {max(label_counts)}")
if confidence_scores:
    print(f"Confidence scores - Mean: {np.mean(confidence_scores):.3f}, Median: {np.median(confidence_scores):.3f}, Min: {np.min(confidence_scores):.3f}, Max: {np.max(confidence_scores):.3f}")

torch.save(
    {
        "doc_ids": review_ids,
        "class_ids": class_ids,
        "confident_cores": core_dict,
    },
    CONFIDENT_CORES_PATH,
)
print(f"Confident core classes saved")