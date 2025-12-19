import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from pathlib import Path
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

ROOT = Path("./Amazon_products")
TRAIN_CORPUS = ROOT / "train/train_corpus.txt"
TEST_CORPUS  = ROOT / "test/test_corpus.txt"
CLASS_NAME_FILE = ROOT / "class_keywords_with_id.txt"

REVIEW_EMB_PATH = ROOT / "bert_review_embeddings.pt"
CLASS_EMB_PATH = ROOT / "bert_class_embeddings.pt"
TEST_EMB_PATH = ROOT / "bert_test_embeddings.pt"

MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME).eval().to(device)

def load_test_corpus(path: Path, corpus_type: str) -> Tuple[List[str], List[str]]:
    rid2text = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            review_idx, review_text = line.strip().split('\t', 1)
            unique_id = f"{corpus_type}_{review_idx}"
            rid2text[unique_id] = review_text.strip()
    ids = list(rid2text.keys())
    texts = list(rid2text.values())
    return ids, texts

def mean_pooling(model_output, attention_mask):
    """
    Apply mean pooling on BERT token embeddings, masking out padding tokens.

    Args:
        model_output: Output object from a BERT model (contains last_hidden_state).
        attention_mask (torch.Tensor): Attention mask of shape (batch_size, seq_len),
                                       where 1 = real token and 0 = padding.

    Returns:
        torch.Tensor: Sentence embeddings of shape (batch_size, hidden_size).
    """
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask

def encode_texts(texts, batch_size=64):
    """
    Encode a list of texts into mean-pooled BERT embeddings.

    Args:
        texts (list of str): Input texts to encode.
        batch_size (int, optional): Batch size for encoding. Default is 64.

    Returns:
        torch.Tensor: Tensor of shape (len(texts), hidden_size) containing embeddings.
    """
    all_embeddings = []

    # Process texts in mini-batches
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]

        # Tokenize and move to model device
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
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

review_ids, review_texts = load_merged_corpus(TRAIN_CORPUS, TEST_CORPUS)
test_ids, test_texts = load_test_corpus(TEST_CORPUS, "TEST")
class_ids, class_texts = load_class_corpus(CLASS_NAME_FILE)

review_emb = encode_texts(review_texts)
torch.save({"ids": review_ids, "embeddings": review_emb}, REVIEW_EMB_PATH)

test_emb = encode_texts(test_texts)
torch.save({"ids": test_ids, "embeddings": test_emb}, TEST_EMB_PATH)

# label encoding
batch_size = 64
all_embeddings = []
for i in tqdm(range(0, len(class_texts), batch_size)):
    batch = class_texts[i:i+batch_size]
    encoded = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model(**encoded)
    emb = mean_pooling(output, encoded["attention_mask"])  # (B, D)
    all_embeddings.append(emb.cpu())

class_embeddings = torch.cat(all_embeddings, dim=0)  # (C, D)
torch.save({"ids": class_ids, "embeddings": class_embeddings}, CLASS_EMB_PATH)