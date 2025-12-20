# Final Project – Big Data Analytics (DATA304)

Hierarchical Multi-Label Text Classification using only class names.
The pipeline includes embedding generation, similarity computation,
core class mining, GCN-based classifier training, and evaluation.

---

## Precomputed Files (Required)

All large `.pt` files (embeddings + trained model) are hosted on Google Drive.

Download:
https://drive.google.com/file/d/1U9soJmWcG0Zh1lwn6IJSaDkxPoMwxxvZ/view?usp=drive_link

After downloading, unzip the file **in the project root directory**.

---

## Dependencies

Install if needed:
pip install torch numpy tqdm scikit-learn sentence-transformers

---
## Reproducing Results
### Option 1: Using the pretrained model (Faster than running whole pipeline
- Skips embedding computations and training process.
- Uses the precomputed best_model.pt state
- Execute: **python evaluate_model.py**
- This generates submission.csv

### Option 2: Running the full pipeline (From scratch)
- Runs all steps sequentially (encode_embedding.py, calculate_similarity.py,
  core_class_mining.py, classifier_training.py, evaluate_model.py)
- Execute: **python run.py**

