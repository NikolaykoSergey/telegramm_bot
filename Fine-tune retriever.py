# fine_tune_retriever.py (skeleton)
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
import joblib
import random

# собери датасет из feedback: для каждой полезной записи (feedback==1)
# - query = question
# - positive = лучший source.text (score max)
# - negatives = случайные другие chunks

# Пример:
examples = []
# pseudo: load feedback rows and a pool of passages (из cached chunks or Qdrant dump)
# for r in good_feedback:
#    pos = best_source_text
#    negs = random.sample(pool, k=5)
#    examples.append(InputExample(texts=[query, pos]))
#    # Для triplet/contrastive нужен другой формат

model = SentenceTransformer("BAAI/bge-multilingual-gemma2")
train_dataloader = DataLoader(examples, shuffle=True, batch_size=8)
train_loss = losses.MultipleNegativesRankingLoss(model)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=2, show_progress_bar=True, output_path="bge-finetuned-local")

#