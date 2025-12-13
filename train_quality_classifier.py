# train_quality_classifier.py
import numpy as np
import joblib
import json
from feedback_store import FeedbackStore
from sentence_transformers import SentenceTransformer
from local_config import EMBEDDING_MODEL_NAME
from sklearn.model_selection import train_test_split
import lightgbm as lgb

fs = FeedbackStore()
rows = fs.fetch_all(limit=100000)

model_emb = SentenceTransformer(EMBEDDING_MODEL_NAME)

def avg_cosine(q_emb, emb_list):
    if not emb_list:
        return 0.0
    sims = [np.dot(q_emb, e)/(np.linalg.norm(q_emb)*np.linalg.norm(e)+1e-9) for e in emb_list]
    return float(np.mean(sims))

X = []
y = []
for r in rows:
    q = r["question"] or ""
    ans = r["answer"] or ""
    sources = r.get("sources", []) or []
    num_sources = len(sources)
    avg_score = None
    scores = [s.get("score") for s in sources if s.get("score") is not None]
    avg_score = float(np.mean(scores)) if scores else 0.0

    # embeddings
    q_emb = model_emb.encode(q, convert_to_numpy=True, normalize_embeddings=True)
    src_texts = [s.get("text","") for s in sources][:5]
    src_embs = []
    if src_texts:
        src_embs = model_emb.encode(src_texts, convert_to_numpy=True, normalize_embeddings=True)
        avg_cos = avg_cosine(q_emb, src_embs)
    else:
        avg_cos = 0.0

    feats = [
        len(q),
        len(ans),
        num_sources,
        avg_score,
        avg_cos
    ]
    X.append(feats)
    y.append(int(r["feedback"]))

X = np.array(X)
y = np.array(y)
if len(y) < 10:
    print("Недостаточно фидбэка для обучения (нужно >=10).")
    exit(1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
dtrain = lgb.Dataset(X_train, label=y_train)
dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

params = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "boosting_type": "gbdt"
}
bst = lgb.train(params, dtrain, valid_sets=[dtrain,dval], num_boost_round=200, early_stopping_rounds=20)
joblib.dump(bst, "quality_model.pkl")
print("Модель сохранена: quality_model.pkl")