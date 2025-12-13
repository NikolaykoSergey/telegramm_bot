
from transformers import AutoModel, AutoTokenizer
import time, os
MODEL = "BAAI/bge-multilingual-gemma2"
os.makedirs("./offload", exist_ok=True)
print("Попытка загрузить через transformers (low_cpu_mem_usage + offload)...")
start = time.time()
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto",         # пусть accelerate сам разбросит веса, можно "cpu" тоже
        offload_folder="./offload"
    )
    print("Loaded OK")
    try:
        print("model.config.hidden_size =", model.config.hidden_size)
        # если есть encoder/decoder и нужно dim эмбеда — можно посмотреть output embedding
    except:
        pass
except Exception as e:
    print("Ошибка при загрузке:", e)
    raise
finally:
    print("Elapsed:", time.time() - start)
