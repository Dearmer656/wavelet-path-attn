import json
from transformers import AutoTokenizer

JSONL = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev_uniform.jsonl"
tok = AutoTokenizer.from_pretrained("gpt2")

examples = []
with open(JSONL) as f:
    for line in f:
        ex = json.loads(line)
        ctx = ex.get("context")
        if not ctx:
            continue
        text = " ".join(sent for _title, sents in ctx for sent in sents)
        ids = tok(text, return_tensors="pt")["input_ids"]
        if ids.shape[1] >= 2048:
            examples.append(ids)
        if len(examples) >= 3:
            break

for i, ids in enumerate(examples):
    print(f"=== example {i} (total tokens={ids.shape[1]}) ===")
    window = tok.decode(ids[0, 470:630])
    print(window)
    print()
