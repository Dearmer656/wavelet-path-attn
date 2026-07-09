#!/bin/bash
#SBATCH --job-name=gen_ruler_8192
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_gen_ruler_8192.txt
#SBATCH --partition=lang_long
#SBATCH --account=lang
#SBATCH --nodelist=ahcclcsa01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00

set -euxo pipefail
if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
    set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

BASE=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long
OUTDIR=${BASE}/data

python3 - <<'PYEOF'
import json, random, sys
from pathlib import Path
from tokenizers import Tokenizer

INFILE = Path("/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/ruler_smoke_100.jsonl")
TOK_JSON = Path("/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/ruler_ft_matrix_20260525/path_pa_s42_save1k/tokenizer.json")
OUTFILE = Path("/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/ruler_eval_8192.jsonl")
TARGET_TOKENS = 8192
EVAL_COUNT = 200
SEED = 45

def load_jsonl(p):
    return [json.loads(l) for l in p.open() if l.strip()]

def count_tokens(tok, text):
    return len(tok.encode(text).ids)

def split_prompt(prompt):
    marker = r"\nQuestion:"
    idx = prompt.find(marker)
    return (prompt[:idx], prompt[idx:]) if idx >= 0 else (prompt, "")

def make_noise_line(rng, j):
    a = rng.choice(["amber","violet","silver","teal","indigo","navy","ivory","bronze"])
    b = rng.choice(["falcon","otter","lynx","maple","cedar","quartz","delta","omega"])
    c = rng.choice(["alpha","beta","gamma","kappa","theta","sigma","lambda","zeta"])
    return rf"\nNoise {j}: tag={a}_{j}; item={b}; flag={c}; code={rng.randint(10,999)}."

def stretch_prompt(tok, prompt, target_tokens, rng):
    prefix, qtail = split_prompt(prompt)
    lines = []
    j = 0
    cur = prefix + qtail
    cur_len = count_tokens(tok, cur)
    if cur_len >= target_tokens:
        return cur
    while cur_len < target_tokens:
        cand_lines = lines + [make_noise_line(rng, j)]
        cand = prefix + "".join(cand_lines) + qtail
        cand_len = count_tokens(tok, cand)
        if cand_len <= target_tokens:
            lines = cand_lines; cur = cand; cur_len = cand_len; j += 1
        else:
            break
    pad_words = ["note","entry","token","field","record","index","cache","value"]
    while cur_len < target_tokens:
        w = rng.choice(pad_words)
        cand = cur + rf"\nPad {w}."
        cand_len = count_tokens(tok, cand)
        if cand_len <= target_tokens:
            cur = cand; cur_len = cand_len
        else:
            break
    return cur

base_rows = load_jsonl(INFILE)
tok = Tokenizer.from_file(str(TOK_JSON))
rng = random.Random(SEED)
out = []
for i in range(EVAL_COUNT):
    src = dict(base_rows[i % len(base_rows)])
    stretched = stretch_prompt(tok, src["input"], TARGET_TOKENS, rng)
    out.append({"input": stretched, "outputs": src["outputs"],
                "ruler_config": src.get("ruler_config","unknown"), "length": TARGET_TOKENS})

with OUTFILE.open("w") as f:
    for row in out:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")

lens = [count_tokens(tok, r["input"]) for r in out]
print(f"[gen] ruler_eval_8192.jsonl: n={len(out)} min={min(lens)} median={sorted(lens)[len(lens)//2]} max={max(lens)}")
PYEOF

echo "=== Done: ruler_eval_8192.jsonl generated ==="
