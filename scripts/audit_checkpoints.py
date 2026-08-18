"""Enumerate OLMo 3 / 3.1 (+ OLMo-2 early-training) repos and their checkpoint
branches on Hugging Face. Writes checkpoints.json (raw) and prints a summary.

Checkpoint intermediates live as git branches (refs) on each repo, e.g.
`stage1-step140000-tokens588B`. Density of RLVR/DPO intermediates determines
paper scope (see project.md section 4).
"""

import json
import re
import sys
from huggingface_hub import HfApi

api = HfApi()

# Repos we care about, found by author search rather than hardcoded names.
SEARCH_TERMS = ["Olmo-3", "Olmo-3.1", "OLMo-3"]
EXTRA_REPOS = ["allenai/OLMo-2-0425-1B"]  # early-training fine-grained suite

EXCLUDE_PAT = re.compile(r"gguf|awq|gptq|4bit|8bit|mlx", re.I)

repos = {}
for term in SEARCH_TERMS:
    for m in api.list_models(author="allenai", search=term):
        if not EXCLUDE_PAT.search(m.id):
            repos[m.id] = None
for r in EXTRA_REPOS:
    repos[r] = None

print(f"Found {len(repos)} candidate repos", file=sys.stderr)

out = {}
for repo_id in sorted(repos):
    try:
        refs = api.list_repo_refs(repo_id)
        branches = sorted(b.name for b in refs.branches)
        out[repo_id] = branches
        print(f"{repo_id}: {len(branches)} branches", file=sys.stderr)
    except Exception as e:
        out[repo_id] = f"ERROR: {e}"
        print(f"{repo_id}: ERROR {e}", file=sys.stderr)

with open("checkpoints.json", "w") as f:
    json.dump(out, f, indent=2)

# Compact summary to stdout
for repo_id, branches in out.items():
    if isinstance(branches, str):
        print(f"\n## {repo_id}\n  {branches}")
        continue
    print(f"\n## {repo_id}  ({len(branches)} branches)")
    non_main = [b for b in branches if b != "main"]
    # Show a sample: first 3, last 3
    if len(non_main) <= 8:
        for b in non_main:
            print(f"  {b}")
    else:
        for b in non_main[:4]:
            print(f"  {b}")
        print(f"  ... ({len(non_main) - 8} more) ...")
        for b in non_main[-4:]:
            print(f"  {b}")
