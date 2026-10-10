"""Compare determinism_check outputs pairwise.

usage: compare_determinism.py <a.json> <b.json> [<c.json> ...]   (compares every file against the first)
"""

import json
import sys

ref_path = sys.argv[1]
ref = json.load(open(ref_path))
for path in sys.argv[2:]:
    other = json.load(open(path))
    identical = 0
    first_diff = []
    max_lp = 0.0
    for a, b in zip(ref, other):
        assert (a["seed"], a["length"]) == (b["seed"], b["length"])
        if a["token_ids"] == b["token_ids"]:
            identical += 1
            max_lp = max(max_lp, max(abs(x - y) for x, y in zip(a["logprobs"], b["logprobs"])))
        else:
            pos = next(i for i, (x, y) in enumerate(zip(a["token_ids"], b["token_ids"])) if x != y)
            first_diff.append(f"len={a['length']} seed={a['seed']} first_diff_at={pos}")
    print(
        f"{path.split('/')[-1]} vs {ref_path.split('/')[-1]}: identical token ids {identical}/{len(ref)}, "
        f"max |logprob diff| on identical prompts {max_lp:.2e}" + (f"; differing: {first_diff}" if first_diff else "")
    )
