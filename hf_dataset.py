#!/usr/bin/env python3
"""
Collect TOP and BOTTOM K Hugging Face DATASET metadata (stream-safe).

✅ Authenticates via HF_TOKEN
✅ Collects ALL datasets sorted by downloads (descending)
✅ Selects top K and bottom K
✅ Enriches each dataset (optional, safe)
✅ Saves every dataset as soon as it's fetched — no progress loss

Usage:
  export HF_TOKEN=hf_xxx
  pip install -U huggingface_hub
  python collect_top_bottom_datasets_streamed.py --topk 1000 --bottomk 1000 --enrich
"""

import argparse, json, os, sys, time
from huggingface_hub import HfApi, whoami
from typing import List, Dict, Set, Any


# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Collect and save top/bottom Hugging Face dataset metadata.")
    p.add_argument("--topk", type=int, default=1000)
    p.add_argument("--bottomk", type=int, default=1000)
    p.add_argument("--outdir", default="hf_ranked_stream")
    p.add_argument("--sleep", type=float, default=0.25)
    p.add_argument("--enrich", action="store_true")
    return p.parse_args()


# ---------------- Utility Functions ----------------
def log(msg: str):
    print(msg, flush=True)

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def write_jsonl_line(path: str, row: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def write_ids(path: str, ids: List[str]):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(ids) + "\n")

def basic_row(item) -> Dict[str, Any]:
    return {
        "id": item.id,
        "downloads": getattr(item, "downloads", None),
        "likes": getattr(item, "likes", None),
        "lastModified": getattr(item, "lastModified", None),
        "tags": getattr(item, "tags", None),
    }

def enrich_dataset(api: HfApi, repo_id: str, sleep: float) -> Dict[str, Any]:
    """Fetch detailed dataset metadata without downloading files."""
    try:
        info = api.dataset_info(repo_id, files_metadata=True)
    except Exception as e:
        return {"id": repo_id, "_error": str(e)}

    siblings = []
    for s in (getattr(info, "siblings", None) or []):
        try:
            siblings.append(s.to_dict())
        except Exception:
            siblings.append({})

    card = getattr(info, "cardData", None)
    out = {
        "id": repo_id,
        "downloads": getattr(info, "downloads", None),
        "likes": getattr(info, "likes", None),
        "lastModified": getattr(info, "lastModified", None),
        "tags": getattr(info, "tags", None),
        "license": getattr(info, "license", None),
        "languages": [t.split(":", 1)[1] for t in (info.tags or []) if t.startswith("language:")],
        "task_categories": [t.split(":", 1)[1] for t in (info.tags or []) if t.startswith("task_categories:")],
        "siblings": siblings,
        "card_sections_present": list(card.keys()) if isinstance(card, dict) else None,
    }
    time.sleep(sleep)
    return out


# ---------------- Main ----------------
def main():
    args = parse_args()
    ensure_dir(args.outdir)

    token = os.getenv("HF_TOKEN", "").strip()
    if not token.startswith("hf_"):
        log("❌ Please set your token: export HF_TOKEN=hf_XXXX")
        sys.exit(1)

    api = HfApi(token=token)
    try:
        me = whoami(token)
        log(f"✅ Authenticated as: {me.get('name','?')}")
    except Exception as e:
        log(f"❌ Invalid token or network issue: {e}")
        sys.exit(1)

    # Step 1: Collect all datasets sorted DESC by downloads
    log("📥 Fetching datasets sorted by downloads (desc)…")
    all_rows, seen = [], set()
    itr = api.list_datasets(sort="downloads", direction=-1, full=True)
    count = 0
    for item in itr:
        row = basic_row(item)
        rid = row["id"]
        if not rid or rid in seen:
            continue
        all_rows.append(row)
        seen.add(rid)
        count += 1
        if count % 2000 == 0:
            log(f"  …collected {count} so far")
            time.sleep(args.sleep)

    log(f"✅ Total datasets collected: {len(all_rows)}")

    # Step 2: Select top & bottom datasets
    top = all_rows[: args.topk]
    top_ids = {r["id"] for r in top}
    bottom_candidates = [r for r in reversed(all_rows) if r["id"] not in top_ids]
    bottom = bottom_candidates[: args.bottomk]

    # Step 3: Prepare output files (stream mode)
    top_path = os.path.join(args.outdir, "top_datasets_stream.jsonl")
    bottom_path = os.path.join(args.outdir, "bottom_datasets_stream.jsonl")
    write_ids(os.path.join(args.outdir, "top_ids.txt"), [r["id"] for r in top])
    write_ids(os.path.join(args.outdir, "bottom_ids.txt"), [r["id"] for r in bottom])

    log(f"Saving live to: {top_path} / {bottom_path}")

    # Step 4: Stream save enriched metadata
    if args.enrich:
        log("🔍 Enriching & saving TOP datasets…")
        for i, r in enumerate(top, start=1):
            meta = enrich_dataset(api, r["id"], args.sleep)
            write_jsonl_line(top_path, meta)
            if i % 50 == 0:
                log(f"  Saved {i}/{len(top)} top datasets")

        log("🔍 Enriching & saving BOTTOM datasets…")
        for i, r in enumerate(bottom, start=1):
            meta = enrich_dataset(api, r["id"], args.sleep)
            write_jsonl_line(bottom_path, meta)
            if i % 50 == 0:
                log(f"  Saved {i}/{len(bottom)} bottom datasets")
    else:
        # Just save the list entries
        for r in top:
            write_jsonl_line(top_path, r)
        for r in bottom:
            write_jsonl_line(bottom_path, r)

    log("🎉 Done — all saved safely.")
    log(f"  → {top_path}")
    log(f"  → {bottom_path}")
    log(f"  → IDs in {args.outdir}/top_ids.txt and bottom_ids.txt")


if __name__ == "__main__":
    main()
