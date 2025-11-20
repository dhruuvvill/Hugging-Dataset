#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Set

from huggingface_hub import HfApi, whoami
import mistune


# ====================== CLI & CONFIG ======================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Crawl Hugging Face models/datasets metadata + features (resume-safe)."
    )
    p.add_argument("--outdir", default="hf_features", help="Output directory (default: hf_features)")
    p.add_argument("--entities", default="models,datasets",
                   help="Comma-separated: models,datasets (default: both)")
    p.add_argument("--cutoff-days", type=int, default=180,
                   help="Exclude items modified within the past N days (default: 180)")
    p.add_argument("--sleep", type=float, default=0.5, help="Sleep between requests (default: 0.5s)")
    p.add_argument("--sleep-on-error", type=float, default=5.0, help="Sleep on transient errors (default: 5s)")
    p.add_argument("--sleep-on-429", type=float, default=60.0, help="Sleep when rate-limited (default: 60s)")
    return p.parse_args()


# ====================== UTILITIES ======================

def log(msg: str) -> None:
    # Use timezone-aware UTC to avoid deprecation warnings
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts} UTC] {msg}", flush=True)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_json_dumps(obj: Dict[str, Any]) -> str:
    def default(o):
        if isinstance(o, (dt.date, dt.datetime)):
            return o.isoformat()
        return str(o)
    return json.dumps(obj, ensure_ascii=False, default=default)


def write_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(safe_json_dumps(obj) + "\n")


def already_done_ids(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()
    done: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                # ignore malformed lines
                pass
    return done


def is_recent_iso8601(iso: Optional[str], now_utc: dt.datetime, cutoff_days: int) -> bool:
    if not iso:
        return False
    try:
        d = dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return (now_utc - d).days < cutoff_days
    except Exception:
        return False


def sum_model_weight_bytes(siblings: List[Dict[str, Any]]) -> int:
    exts = (".bin", ".safetensors", ".pt", ".ckpt")
    total = 0
    for s in siblings or []:
        name = s.get("rfilename") or s.get("path") or s.get("filename") or ""
        if any(name.endswith(ext) for ext in exts):
            try:
                total += int(s.get("size") or 0)
            except Exception:
                pass
    return total


# ---------- Markdown feature parsing ----------
LINK_RE = re.compile(r"\[(.*?)\]\((.*?)\)")
BASE_SUFFIX_RE = re.compile(r"(^|[-_/])base($|[-_])", re.I)
md_parser = mistune.create_markdown()  # not used deeply; regex-based counts below


def parse_markdown_feats(text: str) -> Dict[str, Any]:
    text = text or ""
    feats: Dict[str, Any] = {}
    feats["length_doc"] = len(text)

    code_blocks = text.count("```")
    feats["num_code_block"] = code_blocks
    feats["num_inline_code"] = max(0, text.count("`") - 3 * code_blocks)  # rough

    feats["num_lists"] = len(re.findall(r"^\s*[-*+]\s", text, re.M))
    feats["num_tables"] = text.count("|-") + text.count("| ---")
    feats["num_static_img"] = len(re.findall(r"!\[.*?\]\(.*?\.(png|jpg|jpeg)\)", text, re.I))
    feats["num_animation"] = len(re.findall(r"!\[.*?\]\(.*?\.gif\)", text, re.I))

    links = LINK_RE.findall(text)
    urls = [u for _, u in links]
    feats["num_gh_links"] = sum("github.com" in u for u in urls)
    feats["num_hf_links"] = sum("huggingface.co" in u for u in urls)
    feats["num_arxiv_links"] = sum(("arxiv.org" in u) or ("arXiv:" in u) for u in urls)
    feats["has_video"] = any(("youtube.com" in u) or ("vimeo.com" in u) for u in urls)

    feats["has_bibtex"] = ("@article" in text) or ("@inproceedings" in text)

    feats["length_yaml"] = 0
    if text.strip().startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            feats["length_yaml"] = len(parts[1])

    feats["has_license"] = ("license:" in text.lower()) or ("License" in text)
    return feats


# ====================== FILTER SWEEPS ======================

def default_model_filters() -> List[Optional[Iterable[str]]]:
    # None = no filter; strings = single filter; tuple/list = AND of filters
    return [None, "text-classification", "image-classification", "diffusers", "transformers"]


def default_dataset_filters() -> List[Optional[Iterable[str]]]:
    return [None, "task_categories:text-classification", "task_categories:translation", ("language:en",)]


# ====================== CRAWLERS ======================

class HFCrawler:
    def __init__(
        self,
        api: HfApi,
        outdir: str,
        cutoff_days_recent: int,
        sleep_between: float,
        sleep_on_error: float,
        sleep_on_429: float,
    ) -> None:
        self.api = api
        self.outdir = outdir
        self.cutoff_days_recent = cutoff_days_recent
        self.sleep_between = sleep_between
        self.sleep_on_error = sleep_on_error
        self.sleep_on_429 = sleep_on_429
        # Use timezone-aware UTC
        self.now_utc = dt.datetime.now(dt.timezone.utc)

        ensure_dir(outdir)
        self.idx_models = os.path.join(outdir, "index_models.jsonl")
        self.idx_datasets = os.path.join(outdir, "index_datasets.jsonl")
        self.feat_models = os.path.join(outdir, "features_models.jsonl")
        self.feat_datasets = os.path.join(outdir, "features_datasets.jsonl")

    # ---------- Index ----------
    def crawl_index(self, kind: str, filters: List[Optional[Iterable[str]]]) -> None:
        out_path = self.idx_models if kind == "models" else self.idx_datasets
        done = already_done_ids(out_path)
        log(f"⬇️  Index crawl {kind} (resume: {len(done)} ids) → {out_path}")

        for filt in filters:
            added = 0
            itr = (self.api.list_models(filter=filt, full=True)
                   if kind == "models"
                   else self.api.list_datasets(filter=filt, full=True))
            while True:
                try:
                    item = next(itr)
                except StopIteration:
                    break
                except Exception as e:
                    log(f"⚠️ index error ({kind}, filter={filt}): {e} → sleep & retry")
                    time.sleep(self.sleep_on_error)
                    continue

                row = {
                    "id": item.id,
                    "likes": getattr(item, "likes", None),
                    "downloads": getattr(item, "downloads", None),
                    "lastModified": getattr(item, "lastModified", None),
                    "tags": getattr(item, "tags", None),
                }
                if row["id"] in done:
                    continue
                write_jsonl(out_path, row)
                done.add(row["id"])
                added += 1

                if added % 200 == 0:
                    time.sleep(self.sleep_between)

            log(f"  ✅ filter {filt or '(none)'} → +{added} {kind}")

        log(f"✅ index complete for {kind}: total unique ids = {len(done)}")

    # ---------- Detail (models) ----------
    def enrich_model(self, mid: str) -> Dict[str, Any]:
        info = self.api.model_info(mid, files_metadata=True)

        if is_recent_iso8601(getattr(info, "lastModified", None), self.now_utc, self.cutoff_days_recent):
            return {"id": mid, "_skip_recent": True, "lastModified": getattr(info, "lastModified", None)}

        siblings = [s.to_dict() for s in (info.siblings or [])]

        total_weight = sum_model_weight_bytes(siblings)
        has_safetensors = any((s.get("rfilename", "").endswith(".safetensors")) for s in siblings)
        num_model_files = sum(
            (s.get("rfilename", "").endswith((".bin", ".safetensors", ".pt", ".ckpt")))
            for s in siblings
        )
        root_files = sum(1 for s in siblings if "/" not in (s.get("rfilename", "") or ""))
        modules = len({(s.get("rfilename", "").split("/")[0]) for s in siblings if "/" in (s.get("rfilename", "") or "")})

        readme_text = getattr(info, "readme", "") or ""
        md = parse_markdown_feats(readme_text)

        tags = set(info.tags or [])
        has_space = any("spaces" in t for t in tags)
        has_widgets = bool(getattr(info, "widgetData", None))
        pipeline = getattr(info, "pipeline_tag", None)
        library = getattr(info, "library_name", None)
        is_base = bool(BASE_SUFFIX_RE.search(getattr(info, "modelId", mid))) or ("base_model" in (getattr(info, "cardData", {}) or {}))

        return {
            "id": mid,
            "downloads": getattr(info, "downloads", None),
            "likes": getattr(info, "likes", None),
            "lastModified": getattr(info, "lastModified", None),

            **md,  # documentation features

            "model_size_bytes": int(total_weight),
            "num_model_files": int(num_model_files),
            "num_root_files": int(root_files),
            "num_modules": int(modules),
            "has_config": any(
                name in (s.get("rfilename", "") or "")
                for s in siblings
                for name in ("config.json", "model_index.json")
            ),
            "has_results": ("Results" in readme_text) or ("Evaluation" in readme_text),
            "has_dataset": ("dataset" in (getattr(info, "cardData", {}) or {})) or ("Dataset" in readme_text),
            "num_dataset_links_readme": md["num_hf_links"],  # proxy for dataset references
            "match_hf_dataset": md["num_hf_links"] > 0,
            "has_quantized": any(("int8" in (s.get("rfilename", "") or "")) or ("gguf" in (s.get("rfilename", "") or "")) for s in siblings),

            "has_space": bool(has_space),
            "has_safetensors": bool(has_safetensors),
            "has_widgets": bool(has_widgets),

            "has_pipeline": bool(pipeline),
            "has_impl_lib": bool(library),
            "impl_lib": library,
            "pipeline_tag": pipeline,

            "is_base_model": bool(is_base),
        }

    # ---------- Detail (datasets) ----------
    def enrich_dataset(self, did: str) -> Dict[str, Any]:
        info = self.api.dataset_info(did, files_metadata=True)

        if is_recent_iso8601(getattr(info, "lastModified", None), self.now_utc, self.cutoff_days_recent):
            return {"id": did, "_skip_recent": True, "lastModified": getattr(info, "lastModified", None)}

        card = getattr(info, "cardData", None) or {}
        card_text = json.dumps(card, ensure_ascii=False) if isinstance(card, dict) else (str(card) or "")
        md = parse_markdown_feats(card_text)

        tags = set(info.tags or [])
        languages = [t.split(":", 1)[1] for t in tags if t.startswith("language:")]
        task_categories = [t.split(":", 1)[1] for t in tags if t.startswith("task_categories:")]

        cfgs = getattr(info, "configs", None) or []
        try:
            num_splits = sum((len(getattr(c, "splits", []) or []) for c in cfgs))
        except Exception:
            num_splits = None

        siblings = [s.to_dict() for s in (info.siblings or [])]
        total_repo_bytes = 0
        for s in siblings:
            try:
                total_repo_bytes += int(s.get("size") or 0)
            except Exception:
                pass

        license_val = getattr(info, "license", None)
        if not license_val and isinstance(card, dict):
            license_val = card.get("license")

        has_citation = False
        if isinstance(card, dict):
            j = json.dumps(card, ensure_ascii=False).lower()
            has_citation = bool(card.get("citation")) or ("bibtex" in j)
        else:
            has_citation = ("citation" in card_text.lower()) or ("@article" in card_text) or ("@inproceedings" in card_text)

        return {
            "id": did,
            "downloads": getattr(info, "downloads", None),
            "likes": getattr(info, "likes", None),
            "lastModified": getattr(info, "lastModified", None),

            **md,  # documentation-like counts from card text

            "num_configs": len(cfgs),
            "num_splits": num_splits,
            "total_repo_bytes": int(total_repo_bytes),

            "num_languages": len(languages),
            "languages": languages,
            "task_categories": task_categories,

            "license": license_val,
            "has_citation": bool(has_citation),
            "card_sections_present": list(card.keys()) if isinstance(card, dict) else [],
        }

    # ---------- Detail driver ----------
    def process_detail(self, index_path: str, out_path: str, enricher) -> None:
        done = already_done_ids(out_path)
        wrote = 0

        if not os.path.exists(index_path):
            log(f"⚠️ missing index file: {index_path}")
            return

        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rid = json.loads(line)["id"]
                except Exception:
                    continue
                if rid in done:
                    continue

                try:
                    rec = enricher(rid)
                except Exception as e:
                    rec = {"id": rid, "_error": str(e)}

                write_jsonl(out_path, rec)
                wrote += 1

                if wrote % 50 == 0:
                    time.sleep(self.sleep_between)

        log(f"✅ detail complete → wrote/updated {wrote} rows to {out_path}")


# ====================== MAIN ======================

def main() -> None:
    args = parse_args()

    # Read token from standard env var name
    token = os.getenv("HF_TOKEN", "").strip()
    if not token.startswith("hf_"):
        log("❌ HF_TOKEN is missing or invalid. Set it with: export HF_TOKEN=hf_XXXX")
        sys.exit(1)

    # Auth sanity check
    try:
        me = whoami(token)
        log(f"✅ Auth as: {me.get('name', '?')}")
    except Exception as e:
        log(f"❌ Token invalid or network issue: {e}")
        sys.exit(1)

    api = HfApi(token=token)
    crawler = HFCrawler(
        api=api,
        outdir=args.outdir,
        cutoff_days_recent=args.cutoff_days,
        sleep_between=args.sleep,
        sleep_on_error=args.sleep_on_error,
        sleep_on_429=args.sleep_on_429,
    )

    entities = {e.strip().lower() for e in args.entities.split(",") if e.strip()}
    do_models = "models" in entities
    do_datasets = "datasets" in entities

    # Stage A: index
    if do_models:
        log("=== Stage A: index crawl (models) ===")
        crawler.crawl_index("models", default_model_filters())
    if do_datasets:
        log("=== Stage A: index crawl (datasets) ===")
        crawler.crawl_index("datasets", default_dataset_filters())

    # Stage B: details
    if do_models:
        log("=== Stage B: enrich details (models) ===")
        crawler.process_detail(crawler.idx_models, crawler.feat_models, crawler.enrich_model)
    if do_datasets:
        log("=== Stage B: enrich details (datasets) ===")
        crawler.process_detail(crawler.idx_datasets, crawler.feat_datasets, crawler.enrich_dataset)

    log(f"🎉 Done. Outputs in: {args.outdir}")


if __name__ == "__main__":
    main()