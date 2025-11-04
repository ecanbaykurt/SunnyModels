"""
open_model_datalake.py
Paste this into a .py file and run.

python open_model_datalake.py --hf_token hf_xxx --kaggle_username YOUR --kaggle_key KEY --limit 40

Outputs under ./data_lake
"""

import os
import re
import json
import csv
import argparse
import time
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any
import subprocess

import pandas as pd
from tqdm import tqdm
import networkx as nx

# HF Hub
from huggingface_hub import login, list_models, list_datasets, ModelCard, HfApi

# Embeddings & search
from sentence_transformers import SentenceTransformer
import faiss

# HTML report
from jinja2 import Template

# -----------------------------
# 0) CONFIG
# -----------------------------
CATEGORIES: Dict[str, List[str]] = {
    "Finance": ["finance", "financial", "stock", "bank", "credit", "risk"],
    "Healthcare": ["biomedical", "clinical", "medical", "healthcare", "bio"],
    "Mathematics": ["math", "mathematics", "algebra", "arithmetic"],
    "Marketing": ["marketing", "advertising", "social", "sentiment"],
    "Supply Chain": ["forecasting", "time-series", "demand", "inventory", "supply"],
    "Anomaly Analyses": ["anomaly", "outlier", "novelty"],
    "Data Security": ["pii", "deid", "privacy", "security"],
    "Fraud detection": ["fraud", "anti-fraud", "financial-crime", "aml"],
    "Material science": ["materials", "chemistry", "physics", "mat-sci"],
    "Education": ["education", "qa", "summarization", "tutor"],
    "Law (LLM)": ["legal", "law", "case", "contract"],
    "Environment": ["climate", "sustainability", "earth", "weather"],
    "Engineering": ["mechanical", "structural", "architecture", "software", "hardware", "civil"],
    "LLMs": ["instruct", "assistant", "agent", "orchestrator"],
    "Agents": ["agent", "multi-agent", "orchestration"]
}

# File type map for datasets
FILETYPE_PATTERNS = {
    "excel": [".xlsx", ".xls"],
    "csv": [".csv"],
    "json": [".jsonl", ".json"],
    "parquet": [".parquet"],
    "html": [".html", ".htm"],
    "markdown": [".md"],
    "text": [".txt"],
    "ts": [".ts", ".tsx"],
    "python": [".py"],
    "sql": [".sql"]
}

# -----------------------------
# 1) UTILITIES
# -----------------------------
def ensure_dirs(root: Path) -> Dict[str, Path]:
    paths = {
        "root": root,
        "raw": root / "raw",
        "metadata": root / "metadata",
        "descriptions": root / "descriptions",
        "embeddings": root / "embeddings",
        "exports": root / "exports",
        "graphs": root / "graphs",
        "frontend": root / "frontend"
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths

def set_kaggle_env(username: str, key: str):
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key

def run_kaggle_cmd(args: List[str]) -> Tuple[int, str, str]:
    """Run Kaggle CLI command and capture output."""
    proc = subprocess.Popen(["kaggle", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    return proc.returncode, out, err

def detect_modality(tags: List[str]) -> str:
    if not tags: return "text"
    t = set([x.lower() for x in tags])
    if any(x in t for x in ["image", "vision"]): return "image"
    if any(x in t for x in ["audio", "speech"]): return "audio"
    if any(x in t for x in ["time-series", "timeseries"]): return "time-series"
    if "tabular" in t: return "tabular"
    return "text"

def detect_category(name: str, tags: List[str], description: str) -> str:
    base = (name + " " + " ".join(tags or []) + " " + (description or "")).lower()
    for cat, keywords in CATEGORIES.items():
        for k in keywords:
            if k in base:
                return cat
    return "General"

def parse_metrics_from_text(text: str) -> Dict[str, Any]:
    """
    Heuristic: pull accuracy/F1/ROC-AUC/BLEU from model card text if explicitly present.
    """
    metrics = {}
    patterns = {
        "accuracy": r"(accuracy|acc)\s*[:=]\s*([0-9]*\.?[0-9]+)%?",
        "f1": r"(f1|f1-score)\s*[:=]\s*([0-9]*\.?[0-9]+)%?",
        "roc_auc": r"(roc[-_\s]?auc)\s*[:=]\s*([0-9]*\.?[0-9]+)%?",
        "bleu": r"(bleu)\s*[:=]\s*([0-9]*\.?[0-9]+)"
    }
    for k, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                val = float(m.group(2))
                metrics[k] = val
            except:
                pass
    return metrics

def list_filetypes(files: List[str]) -> List[str]:
    found = set()
    for f in files:
        lf = f.lower()
        for ftype, exts in FILETYPE_PATTERNS.items():
            if any(lf.endswith(ext) for ext in exts):
                found.add(ftype)
    return sorted(found)

def to_ts_export(obj: Any, var_name: str = "catalog") -> str:
    return f"export const {var_name} = {json.dumps(obj, indent=2)} as const;\n"

# -----------------------------
# 2) COLLECT FROM HUGGING FACE
# -----------------------------
def collect_hf(hf_token: str, limit: int, paths: Dict[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    login(token=hf_token)
    api = HfApi()

    # MODELS
    model_records = []
    pbar = tqdm(desc="HuggingFace models", total=len(CATEGORIES))
    for cat, kws in CATEGORIES.items():
        # for each keyword, fetch top-N
        for kw in set(kws):
            try:
                infos = list_models(filter=kw, sort="downloads", direction=-1, limit=limit)
                for info in infos:
                    # model card
                    desc_text = ""
                    try:
                        card = ModelCard.load(info.modelId)
                        desc_text = card.data.get("model_card_text", card.text) or ""
                    except Exception:
                        pass

                    metrics = parse_metrics_from_text(desc_text)
                    modality = detect_modality(getattr(info, "tags", []) or [])
                    category = detect_category(info.modelId, getattr(info, "tags", []) or [], desc_text)
                    license_ = (getattr(info, "cardData", {}) or {}).get("license", None)

                    # save description file
                    dfile = paths["descriptions"] / f"hf_model__{info.modelId.replace('/', '_')}.txt"
                    if desc_text:
                        dfile.write_text(desc_text, encoding="utf-8")

                    model_records.append({
                        "hub": "HuggingFace",
                        "kind": "model",
                        "category": category,
                        "name": info.modelId,
                        "url": f"https://huggingface.co/{info.modelId}",
                        "license": license_,
                        "downloads": getattr(info, "downloads", None),
                        "likes": getattr(info, "likes", None),
                        "tags": ",".join(getattr(info, "tags", []) or []),
                        "modality": modality,
                        "metrics": json.dumps(metrics),
                        "description_file": str(dfile) if desc_text else None
                    })
            except Exception as e:
                print("HF model error:", kw, e)
        pbar.update(1)
    pbar.close()
    hf_models_df = pd.DataFrame(model_records).drop_duplicates(subset=["name"])

    # DATASETS
    dataset_records = []
    pbar = tqdm(desc="HuggingFace datasets", total=len(CATEGORIES))
    for cat, kws in CATEGORIES.items():
        for kw in set(kws):
            try:
                dinfos = list_datasets(search=kw, limit=limit, sort="downloads", direction=-1)
                for dinfo in dinfos:
                    owner_repo = dinfo.id  # e.g., "owner/ds"
                    # list files to detect filetypes
                    ds_files = []
                    try:
                        ds_files = api.list_repo_files(repo_id=owner_repo, repo_type="dataset")
                    except Exception:
                        pass
                    ftypes = list_filetypes(ds_files)

                    # there's no standard dataset card loader; try README
                    desc_text = ""
                    try:
                        # Try fetch README (dataset card)
                        readme_path = api.dataset_info(owner_repo).cardData.get("model_card_text", "")
                        desc_text = readme_path or ""
                    except Exception:
                        pass

                    category = detect_category(owner_repo, dinfo.tags or [], desc_text)
                    dfile = paths["descriptions"] / f"hf_dataset__{owner_repo.replace('/', '_')}.txt"
                    if desc_text:
                        dfile.write_text(desc_text, encoding="utf-8")

                    dataset_records.append({
                        "hub": "HuggingFace",
                        "kind": "dataset",
                        "category": category,
                        "name": owner_repo,
                        "url": f"https://huggingface.co/datasets/{owner_repo}",
                        "license": (dinfo.cardData or {}).get("license", None),
                        "downloads": getattr(dinfo, "downloads", None),
                        "likes": getattr(dinfo, "likes", None),
                        "tags": ",".join(dinfo.tags or []),
                        "filetypes": ",".join(ftypes),
                        "description_file": str(dfile) if desc_text else None
                    })
            except Exception as e:
                print("HF dataset error:", kw, e)
        pbar.update(1)
    pbar.close()
    hf_datasets_df = pd.DataFrame(dataset_records).drop_duplicates(subset=["name"])

    return hf_models_df, hf_datasets_df

# -----------------------------
# 3) COLLECT FROM KAGGLE
# -----------------------------
def collect_kaggle(limit: int, paths: Dict[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Kaggle CLI returns CSVs; we'll parse them. We:
     - List datasets per keyword
     - List models (where available)
    """
    kaggle_ds_records = []
    kaggle_model_records = []

    # Datasets
    for cat, kws in CATEGORIES.items():
        for kw in set(kws):
            code, out, err = run_kaggle_cmd(["datasets", "list", "-s", kw, "-v"])
            if code != 0:
                continue
            # parse lines (CSV-like but CLI formatted); fallback to standard CSV parse not reliable; use --csv next:
            code, out, err = run_kaggle_cmd(["datasets", "list", "-s", kw, "--csv"])
            if code != 0 or not out.strip():
                continue
            # Load CSV from string
            rows = list(csv.DictReader(out.splitlines()))
            for r in rows[:limit]:
                ref = r.get("ref")  # owner/dataset
                if not ref:
                    continue
                # list files
                code2, out2, err2 = run_kaggle_cmd(["datasets", "files", ref, "--csv"])
                filetypes = []
                if code2 == 0 and out2.strip():
                    files = list(csv.DictReader(out2.splitlines()))
                    filetypes = list_filetypes([f.get("name","") for f in files])
                # simple description not provided directly; leave None
                kaggle_ds_records.append({
                    "hub": "Kaggle",
                    "kind": "dataset",
                    "category": detect_category(ref, [], ""),
                    "name": ref,
                    "url": f"https://www.kaggle.com/datasets/{ref}",
                    "license": r.get("licenseName"),
                    "downloads": r.get("downloadCount"),
                    "likes": r.get("voteCount"),
                    "tags": r.get("tags", ""),
                    "filetypes": ",".join(filetypes),
                    "description_file": None
                })
            time.sleep(0.5)

    # Models hub (limited public index; pull Gemma etc. as examples)
    for kw in ["gemma", "codegemma", "llama", "mistral"]:
        code, out, err = run_kaggle_cmd(["models", "list", "-s", kw, "--csv"])
        if code != 0 or not out.strip():
            continue
        rows = list(csv.DictReader(out.splitlines()))
        for r in rows[:limit]:
            ref = r.get("ref")
            if not ref:
                continue
            kaggle_model_records.append({
                "hub": "Kaggle",
                "kind": "model",
                "category": "LLMs",
                "name": ref,
                "url": f"https://www.kaggle.com/models/{ref}",
                "license": r.get("licenseName"),
                "downloads": r.get("downloadCount"),
                "likes": r.get("voteCount"),
                "tags": r.get("task", ""),
                "modality": "text",
                "metrics": None,
                "description_file": None
            })
        time.sleep(0.5)

    return (
        pd.DataFrame(kaggle_model_records).drop_duplicates(subset=["name"]),
        pd.DataFrame(kaggle_ds_records).drop_duplicates(subset=["name"])
    )

# -----------------------------
# 4) MATCHING (Models ↔ Datasets)
# -----------------------------
def match_models_datasets(models_df: pd.DataFrame, datasets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple overlap on category and any shared tag token.
    """
    def tokens(s: str) -> set:
        return set((s or "").lower().replace(",", " ").split())

    matches = []
    for _, m in models_df.iterrows():
        m_tags = tokens(m.get("tags",""))
        m_cat = m.get("category","General")
        for _, d in datasets_df.iterrows():
            if d.get("category","") != m_cat:
                continue
            d_tags = tokens(d.get("tags",""))
            if m_tags & d_tags or m_cat != "General":
                matches.append({
                    "model": m["name"],
                    "dataset": d["name"],
                    "category": m_cat
                })
    return pd.DataFrame(matches).drop_duplicates()

# -----------------------------
# 5) GRAPH, EXPORTS, SEARCH
# -----------------------------
def build_graph(models: pd.DataFrame, datasets: pd.DataFrame, match_df: pd.DataFrame, paths: Dict[str, Path]):
    G = nx.DiGraph()
    # category nodes
    for cat in sorted(set(list(models["category"]) + list(datasets["category"]))):
        G.add_node(f"cat::{cat}", kind="category")

    # items
    for _, r in models.iterrows():
        G.add_node(f"model::{r['name']}", kind="model", hub=r["hub"], url=r["url"])
        G.add_edge(f"cat::{r['category']}", f"model::{r['name']}")
    for _, r in datasets.iterrows():
        G.add_node(f"ds::{r['name']}", kind="dataset", hub=r["hub"], url=r["url"])
        G.add_edge(f"cat::{r['category']}", f"ds::{r['name']}")

    # matches
    for _, r in match_df.iterrows():
        G.add_edge(f"model::{r['model']}", f"ds::{r['dataset']}", kind="matches")

    nx.write_graphml(G, paths["graphs"] / "catalog.graphml")
    nx.readwrite.json_graph.node_link_data(G)  # if needed, can dump to JSON

def export_tables(models_hf: pd.DataFrame, models_kg: pd.DataFrame, ds_hf: pd.DataFrame, ds_kg: pd.DataFrame,
                  matches: pd.DataFrame, paths: Dict[str, Path]):
    exports = paths["exports"]
    # unify models
    models_all = pd.concat([models_hf, models_kg], ignore_index=True, sort=False)
    datasets_all = pd.concat([ds_hf, ds_kg], ignore_index=True, sort=False)

    # Excel (multi-sheet)
    with pd.ExcelWriter(exports / "catalog.xlsx", engine="xlsxwriter") as writer:
        models_all.to_excel(writer, index=False, sheet_name="Models")
        datasets_all.to_excel(writer, index=False, sheet_name="Datasets")
        matches.to_excel(writer, index=False, sheet_name="Matches")

    # CSV
    models_all.to_csv(exports / "models.csv", index=False)
    datasets_all.to_csv(exports / "datasets.csv", index=False)
    matches.to_csv(exports / "matches.csv", index=False)

    # JSON
    models_all.to_json(exports / "models.json", orient="records", indent=2)
    datasets_all.to_json(exports / "datasets.json", orient="records", indent=2)
    matches.to_json(exports / "matches.json", orient="records", indent=2)

    # HTML quick report
    html_tpl = Template("""
    <html><head><meta charset="utf-8"><title>Open Model/Data Catalog</title></head>
    <body>
      <h1>Open Model/Data Catalog</h1>
      <h2>Models ({{ models|length }})</h2>
      {{ models_tbl|safe }}
      <h2>Datasets ({{ datasets|length }})</h2>
      {{ datasets_tbl|safe }}
      <h2>Matches ({{ matches|length }})</h2>
      {{ matches_tbl|safe }}
      <p>Generated from Hugging Face + Kaggle.</p>
    </body></html>
    """)
    html = html_tpl.render(
        models=models_all.to_dict("records"),
        datasets=datasets_all.to_dict("records"),
        matches=matches.to_dict("records"),
        models_tbl=models_all.head(300).to_html(index=False),
        datasets_tbl=datasets_all.head(300).to_html(index=False),
        matches_tbl=matches.head(300).to_html(index=False)
    )
    (exports / "report.html").write_text(html, encoding="utf-8")

    # TypeScript export
    ts_obj = {
        "models": models_all.to_dict("records"),
        "datasets": datasets_all.to_dict("records"),
        "matches": matches.to_dict("records")
    }
    (paths["frontend"] / "catalog.ts").write_text(to_ts_export(ts_obj, "catalog"), encoding="utf-8")

    return models_all, datasets_all

def build_embeddings_and_faiss(descriptions_dir: Path, exports_dir: Path):
    txts = []
    ids = []
    for f in descriptions_dir.glob("*.txt"):
        try:
            txts.append(f.read_text(encoding="utf-8")[:5000])
            ids.append(f.stem)
        except Exception:
            pass
    if not txts:
        return None
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embs = embedder.encode(txts, show_progress_bar=True)
    index = faiss.IndexFlatL2(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, str(exports_dir / "descriptions.faiss"))
    json.dump({"ids": ids}, open(exports_dir / "id_map.json", "w"))
    return {"index_path": str(exports_dir / "descriptions.faiss"), "id_map": str(exports_dir / "id_map.json")}

# -----------------------------
# 6) MAIN ORCHESTRATOR
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, default=os.getenv("HUGGINGFACE_TOKEN", ""))
    parser.add_argument("--kaggle_username", type=str, default=os.getenv("KAGGLE_USERNAME", ""))
    parser.add_argument("--kaggle_key", type=str, default=os.getenv("KAGGLE_KEY", ""))
    parser.add_argument("--limit", type=int, default=30, help="Items per keyword per hub list call")
    parser.add_argument("--root", type=str, default="data_lake")
    args = parser.parse_args()

    root = Path(args.root)
    if root.exists() and not (root / ".ok").exists():
        # avoid accidental overwrite
        print(f"Using existing {root}. (Create {root}/.ok to skip this guard.)")
    paths = ensure_dirs(root)

    # Auth
    if args.hf_token:
        login(token=args.hf_token)
    if args.kaggle_username and args.kaggle_key:
        set_kaggle_env(args.kaggle_username, args.kaggle_key)

    # Collect
    print("==> Collecting from Hugging Face")
    hf_models_df, hf_datasets_df = collect_hf(args.hf_token, args.limit, paths)

    print("==> Collecting from Kaggle")
    kg_models_df, kg_datasets_df = collect_kaggle(args.limit, paths)

    # Export raw metadata snapshots
    hf_models_df.to_json(paths["metadata"] / "hf_models.json", orient="records", indent=2)
    hf_datasets_df.to_json(paths["metadata"] / "hf_datasets.json", orient="records", indent=2)
    kg_models_df.to_json(paths["metadata"] / "kaggle_models.json", orient="records", indent=2)
    kg_datasets_df.to_json(paths["metadata"] / "kaggle_datasets.json", orient="records", indent=2)

    # Matches
    print("==> Matching models ↔ datasets")
    models_all = pd.concat([hf_models_df, kg_models_df], ignore_index=True, sort=False)
    datasets_all = pd.concat([hf_datasets_df, kg_datasets_df], ignore_index=True, sort=False)
    matches_df = match_models_datasets(models_all, datasets_all)

    # Graph + Exports
    print("==> Building graph & exports")
    build_graph(models_all, datasets_all, matches_df, paths)
    models_unified, datasets_unified = export_tables(hf_models_df, kg_models_df, hf_datasets_df, kg_datasets_df, matches_df, paths)

    # Embeddings + FAISS
    print("==> Embeddings & FAISS")
    emb_info = build_embeddings_and_faiss(paths["descriptions"], paths["embeddings"])
    if emb_info:
        print("FAISS index:", emb_info["index_path"])
        print("ID map:", emb_info["id_map"])
    print("✅ Done. See:", root.resolve())

if __name__ == "__main__":
    main()

# -------------------------------------------
# OPTIONAL: FastAPI search (save below code in a separate server file if needed)
# -------------------------------------------
"""
from fastapi import FastAPI, Query
import faiss, json
from sentence_transformers import SentenceTransformer

app = FastAPI()
embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("data_lake/embeddings/descriptions.faiss")
id_map = json.load(open("data_lake/embeddings/id_map.json"))

@app.get("/search")
def search(q: str = Query(...), k: int = 5):
    emb = embedder.encode([q])
    D, I = index.search(emb, k)
    results = [{"id": id_map["ids"][int(i)], "score": float(d)} for i, d in zip(I[0], D[0])]
    return {"query": q, "results": results}
"""
