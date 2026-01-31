"""
Fetch & curate a *large* PC parts catalog from an online dataset (optional).

Source dataset (MIT license): docyx/pc-part-dataset (PCPartPicker scrape)
- The repository provides parts in JSON/JSONL/CSV under ./data and a json.zip bundle.

This script downloads `data/json.zip` from GitHub, extracts only the categories we need,
curates a manageable subset (hundreds of parts, not tens of thousands), and writes:

  data/catalog_curated.json

The project will automatically use this curated catalog if present; otherwise it
falls back to the small built-in catalog.

Usage:
  python scripts/fetch_catalog.py --max-per-category 150

Notes:
- The mapping from raw fields -> attributes (socket, DDR, etc.) is heuristic,
  because upstream schemas vary slightly by category. The goal is to keep
  *constraint-relevant* attributes.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import urlopen, Request


DATASET_ZIP_URL = "https://raw.githubusercontent.com/docyx/pc-part-dataset/main/data/json.zip"

VARIABLES = ["CPU", "Motherboard", "RAM", "GPU", "PSU", "Case", "Storage"]


def _get(d: Dict[str, Any], *keys: str) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def _as_int_price(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return int(round(v))
    s = str(v)
    s = s.replace(",", ".")
    m = re.search(r"(\d+(\.\d+)?)", s)
    if not m:
        return None
    return int(round(float(m.group(1))))


def _find_str_with(d: Dict[str, Any], token: str) -> Optional[str]:
    token = token.lower()
    for k, v in d.items():
        if isinstance(v, str) and token in v.lower():
            return v
    return None


def _infer_socket(item: Dict[str, Any]) -> Optional[str]:
    # look for common keys
    v = _get(item, "socket", "cpu_socket", "socket_cpu", "socketType", "socket_type")
    if isinstance(v, str):
        return v.strip().lower().replace(" ", "")
    # fallback: scan strings
    for token in ["lga1700", "am4", "am5"]:
        s = _find_str_with(item, token)
        if s:
            return token
    return None


def _infer_ddr(item: Dict[str, Any]) -> Optional[str]:
    for token in ["ddr4", "ddr5"]:
        s = _find_str_with(item, token)
        if s:
            return token
    # sometimes "Memory Type" or "Modules" fields
    v = _get(item, "memory_type", "type", "modules", "speed")
    if isinstance(v, str):
        if "ddr4" in v.lower():
            return "ddr4"
        if "ddr5" in v.lower():
            return "ddr5"
    return None


def _infer_form_factor(item: Dict[str, Any]) -> Optional[str]:
    # motherboards/cases often use "form_factor" or "type"
    v = _get(item, "form_factor", "formFactor", "type", "case_type")
    if isinstance(v, str):
        s = v.lower()
        if "micro" in s or "m-atx" in s or "matx" in s:
            return "matx"
        if "atx" in s:
            return "atx"
    # scan strings
    if _find_str_with(item, "micro-atx") or _find_str_with(item, "matx"):
        return "matx"
    if _find_str_with(item, "atx"):
        return "atx"
    return None


def _infer_psu_watt(item: Dict[str, Any]) -> Optional[int]:
    v = _get(item, "wattage", "watts", "watt", "power", "capacity_w")
    p = _as_int_price(v)
    if p:
        # wattage should be in hundreds
        if 300 <= p <= 2000:
            return p
    # sometimes part name has "750 W"
    name = str(_get(item, "name") or "")
    m = re.search(r"(\d{3,4})\s*W", name, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _infer_gpu_psu_min(item: Dict[str, Any]) -> Optional[int]:
    # dataset sometimes has "psu" or "recommended_wattage"
    v = _get(item, "recommended_psu", "recommended_wattage", "psu", "min_psu", "tdp")
    w = _as_int_price(v)
    if w and 300 <= w <= 2000:
        # crude: if they provide TDP, translate to PSU min
        if w < 250:
            return 450
        if w < 350:
            return 550
        if w < 450:
            return 650
        return 750
    # fallback: scan name
    name = str(_get(item, "name") or "").lower()
    # very crude tiers
    if "4090" in name or "4080" in name or "7900 xtx" in name:
        return 850
    if "4070" in name or "7800 xt" in name or "7900 xt" in name:
        return 750
    if "4060" in name or "7600" in name:
        return 550
    return 650  # safe default


def _infer_storage_kind(item: Dict[str, Any]) -> str:
    name = str(_get(item, "name") or "").lower()
    interface = str(_get(item, "interface") or "").lower()
    if "nvme" in name or "nvme" in interface or "m.2" in name:
        return "nvme"
    if "ssd" in name or "solid" in name:
        return "ssd"
    return "hdd"


def _infer_storage_size_gb(item: Dict[str, Any]) -> Optional[int]:
    v = _get(item, "capacity", "capacity_gb", "size", "storage")
    if isinstance(v, (int, float)):
        return int(v)
    s = str(v or _get(item, "name") or "")
    m = re.search(r"(\d+)\s*TB", s, re.IGNORECASE)
    if m:
        return int(m.group(1)) * 1000
    m = re.search(r"(\d+)\s*GB", s, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _download_zip(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "nscc-fetch/1.0"})
    with urlopen(req) as r:
        return r.read()


def _pick_member(namelist: List[str], keywords: List[str]) -> Optional[str]:
    # choose first match where all keywords appear
    for n in namelist:
        ln = n.lower()
        if all(k in ln for k in keywords):
            return n
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-category", type=int, default=150)
    ap.add_argument("--out", type=str, default="data/catalog_curated.json")
    args = ap.parse_args()

    print(f"Downloading dataset zip from:\n  {DATASET_ZIP_URL}")
    blob = _download_zip(DATASET_ZIP_URL)
    zf = zipfile.ZipFile(io.BytesIO(blob))

    namelist = zf.namelist()

    # try to find JSON members
    members = {
        "CPU": _pick_member(namelist, ["cpu", ".json"]),
        "Motherboard": _pick_member(namelist, ["motherboard", ".json"]),
        "RAM": _pick_member(namelist, ["memory", ".json"]),
        "GPU": _pick_member(namelist, ["video", "card", ".json"]) or _pick_member(namelist, ["gpu", ".json"]),
        "PSU": _pick_member(namelist, ["power", "supply", ".json"]),
        "Case": _pick_member(namelist, ["case", ".json"]),
        "Storage": _pick_member(namelist, ["storage", ".json"]) or _pick_member(namelist, ["internal", "hard", "drive", ".json"]),
    }

    missing = [k for k, v in members.items() if v is None]
    if missing:
        print("WARNING: some categories were not found in zip:", missing)
        print("Zip members sample:", namelist[:30])

    # Load items for each category
    raw: Dict[str, List[Dict[str, Any]]] = {}
    for cat, mem in members.items():
        if mem is None:
            raw[cat] = []
            continue
        with zf.open(mem) as f:
            raw[cat] = json.loads(f.read().decode("utf-8"))

    # curate: keep items with price and essential attrs
    domains: Dict[str, List[str]] = {v: [] for v in VARIABLES}
    prices: Dict[str, int] = {}
    display: Dict[str, str] = {}

    # attrs for constraints
    cpu_socket: Dict[str, str] = {}
    mb_socket: Dict[str, str] = {}
    mb_ram: Dict[str, str] = {}
    mb_ff: Dict[str, str] = {}
    ram_std: Dict[str, str] = {}
    gpu_psu_min: Dict[str, int] = {}
    psu_watt: Dict[str, int] = {}
    case_max_ff: Dict[str, str] = {}

    def add(var: str, vid: str, name: str, price: int) -> None:
        domains[var].append(vid)
        prices[vid] = int(price)
        display[vid] = name

    # CPU
    for idx, it in enumerate(raw["CPU"][: args.max_per_category * 5]):
        name = str(_get(it, "name") or "").strip()
        if not name:
            continue
        price = _as_int_price(_get(it, "price"))
        if price is None:
            continue
        sock = _infer_socket(it)
        if sock not in ("lga1700", "am4", "am5"):
            continue
        vid = f"cpu_ext_{idx}"
        add("CPU", vid, name, price)
        cpu_socket[vid] = sock
        if len(domains["CPU"]) >= args.max_per_category:
            break

    # Motherboard
    for idx, it in enumerate(raw["Motherboard"][: args.max_per_category * 8]):
        name = str(_get(it, "name") or "").strip()
        price = _as_int_price(_get(it, "price"))
        if not name or price is None:
            continue
        sock = _infer_socket(it)
        ddr = _infer_ddr(it)
        ff = _infer_form_factor(it)
        if sock not in ("lga1700", "am4", "am5") or ddr not in ("ddr4", "ddr5") or ff not in ("matx", "atx"):
            continue
        # basic platform sanity: AM4 typically DDR4, AM5 DDR5 (allow exceptions but prefer these)
        if sock == "am4" and ddr != "ddr4":
            continue
        if sock == "am5" and ddr != "ddr5":
            continue
        vid = f"mb_ext_{idx}"
        add("Motherboard", vid, name, price)
        mb_socket[vid] = sock
        mb_ram[vid] = ddr
        mb_ff[vid] = ff
        if len(domains["Motherboard"]) >= args.max_per_category:
            break

    # RAM
    for idx, it in enumerate(raw["RAM"][: args.max_per_category * 8]):
        name = str(_get(it, "name") or "").strip()
        price = _as_int_price(_get(it, "price"))
        if not name or price is None:
            continue
        ddr = _infer_ddr(it)
        if ddr not in ("ddr4", "ddr5"):
            continue
        vid = f"ram_ext_{idx}"
        add("RAM", vid, name, price)
        ram_std[vid] = ddr
        if len(domains["RAM"]) >= args.max_per_category:
            break

    # GPU
    for idx, it in enumerate(raw["GPU"][: args.max_per_category * 8]):
        name = str(_get(it, "name") or "").strip()
        price = _as_int_price(_get(it, "price"))
        if not name or price is None:
            continue
        vid = f"gpu_ext_{idx}"
        add("GPU", vid, name, price)
        gpu_psu_min[vid] = int(_infer_gpu_psu_min(it) or 650)
        if len(domains["GPU"]) >= args.max_per_category:
            break

    # PSU
    for idx, it in enumerate(raw["PSU"][: args.max_per_category * 8]):
        name = str(_get(it, "name") or "").strip()
        price = _as_int_price(_get(it, "price"))
        if not name or price is None:
            continue
        w = _infer_psu_watt(it)
        if w is None:
            continue
        vid = f"psu_ext_{idx}"
        add("PSU", vid, name, price)
        psu_watt[vid] = int(w)
        if len(domains["PSU"]) >= args.max_per_category:
            break

    # Case (no compact: keep only ATX/mATX towers)
    for idx, it in enumerate(raw["Case"][: args.max_per_category * 10]):
        name = str(_get(it, "name") or "").strip()
        price = _as_int_price(_get(it, "price"))
        if not name or price is None:
            continue
        ff = _infer_form_factor(it)
        if ff not in ("matx", "atx"):
            continue
        vid = f"case_ext_{idx}"
        add("Case", vid, name, price)
        case_max_ff[vid] = ff
        if len(domains["Case"]) >= args.max_per_category:
            break

    # Storage
    for idx, it in enumerate(raw["Storage"][: args.max_per_category * 10]):
        name = str(_get(it, "name") or "").strip()
        price = _as_int_price(_get(it, "price"))
        if not name or price is None:
            continue
        size_gb = _infer_storage_size_gb(it)
        if size_gb is None:
            continue
        kind = _infer_storage_kind(it)
        vid = f"storage_ext_{idx}"
        # keep kind/size in name for readability
        pretty = f"{name} ({kind.upper()} {size_gb}GB)"
        add("Storage", vid, pretty, price)
        if len(domains["Storage"]) >= args.max_per_category:
            break

    out = {
        "meta": {
            "source": "docyx/pc-part-dataset (GitHub, MIT)",
            "source_url": "https://github.com/docyx/pc-part-dataset",
            "download_url": DATASET_ZIP_URL,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "max_per_category": args.max_per_category,
            "counts": {k: len(v) for k, v in domains.items()},
        },
        "variables": VARIABLES,
        "domains": domains,
        "prices": prices,
        "display_name": display,
        "attrs": {
            "cpu_socket": cpu_socket,
            "mb_socket": mb_socket,
            "mb_ram": mb_ram,
            "mb_ff": mb_ff,
            "ram_std": ram_std,
            "gpu_psu_min": gpu_psu_min,
            "psu_watt": psu_watt,
            "case_max_ff": case_max_ff,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\nDONE. Wrote curated catalog to:", out_path)
    print("Counts:", out["meta"]["counts"])
    print("\nNow run:")
    print("  python scripts/gui.py")
    print("or:")
    print("  python scripts/run_experiment.py --plot")


if __name__ == "__main__":
    main()
