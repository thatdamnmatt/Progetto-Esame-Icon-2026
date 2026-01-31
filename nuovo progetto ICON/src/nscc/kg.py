from __future__ import annotations

"""Knowledge Graph + catalog per il dominio "PC configuration".

Il progetto punta a un dominio *realistico* ma gestibile in ~25 ore.

- **Catalogo**: insieme finito di componenti con nomi "umani" e prezzi.
- **KB/KG**: regole di compatibilità (socket CPU/MB, DDR4/DDR5, PSU↔GPU, ecc.).
- **Estrazione vincoli**: la KB viene compilata in vincoli utilizzabili dal CSP.

La KB non è usata come DB: contiene assiomi/relazioni che inducono vincoli non banali.
"""

from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Dict, List, Tuple

from rdflib import Graph, Namespace, RDF, URIRef

NSCC = Namespace("http://example.org/nscc/")


@dataclass(frozen=True)
class Catalog:
    """Catalogo dei componenti (domini delle variabili, prezzi e attributi utili).

    Nota: i valori (IDs) sono stringhe stabili e vengono usati in ML/CSP.
    `display_name` serve solo per output leggibile in CLI/GUI.
    """

    variables: List[str]
    domains: Dict[str, List[str]]
    prices: Dict[str, int]  # value_id -> price EUR
    display_name: Dict[str, str]  # value_id -> pretty label

    # attributi per regole (usati sia per feature sintetiche sia per vincoli)
    cpu_socket: Dict[str, str]  # cpu_id -> socket
    mb_socket: Dict[str, str]  # mb_id -> socket
    mb_ram: Dict[str, str]  # mb_id -> ddr4/ddr5
    ram_type: Dict[str, str]  # ram_id -> ddr4/ddr5
    gpu_psu_min: Dict[str, int]  # gpu_id -> W min consigliati
    psu_watt: Dict[str, int]  # psu_id -> W

    # score/metadata per ottimizzazione (proxy prestazioni)
    cpu_score: Dict[str, int]  # 0..100
    gpu_score: Dict[str, int]  # 0..100
    ram_gb: Dict[str, int]
    storage_gb: Dict[str, int]


@dataclass(frozen=True)
class ConstraintSpec:
    """Vincoli estratti dalla KB in forma utilizzabile dal CSP.

    - `incompatible`: lista di coppie ground che NON possono coesistere.
      Ogni elemento e' ((var1, value1), (var2, value2)).
    - `requires`: lista di implicazioni ground. Ogni elemento e'
      ((var1, value1), (var2, value2)) e significa:
        se var1=value1 allora var2=value2.

    Nota: più implicazioni con stesso antecedente e stesso consequente-var
    rappresentano una OR (var2 può assumere *uno qualunque* dei valori ammessi).
    """

    incompatible: List[Tuple[Tuple[str, str], Tuple[str, str]]]
    requires: List[Tuple[Tuple[str, str], Tuple[str, str]]]


def _u(name: str) -> URIRef:
    return NSCC[name]




def load_curated_catalog(path: str | Path) -> Catalog:
    """Carica un catalogo curato (generato da scripts/fetch_catalog.py).

    Il catalogo curato contiene molti individui e gli attributi minimi per
    generare i vincoli (socket/DDR/watt/form factor). Per mantenere la
    coerenza con il resto del progetto, qui ricaviamo anche proxy di
    "prestazioni" (score) e dimensioni (RAM/storage GB) con euristiche
    leggere e deterministiche.
    """
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))

    variables = data["variables"]
    domains = {k: list(v) for k, v in data["domains"].items()}
    prices = {k: int(v) for k, v in data["prices"].items()}
    display_name = {k: str(v) for k, v in data["display_name"].items()}

    attrs = data.get("attrs", {})

    def parse_gb(label: str) -> int | None:
        """Estrae una capacità in GB da una stringa (es. '32GB', '2x16GB', '1TB')."""
        s = label.lower()
        m = re.search(r"(\d+)\s*tb", s)
        if m:
            return int(m.group(1)) * 1000
        # prendi il massimo tra eventuali occorrenze (es. '2x16GB' -> 16, ma spesso c'è anche '32GB')
        gbs = [int(x) for x in re.findall(r"(\d+)\s*gb", s)]
        if gbs:
            return max(gbs)
        return None

    # RAM size (GB)
    ram_gb: Dict[str, int] = {}
    for rid in domains.get("RAM", []):
        gb = parse_gb(display_name.get(rid, ""))
        ram_gb[rid] = int(gb) if gb is not None else 16

    # Storage size (GB)
    storage_gb: Dict[str, int] = {}
    for sid in domains.get("Storage", []):
        gb = parse_gb(display_name.get(sid, ""))
        storage_gb[sid] = int(gb) if gb is not None else 1000

    def quantile_scores(ids: List[str], lo: int = 25, hi: int = 95) -> Dict[str, int]:
        """Score 0..100 deterministico basato sul quantile del prezzo (proxy prestazioni)."""
        if not ids:
            return {}
        # ordina per prezzo
        ordered = sorted(ids, key=lambda k: prices.get(k, 0))
        n = len(ordered)
        out: Dict[str, int] = {}
        for i, k in enumerate(ordered):
            q = 0.0 if n == 1 else i / (n - 1)
            out[k] = int(round(lo + (hi - lo) * q))
        return out

    cpu_score = quantile_scores(domains.get("CPU", []), lo=28, hi=96)
    gpu_score = quantile_scores(domains.get("GPU", []), lo=20, hi=97)

    # RAM type: map ram_std -> ram_type (same semantics)
    ram_type = {k: v for k, v in attrs.get("ram_std", {}).items() if isinstance(v, str)}

    cat = Catalog(
        variables=variables,
        domains=domains,
        prices=prices,
        display_name=display_name,
        cpu_socket=attrs.get("cpu_socket", {}),
        mb_socket=attrs.get("mb_socket", {}),
        mb_ram=attrs.get("mb_ram", {}),
        ram_type=ram_type,
        gpu_psu_min={k: int(v) for k, v in attrs.get("gpu_psu_min", {}).items()},
        psu_watt={k: int(v) for k, v in attrs.get("psu_watt", {}).items()},
        cpu_score=cpu_score,
        gpu_score=gpu_score,
        ram_gb=ram_gb,
        storage_gb=storage_gb,
    )
    return cat


def build_default_catalog(seed: int = 0) -> Catalog:
    """Crea un catalogo.

    Se esiste `data/catalog_curated.json` (generato da scripts/fetch_catalog.py),
    viene caricato automaticamente per ottenere un dominio *molto più ampio*.
    Altrimenti usa il catalogo interno (più piccolo) come fallback.
    """

    curated_path = Path(__file__).resolve().parents[2] / "data" / "catalog_curated.json"
    if curated_path.exists():
        try:
            cat = load_curated_catalog(curated_path)
            # Validazione: se il catalogo curato è incompleto (domini vuoti), usa fallback interno.
            if not _catalog_is_valid(cat):
                # fallback
                pass
            else:
                return cat
        except Exception:
            # qualsiasi errore di parsing/struttura -> fallback interno
            pass
    # Fallback: catalogo interno (realistico ma più piccolo).

    variables = ["CPU", "Motherboard", "RAM", "GPU", "PSU", "Case", "Storage"]

    # --- CPU ---
    # (price, name, socket, score)
    # NOTE: score = proxy prestazioni (0..100). Non è una misura assoluta.
    cpus = {
        # entry level
        "cpu_intel_i3_14100f": (130, "Intel Core i3-14100F", "lga1700", 35),
        "cpu_amd_r5_5500": (95, "AMD Ryzen 5 5500", "am4", 32),
        # mid
        "cpu_intel_i5_14400f": (185, "Intel Core i5-14400F", "lga1700", 55),
        "cpu_amd_r5_7600": (220, "AMD Ryzen 5 7600", "am5", 60),
        # upper-mid
        "cpu_intel_i5_14600k": (320, "Intel Core i5-14600K", "lga1700", 72),
        "cpu_amd_r7_7700": (295, "AMD Ryzen 7 7700", "am5", 70),
        # high
        "cpu_intel_i7_14700k": (430, "Intel Core i7-14700K", "lga1700", 82),
        "cpu_amd_r7_7800x3d": (390, "AMD Ryzen 7 7800X3D", "am5", 88),
        # enthusiast
        "cpu_intel_i9_14900k": (590, "Intel Core i9-14900K", "lga1700", 95),
        "cpu_amd_r9_7950x": (560, "AMD Ryzen 9 7950X", "am5", 93),
    }

    # --- Motherboard ---
    # (price, name, socket, ram)
    mbs = {
        # AM4 (budget platform)
        "mb_am4_b550_ddr4_matx": (105, "MSI B550 mATX (DDR4) AM4", "am4", "ddr4"),
        "mb_am4_b550_ddr4_atx": (125, "ASUS B550 ATX (DDR4) AM4", "am4", "ddr4"),

        # LGA1700
        "mb_lga1700_b760_ddr4_atx": (135, "Gigabyte B760 ATX (DDR4) LGA1700", "lga1700", "ddr4"),
        "mb_lga1700_b760_ddr5_atx": (190, "MSI B760 ATX (DDR5) LGA1700", "lga1700", "ddr5"),
        "mb_lga1700_z790_ddr5_atx": (260, "ASUS Z790 ATX (DDR5) LGA1700", "lga1700", "ddr5"),

        # AM5
        "mb_am5_b650_ddr5_matx": (180, "ASRock B650 mATX (DDR5) AM5", "am5", "ddr5"),
        "mb_am5_b650_ddr5_atx": (210, "MSI B650 ATX (DDR5) AM5", "am5", "ddr5"),
        "mb_am5_x670_ddr5_atx": (290, "ASUS X670 ATX (DDR5) AM5", "am5", "ddr5"),
    }

    # --- RAM ---
    # (price, name, type)
    # (price, name, type, gb)
    rams = {
        "ram_ddr4_16gb_3200": (45, "16GB (2x8) DDR4-3200", "ddr4", 16),
        "ram_ddr4_32gb_3600": (75, "32GB (2x16) DDR4-3600", "ddr4", 32),
        "ram_ddr4_64gb_3600": (145, "64GB (2x32) DDR4-3600", "ddr4", 64),
        "ram_ddr5_32gb_6000": (110, "32GB (2x16) DDR5-6000", "ddr5", 32),
        "ram_ddr5_64gb_6000": (190, "64GB (2x32) DDR5-6000", "ddr5", 64),
    }

    # --- GPU ---
    # (price, name, psu_min)
    # (price, name, psu_min, score)
    gpus = {
        # entry
        "gpu_rx_6600": (210, "AMD Radeon RX 6600 8GB", 500, 35),
        "gpu_rtx_4060": (290, "NVIDIA GeForce RTX 4060 8GB", 550, 42),
        # mid
        "gpu_rx_7600_xt": (330, "AMD Radeon RX 7600 XT 16GB", 550, 48),
        "gpu_rtx_4060_ti": (430, "NVIDIA GeForce RTX 4060 Ti 16GB", 650, 55),
        # upper-mid
        "gpu_rx_7700_xt": (480, "AMD Radeon RX 7700 XT 12GB", 650, 62),
        "gpu_rtx_4070_super": (640, "NVIDIA GeForce RTX 4070 SUPER 12GB", 650, 75),
        # high
        "gpu_rx_7900_xt": (790, "AMD Radeon RX 7900 XT 20GB", 750, 83),
        "gpu_rx_7900_xtx": (980, "AMD Radeon RX 7900 XTX 24GB", 850, 90),
        "gpu_rtx_4080_super": (1100, "NVIDIA GeForce RTX 4080 SUPER 16GB", 750, 92),
    }

    # --- PSU ---
    # (price, name, watt)
    psus = {
        "psu_550w_bronze": (70, "550W 80+ Bronze", 550),
        "psu_650w_gold": (95, "650W 80+ Gold", 650),
        "psu_750w_gold": (115, "750W 80+ Gold", 750),
        "psu_850w_gold": (145, "850W 80+ Gold", 850),
        "psu_1000w_gold": (190, "1000W 80+ Gold", 1000),
    }

    # --- Case --- (no compact)
    cases = {
        "case_atx_mid_tower": (95, "ATX Mid Tower (airflow)", 0),
        "case_atx_full_tower": (140, "ATX Full Tower", 0),
    }

    # --- Storage ---
    # (price, name, gb)
    stor = {
        "storage_sata_1tb": (55, "SATA SSD 1TB", 1000),
        "storage_nvme_1tb": (80, "NVMe SSD 1TB PCIe 4.0", 1000),
        "storage_nvme_2tb": (135, "NVMe SSD 2TB PCIe 4.0", 2000),
        "storage_nvme_4tb": (260, "NVMe SSD 4TB PCIe 4.0", 4000),
    }

    domains: Dict[str, List[str]] = {
        "CPU": list(cpus.keys()),
        "Motherboard": list(mbs.keys()),
        "RAM": list(rams.keys()),
        "GPU": list(gpus.keys()),
        "PSU": list(psus.keys()),
        "Case": list(cases.keys()),
        "Storage": list(stor.keys()),
    }

    prices: Dict[str, int] = {}
    display: Dict[str, str] = {}

    cpu_socket: Dict[str, str] = {}
    mb_socket: Dict[str, str] = {}
    mb_ram: Dict[str, str] = {}
    ram_type: Dict[str, str] = {}
    ram_gb: Dict[str, int] = {}
    gpu_psu_min: Dict[str, int] = {}
    gpu_score: Dict[str, int] = {}
    psu_watt: Dict[str, int] = {}

    cpu_score: Dict[str, int] = {}
    storage_gb: Dict[str, int] = {}

    for k, (p, n, sock, sc) in cpus.items():
        prices[k] = int(p)
        display[k] = n
        cpu_socket[k] = sock
        cpu_score[k] = int(sc)

    for k, (p, n, sock, rt) in mbs.items():
        prices[k] = int(p)
        display[k] = n
        mb_socket[k] = sock
        mb_ram[k] = rt

    for k, (p, n, rt, gb) in rams.items():
        prices[k] = int(p)
        display[k] = n
        ram_type[k] = rt
        ram_gb[k] = int(gb)

    for k, (p, n, psu_min, sc) in gpus.items():
        prices[k] = int(p)
        display[k] = n
        gpu_psu_min[k] = int(psu_min)
        gpu_score[k] = int(sc)

    for k, (p, n, w) in psus.items():
        prices[k] = int(p)
        display[k] = n
        psu_watt[k] = int(w)

    for k, (p, n, _) in cases.items():
        prices[k] = int(p)
        display[k] = n

    for k, (p, n, gb) in stor.items():
        prices[k] = int(p)
        display[k] = n
        storage_gb[k] = int(gb)

    return Catalog(
        variables=variables,
        domains=domains,
        prices=prices,
        display_name=display,
        cpu_socket=cpu_socket,
        mb_socket=mb_socket,
        mb_ram=mb_ram,
        ram_type=ram_type,
        gpu_psu_min=gpu_psu_min,
        psu_watt=psu_watt,
        cpu_score=cpu_score,
        gpu_score=gpu_score,
        ram_gb=ram_gb,
        storage_gb=storage_gb,
    )


def build_default_kg(seed: int = 0) -> Graph:
    """Crea la KB/KG con le regole di compatibilità.

    Implementazione: ogni regola è una relazione tra *assegnazioni reificate*
    (Assignment), in modo che la KB resti un grafo RDF semplice.
    """

    cat = build_default_catalog(seed=seed)

    g = Graph()
    g.bind("nscc", NSCC)

    # predicati
    incompatible_with = _u("incompatibleWith")
    requires = _u("requires")
    has_value = _u("hasValue")

    # variabili (individui)
    var_uris = {v: _u(v) for v in cat.variables}
    for v in cat.variables:
        g.add((var_uris[v], RDF.type, _u("Variable")))

    # valori (individui)
    for v in cat.variables:
        for val in cat.domains[v]:
            g.add((_u(val), RDF.type, _u("Value")))
            g.add((var_uris[v], has_value, _u(val)))

    def assign_node(var: str, val: str) -> URIRef:
        return _u(f"Assign_{var}_{val}")

    # reificazione base per ogni (var,val)
    for v in cat.variables:
        for val in cat.domains[v]:
            a = assign_node(v, val)
            g.add((a, RDF.type, _u("Assignment")))
            g.add((a, _u("var"), var_uris[v]))
            g.add((a, _u("val"), _u(val)))

    def ensure_assignment(v: str, val: str) -> None:
        """Assicura che il nodo di assegnazione Assign_{v}_{val} abbia var/val reificati.
        
        Serve per robustezza: alcune regole possono riferirsi a valori non presenti nel catalogo
        (es. hard-coded) oppure a cataloghi importati dal web con nomi diversi. In tal caso,
        creiamo comunque il nodo (così extract_constraints non fallisce) e lasciamo che il CSP
        ignori quei valori se non compaiono nei domini.
        """
        a = assign_node(v, val)
        # se esiste già var/val, niente da fare
        if any(True for _ in g.objects(a, _u("var"))) and any(True for _ in g.objects(a, _u("val"))):
            return
        # variabile e valore
        if v not in var_uris:
            var_uris[v] = _u(v)
            g.add((var_uris[v], RDF.type, _u("Variable")))
        g.add((a, RDF.type, _u("Assignment")))
        g.add((a, _u("var"), var_uris[v]))
        g.add((a, _u("val"), _u(val)))

    def add_incompat(v1: str, val1: str, v2: str, val2: str) -> None:
                ensure_assignment(v1, val1)
                ensure_assignment(v2, val2)
                g.add((assign_node(v1, val1), incompatible_with, assign_node(v2, val2)))

    def add_requires(v1: str, val1: str, v2: str, val2: str) -> None:
                ensure_assignment(v1, val1)
                ensure_assignment(v2, val2)
                g.add((assign_node(v1, val1), requires, assign_node(v2, val2)))

    # --- Regole 1) CPU socket -> MB compatibili (OR su valori Motherboard) ---
    for cpu, sock in cat.cpu_socket.items():
        for mb, mb_sock in cat.mb_socket.items():
            if mb_sock == sock:
                add_requires("CPU", cpu, "Motherboard", mb)

    # --- Regole 2) Motherboard ram-type -> RAM compatibili ---
    for mb, rt in cat.mb_ram.items():
        for ram, ram_rt in cat.ram_type.items():
            if ram_rt == rt:
                add_requires("Motherboard", mb, "RAM", ram)

    # --- Regole 3) GPU richiede PSU adeguato ---
    for gpu, minw in cat.gpu_psu_min.items():
        for psu, w in cat.psu_watt.items():
            if w >= minw:
                add_requires("GPU", gpu, "PSU", psu)
            else:
                # opzionale: esplicitiamo anche l'incompatibilità per pruning forte
                add_incompat("GPU", gpu, "PSU", psu)

    # --- Regole 4) (semplice) CPU very high-end non con PSU basse ---
    for cpu in ["cpu_intel_i9_14900k", "cpu_amd_r9_7950x"]:
        for psu in ["psu_550w_bronze", "psu_650w_gold"]:
            add_incompat("CPU", cpu, "PSU", psu)

    return g


def save_kg(g: Graph, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(path), format="turtle")


def load_kg(path: str | Path) -> Graph:
    path = Path(path)
    g = Graph()
    g.parse(str(path), format="turtle")
    return g


def extract_constraints(g: Graph) -> ConstraintSpec:
    """Estrae vincoli (incompatibilita' e implicazioni) dalla KB."""

    incompatible: List[Tuple[Tuple[str, str], Tuple[str, str]]] = []
    requires_list: List[Tuple[Tuple[str, str], Tuple[str, str]]] = []

    pred_incompat = _u("incompatibleWith")
    pred_requires = _u("requires")
    pred_var = _u("var")
    pred_val = _u("val")

    def decode_assignment(a: URIRef) -> Tuple[str, str]:
                # forma standard: proprietà var/val
                var_objs = list(g.objects(a, pred_var))
                val_objs = list(g.objects(a, pred_val))
                if var_objs and val_objs:
                    var_uri = var_objs[0]
                    val_uri = val_objs[0]
                    var = str(var_uri).split("/")[-1]
                    val = str(val_uri).split("/")[-1]
                    return var, val
                # fallback robusto: prova a decodificare dall'IRI Assign_<Var>_<Val>
                s = str(a)
                tail = s.split("/")[-1]
                if tail.startswith("Assign_"):
                    rest = tail[len("Assign_"):]
                    # split solo sul primo '_' per preservare eventuali '_' nel valore
                    parts = rest.split("_", 1)
                    if len(parts) == 2:
                        return parts[0], parts[1]
                raise ValueError(f"Assignment malformato (manca var/val): {a}")

    for a, _, b in g.triples((None, pred_incompat, None)):
                try:
                    incompatible.append((decode_assignment(a), decode_assignment(b)))
                except Exception:
                    # ignora triple malformate (tipicamente da regole riferite a valori non nel catalogo)
                    continue

    for a, _, b in g.triples((None, pred_requires, None)):
                try:
                    requires_list.append((decode_assignment(a), decode_assignment(b)))
                except Exception:
                    continue

    return ConstraintSpec(incompatible=incompatible, requires=requires_list)