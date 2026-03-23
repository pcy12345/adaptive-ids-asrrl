"""
Real Dataset Loader for ASRRL IDS Framework

Downloads and preprocesses real IDS benchmark datasets:
  - UNSW-NB15
  - CSE-CIC-IDS-2018
  - CIC-IDS2017

Each dataset is standardised to a common schema:
  Features: flow_duration, pkt_rate, byte_rate, entropy, port_cat, size_cat, protocol
  Labels:   0=benign, 1=attack
  Multi-class: attack_cat column (string) when available

Falls back to enhanced synthetic data if download/loading fails.

Usage:
    from real_data import load_dataset
    df = load_dataset("UNSW")          # returns DataFrame
    df = load_dataset("CSE")
    df = load_dataset("CIC2017")
"""

import os
import io
import zipfile
import warnings
import hashlib

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
#  DOWNLOAD HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _download(url, dest_path, chunk_size=8192):
    """Download file with progress indication."""
    if os.path.exists(dest_path):
        return True
    print(f"    Downloading {os.path.basename(dest_path)} ...")
    try:
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 / total
                    print(f"\r    {pct:5.1f}%  ({downloaded // 1024}KB / {total // 1024}KB)",
                          end="", flush=True)
        print()
        return True
    except Exception as e:
        print(f"    Download failed: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


# ═══════════════════════════════════════════════════════════════════════════
#  UNSW-NB15
# ═══════════════════════════════════════════════════════════════════════════

# Official processed CSV from UNSW research (training + testing sets)
UNSW_URLS = [
    "https://research.unsw.edu.au/sites/default/files/documents/UNSW_NB15_training-set.csv",
    "https://research.unsw.edu.au/sites/default/files/documents/UNSW_NB15_testing-set.csv",
]

UNSW_FEATURE_MAP = {
    "dur": "flow_duration",
    "rate": "pkt_rate",
    "sbytes": "byte_rate",
    "ct_srv_src": "entropy",   # service connection count as proxy for entropy
}

UNSW_ATTACK_CATS = {
    "Normal": "benign",
    "Exploits": "exploits",
    "Fuzzers": "fuzzers",
    "DoS": "dos",
    "Reconnaissance": "recon",
    "Analysis": "analysis",
    "Backdoor": "backdoor",
    "Shellcode": "shellcode",
    "Worms": "worms",
    "Generic": "generic",
}


def _load_unsw_local():
    """Try loading UNSW-NB15 from local data/ directory."""
    # Check for various common filenames
    candidates = [
        os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv"),
        os.path.join(DATA_DIR, "UNSW-NB15_training-set.csv"),
        os.path.join(DATA_DIR, "unsw_nb15.csv"),
        os.path.join(DATA_DIR, "UNSW_NB15.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_csv(path, low_memory=False)

    # Try downloading
    train_path = os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv")
    test_path = os.path.join(DATA_DIR, "UNSW_NB15_testing-set.csv")

    if _download(UNSW_URLS[0], train_path) and _download(UNSW_URLS[1], test_path):
        train = pd.read_csv(train_path, low_memory=False)
        test = pd.read_csv(test_path, low_memory=False)
        return pd.concat([train, test], ignore_index=True)

    return None


def _preprocess_unsw(raw_df):
    """Convert UNSW-NB15 to standardised schema."""
    df = raw_df.copy()

    # Standardise column names
    df.columns = df.columns.str.strip().str.lower()

    # Map features
    result = pd.DataFrame()

    # flow_duration: 'dur' column (seconds)
    if "dur" in df.columns:
        result["flow_duration"] = pd.to_numeric(df["dur"], errors="coerce").fillna(0) * 1000
    else:
        result["flow_duration"] = 0

    # pkt_rate: 'rate' or compute from spkts+dpkts / dur
    if "rate" in df.columns:
        result["pkt_rate"] = pd.to_numeric(df["rate"], errors="coerce").fillna(0)
    elif "spkts" in df.columns and "dpkts" in df.columns:
        dur = pd.to_numeric(df.get("dur", 1), errors="coerce").fillna(1).clip(lower=0.001)
        result["pkt_rate"] = (pd.to_numeric(df["spkts"], errors="coerce").fillna(0) +
                              pd.to_numeric(df["dpkts"], errors="coerce").fillna(0)) / dur

    # byte_rate: sbytes + dbytes
    if "sbytes" in df.columns:
        result["byte_rate"] = pd.to_numeric(df["sbytes"], errors="coerce").fillna(0)
        if "dbytes" in df.columns:
            result["byte_rate"] += pd.to_numeric(df["dbytes"], errors="coerce").fillna(0)
    else:
        result["byte_rate"] = 0

    # entropy: use ct_srv_src (connection count as proxy) or sttl variation
    for col in ["ct_srv_src", "sttl", "sjit"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
            # Normalise to [0, 1] range
            vmin, vmax = vals.min(), vals.max()
            if vmax > vmin:
                result["entropy"] = (vals - vmin) / (vmax - vmin)
            else:
                result["entropy"] = 0.5
            break
    else:
        result["entropy"] = np.random.uniform(0, 1, len(df))

    # port_cat: bin destination port
    if "dsport" in df.columns:
        ports = pd.to_numeric(df["dsport"], errors="coerce").fillna(0).astype(int)
        result["port_cat"] = pd.cut(ports, bins=[-1, 80, 443, 1024, 5000, 10000, 65536],
                                    labels=[0, 1, 2, 3, 4, 5]).astype(float).fillna(0).astype(int)
    elif "sport" in df.columns:
        ports = pd.to_numeric(df["sport"], errors="coerce").fillna(0).astype(int)
        result["port_cat"] = pd.cut(ports, bins=[-1, 80, 443, 1024, 5000, 10000, 65536],
                                    labels=[0, 1, 2, 3, 4, 5]).astype(float).fillna(0).astype(int)
    else:
        result["port_cat"] = np.random.randint(0, 6, len(df))

    # size_cat: bin total bytes
    total_bytes = result["byte_rate"].values
    result["size_cat"] = pd.cut(total_bytes,
                                bins=[-1, 100, 1000, 10000, float("inf")],
                                labels=[0, 1, 2, 3]).astype(float).fillna(0).astype(int)

    # protocol
    if "proto" in df.columns:
        proto_map = {"tcp": 0, "udp": 1, "icmp": 2}
        result["protocol"] = df["proto"].astype(str).str.strip().str.lower().map(proto_map).fillna(0).astype(int)
    else:
        result["protocol"] = np.random.randint(0, 3, len(df))

    # Binary label
    if "label" in df.columns:
        result["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    elif "attack_cat" in df.columns:
        result["label"] = (df["attack_cat"].astype(str).str.strip().str.lower() != "normal").astype(int)

    # Multi-class attack category
    if "attack_cat" in df.columns:
        result["attack_cat"] = df["attack_cat"].astype(str).str.strip()
    else:
        result["attack_cat"] = np.where(result["label"] == 0, "Normal", "Attack")

    # Drop NaN rows
    result = result.dropna().reset_index(drop=True)

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  CSE-CIC-IDS-2018
# ═══════════════════════════════════════════════════════════════════════════

CSE_LOCAL_PATTERNS = [
    "CSE-CIC-IDS2018*.csv",
    "cse_cic_ids2018*.csv",
    "Friday-02-03-2018*.csv",
    "ids2018*.csv",
]


def _load_cse_local():
    """Try loading CSE-CIC-IDS-2018 from local data/ directory."""
    import glob
    for pattern in CSE_LOCAL_PATTERNS:
        matches = glob.glob(os.path.join(DATA_DIR, pattern))
        if matches:
            dfs = [pd.read_csv(f, low_memory=False) for f in sorted(matches)]
            return pd.concat(dfs, ignore_index=True)

    # Also check for any CSV with 'cic' or 'ids2018' in name
    for f in os.listdir(DATA_DIR):
        if f.lower().endswith(".csv") and ("cic" in f.lower() or "ids2018" in f.lower()):
            return pd.read_csv(os.path.join(DATA_DIR, f), low_memory=False)

    return None


def _preprocess_cic(raw_df, dataset_name="CSE"):
    """Convert CIC-IDS-2017/2018 to standardised schema.
    Both datasets share the same CICFlowMeter feature format."""
    df = raw_df.copy()
    df.columns = df.columns.str.strip()

    result = pd.DataFrame()

    # flow_duration (microseconds → milliseconds)
    for col in ["Flow Duration", "flow_duration", "Flow duration"]:
        if col in df.columns:
            result["flow_duration"] = pd.to_numeric(df[col], errors="coerce").fillna(0) / 1000
            break
    else:
        result["flow_duration"] = 0

    # pkt_rate: Flow Packets/s or total packets / duration
    for col in ["Flow Packets/s", "flow_packets_s", "Flow Pkts/s"]:
        if col in df.columns:
            result["pkt_rate"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            break
    else:
        # Compute from fwd + bwd packets
        fwd = bwd = 0
        for c in df.columns:
            if "total fwd" in c.lower() and "packet" in c.lower():
                fwd = pd.to_numeric(df[c], errors="coerce").fillna(0)
            if "total bwd" in c.lower() and "packet" in c.lower() or "total backward" in c.lower():
                bwd = pd.to_numeric(df[c], errors="coerce").fillna(0)
        dur = result["flow_duration"].clip(lower=0.001)
        result["pkt_rate"] = (fwd + bwd) / dur * 1000

    # byte_rate: Flow Bytes/s
    for col in ["Flow Bytes/s", "flow_bytes_s", "Flow Byts/s"]:
        if col in df.columns:
            result["byte_rate"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            break
    else:
        result["byte_rate"] = 0

    # entropy: use Fwd IAT Std or packet length variance as proxy
    for col in ["Fwd IAT Std", "fwd_iat_std", "Flow IAT Std"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
            vmin, vmax = vals.quantile(0.01), vals.quantile(0.99)
            if vmax > vmin:
                result["entropy"] = ((vals - vmin) / (vmax - vmin)).clip(0, 1)
            else:
                result["entropy"] = 0.5
            break
    else:
        result["entropy"] = np.random.uniform(0, 1, len(df))

    # port_cat
    for col in ["Dst Port", "Destination Port", "dst_port"]:
        if col in df.columns:
            ports = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            result["port_cat"] = pd.cut(ports, bins=[-1, 80, 443, 1024, 5000, 10000, 65536],
                                        labels=[0, 1, 2, 3, 4, 5]).astype(float).fillna(0).astype(int)
            break
    else:
        result["port_cat"] = np.random.randint(0, 6, len(df))

    # size_cat
    total_bytes = result["byte_rate"].values * (result["flow_duration"].values / 1000).clip(min=0.001)
    result["size_cat"] = pd.cut(total_bytes,
                                bins=[-1, 100, 1000, 10000, float("inf")],
                                labels=[0, 1, 2, 3]).astype(float).fillna(0).astype(int)

    # protocol
    for col in ["Protocol", "protocol"]:
        if col in df.columns:
            proto = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            # IP protocol numbers: 6=TCP, 17=UDP, 1=ICMP
            result["protocol"] = proto.map({6: 0, 17: 1, 1: 2}).fillna(0).astype(int)
            break
    else:
        result["protocol"] = np.random.randint(0, 3, len(df))

    # Label
    for col in ["Label", "label"]:
        if col in df.columns:
            labels = df[col].astype(str).str.strip()
            # Binary: BENIGN=0, everything else=1
            result["label"] = (~labels.str.upper().isin(["BENIGN", "0", "NORMAL"])).astype(int)
            # Multi-class
            result["attack_cat"] = labels.replace({"BENIGN": "Normal", "0": "Normal"})
            break
    else:
        result["label"] = 0
        result["attack_cat"] = "Normal"

    # Clean infinities and NaN
    for col in result.columns:
        if result[col].dtype in [np.float64, np.float32]:
            result[col] = result[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    result = result.dropna().reset_index(drop=True)
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  CIC-IDS2017
# ═══════════════════════════════════════════════════════════════════════════

CIC2017_LOCAL_PATTERNS = [
    "CIC-IDS2017*.csv",
    "cic_ids2017*.csv",
    "Friday-WorkingHours*.csv",
    "ids2017*.csv",
    "MachineLearningCVE/*.csv",
]


def _load_cic2017_local():
    """Try loading CIC-IDS2017 from local data/ directory."""
    import glob
    for pattern in CIC2017_LOCAL_PATTERNS:
        matches = glob.glob(os.path.join(DATA_DIR, pattern))
        if matches:
            dfs = [pd.read_csv(f, low_memory=False, encoding="latin-1") for f in sorted(matches)]
            return pd.concat(dfs, ignore_index=True)

    for f in os.listdir(DATA_DIR):
        if f.lower().endswith(".csv") and ("ids2017" in f.lower() or "cicids2017" in f.lower()):
            return pd.read_csv(os.path.join(DATA_DIR, f), low_memory=False, encoding="latin-1")

    return None


# ═══════════════════════════════════════════════════════════════════════════
#  ENHANCED SYNTHETIC DATA (fallback)
# ═══════════════════════════════════════════════════════════════════════════

def _generate_enhanced_synthetic(n, ds, seed=42, multi_class=False):
    """Generate synthetic data with optional multi-class attack types.
    More realistic than the basic version — includes temporal ordering
    and multiple attack categories."""
    rng = np.random.default_rng(seed)

    if ds == "CSE":
        atk_ratio = 0.15
        attack_types = {"DoS": 0.30, "DDoS": 0.25, "BruteForce": 0.20,
                        "Infiltration": 0.15, "BotNet": 0.10}
        dur_n, dur_a = (3000, 800), (1200, 500)
        pr_n, pr_a = (1200, 600), (2400, 700)
        br_n, br_a = (1e6, 7e5), (2.8e6, 1.2e6)
        en_n, en_a = (0.45, 0.20), (0.95, 0.25)
    elif ds == "CIC2017":
        atk_ratio = 0.20
        attack_types = {"DoS": 0.25, "PortScan": 0.20, "DDoS": 0.20,
                        "BruteForce": 0.15, "WebAttack": 0.10, "Infiltration": 0.10}
        dur_n, dur_a = (2000, 700), (900, 500)
        pr_n, pr_a = (900, 400), (2000, 600)
        br_n, br_a = (7e5, 3e5), (1.8e6, 7e5)
        en_n, en_a = (0.42, 0.18), (0.85, 0.22)
    else:  # UNSW
        atk_ratio = 0.30
        attack_types = {"Generic": 0.25, "Exploits": 0.20, "Fuzzers": 0.15,
                        "DoS": 0.15, "Recon": 0.10, "Analysis": 0.08,
                        "Backdoor": 0.04, "Shellcode": 0.02, "Worms": 0.01}
        dur_n, dur_a = (1500, 400), (700, 300)
        pr_n, pr_a = (600, 250), (1400, 350)
        br_n, br_a = (4e5, 1.5e5), (9e5, 2.5e5)
        en_n, en_a = (0.40, 0.15), (0.75, 0.18)

    labels = rng.choice([0, 1], p=[1 - atk_ratio, atk_ratio], size=n)
    m0, m1 = labels == 0, labels == 1

    def _col(norm, atk):
        out = np.empty(n)
        out[m0] = rng.normal(norm[0], norm[1], m0.sum()).clip(1)
        out[m1] = rng.normal(atk[0], atk[1], m1.sum()).clip(1)
        return out

    df = pd.DataFrame({
        "flow_duration": _col(dur_n, dur_a),
        "pkt_rate": _col(pr_n, pr_a),
        "byte_rate": _col(br_n, br_a),
        "entropy": _col(en_n, en_a),
        "port_cat": rng.integers(0, 6, n),
        "size_cat": rng.integers(0, 4, n),
        "protocol": rng.integers(0, 3, n),
        "label": labels,
    })

    # Multi-class attack categories
    atk_names = list(attack_types.keys())
    atk_probs = list(attack_types.values())
    # Normalise probabilities
    atk_probs = np.array(atk_probs) / sum(atk_probs)

    attack_cats = np.array(["Normal"] * n, dtype=object)
    attack_indices = np.where(labels == 1)[0]
    if len(attack_indices) > 0:
        assigned = rng.choice(atk_names, p=atk_probs, size=len(attack_indices))
        attack_cats[attack_indices] = assigned

        # Make different attack types have slightly different feature signatures
        for i, atype in enumerate(atk_names):
            mask = attack_cats == atype
            if mask.sum() > 0:
                # Shift features slightly per attack type
                shift = (i + 1) * 0.05
                df.loc[mask, "entropy"] += rng.normal(shift, 0.05, mask.sum())
                df.loc[mask, "pkt_rate"] *= (1 + shift * rng.normal(0, 0.3, mask.sum()))

    df["attack_cat"] = attack_cats

    # Add temporal ordering (simulate time-based flow capture)
    df["timestamp"] = np.sort(rng.uniform(0, 86400, n))  # seconds in a day

    return df


# ═══════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

def load_dataset(ds, n=50000, seed=42):
    """Load a dataset (real if available, synthetic fallback).

    Parameters
    ----------
    ds : str
        "UNSW", "CSE", or "CIC2017"
    n : int
        Desired number of samples (real datasets are subsampled if larger)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Columns: flow_duration, pkt_rate, byte_rate, entropy, port_cat,
                 size_cat, protocol, label, attack_cat
    str
        "real" or "synthetic" indicating data source
    """
    real_df = None

    print(f"  Loading {ds} dataset ...")

    if ds == "UNSW":
        raw = _load_unsw_local()
        if raw is not None:
            real_df = _preprocess_unsw(raw)

    elif ds == "CSE":
        raw = _load_cse_local()
        if raw is not None:
            real_df = _preprocess_cic(raw, "CSE")

    elif ds == "CIC2017":
        raw = _load_cic2017_local()
        if raw is not None:
            real_df = _preprocess_cic(raw, "CIC2017")

    if real_df is not None and len(real_df) > 100:
        print(f"    Loaded REAL {ds} data: {len(real_df)} records")
        # Subsample if needed
        if len(real_df) > n:
            real_df = real_df.sample(n=n, random_state=seed).reset_index(drop=True)
            print(f"    Subsampled to {n} records")
        return real_df, "real"

    print(f"    Real data not found. Using enhanced synthetic data.")
    print(f"    To use real data, place CSV files in: {DATA_DIR}/")
    print(f"    Download links:")
    if ds == "UNSW":
        print(f"      https://research.unsw.edu.au/projects/unsw-nb15-dataset")
    elif ds == "CSE":
        print(f"      https://www.unb.ca/cic/datasets/ids-2018.html")
    else:
        print(f"      https://www.unb.ca/cic/datasets/ids-2017.html")

    return _generate_enhanced_synthetic(n, ds, seed=seed, multi_class=True), "synthetic"


def load_all_datasets(n=50000, seed=42):
    """Load all three datasets, return dict."""
    results = {}
    for ds in ["CSE", "UNSW", "CIC2017"]:
        df, source = load_dataset(ds, n=n, seed=seed)
        results[ds] = {"df": df, "source": source}
    return results


if __name__ == "__main__":
    # Quick test
    for ds in ["UNSW", "CSE", "CIC2017"]:
        df, source = load_dataset(ds, n=5000)
        print(f"  {ds}: {len(df)} rows, {source}, "
              f"attack_rate={df['label'].mean():.2%}, "
              f"attack_types={df['attack_cat'].nunique()}")
        print(f"    Columns: {list(df.columns)}")
        print()
