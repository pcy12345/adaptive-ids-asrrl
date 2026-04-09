#!/usr/bin/env python3
"""
Upload IDS Benchmark Datasets to Hugging Face

Downloads and uploads three IDS benchmark datasets to a Hugging Face account:
  1. UNSW-NB15
  2. CSE-CIC-IDS 2018
  3. CIC-IDS 2017

Usage:
    python upload_datasets_to_hf.py --token YOUR_HF_TOKEN --username YOUR_HF_USERNAME

    # Or set environment variable:
    export HF_TOKEN=your_token
    python upload_datasets_to_hf.py --username pcy12345BSU

    # Upload only specific datasets:
    python upload_datasets_to_hf.py --token YOUR_TOKEN --username pcy12345BSU --datasets UNSW CSE CIC2017

Requirements:
    pip install huggingface_hub requests pandas
"""

import os
import sys
import argparse
import time
import zipfile
import io
import glob

import requests
import pandas as pd
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder

# ============================================================================
#  Configuration
# ============================================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

DATASET_CONFIGS = {
    "UNSW": {
        "repo_name": "UNSW-NB15",
        "description": "UNSW-NB15 Network Intrusion Detection Dataset",
        "sources": {
            "github_csv": "https://raw.githubusercontent.com/abhinav-bhardwaj/IoT-Network-Intrusion-Detection-System-UNSW-NB15/master/datasets/UNSW_NB15.csv",
            "github_features": "https://raw.githubusercontent.com/abhinav-bhardwaj/IoT-Network-Intrusion-Detection-System-UNSW-NB15/master/datasets/UNSW_NB15_features.csv",
            "official_train": "https://research.unsw.edu.au/sites/default/files/documents/UNSW_NB15_training-set.csv",
            "official_test": "https://research.unsw.edu.au/sites/default/files/documents/UNSW_NB15_testing-set.csv",
        },
        "local_dir": os.path.join(DATA_DIR, "unsw_nb15"),
    },
    "CSE": {
        "repo_name": "CSE-CIC-IDS-2018",
        "description": "CSE-CIC-IDS 2018 Network Intrusion Detection Dataset",
        "s3_bucket": "cse-cic-ids2018",
        "s3_prefix": "Processed Traffic Data for ML Algorithms/",
        "local_dir": os.path.join(DATA_DIR, "cse_cic_ids2018"),
    },
    "CIC2017": {
        "repo_name": "CIC-IDS-2017",
        "description": "CIC-IDS 2017 Network Intrusion Detection Dataset",
        "sources": {
            "official": "https://iscxdownloads.cs.unb.ca/iscxdownloads/CIC-IDS-2017/MachineLearningCSV.zip",
        },
        "individual_csvs": [
            "https://raw.githubusercontent.com/Mamcose/CIC-IDS-2017/main/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
            "https://raw.githubusercontent.com/Mamcose/CIC-IDS-2017/main/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
            "https://raw.githubusercontent.com/Mamcose/CIC-IDS-2017/main/MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv",
            "https://raw.githubusercontent.com/Mamcose/CIC-IDS-2017/main/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv",
            "https://raw.githubusercontent.com/Mamcose/CIC-IDS-2017/main/MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
            "https://raw.githubusercontent.com/Mamcose/CIC-IDS-2017/main/MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
            "https://raw.githubusercontent.com/Mamcose/CIC-IDS-2017/main/MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv",
            "https://raw.githubusercontent.com/Mamcose/CIC-IDS-2017/main/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv",
        ],
        "local_dir": os.path.join(DATA_DIR, "cic_ids2017"),
    },
}


# ============================================================================
#  Download Helpers
# ============================================================================

def download_file(url, dest_path, chunk_size=8192, timeout=300):
    """Download a file with progress indication."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"  [SKIP] Already exists: {os.path.basename(dest_path)}")
        return True

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"  Downloading: {os.path.basename(dest_path)} ...")

    try:
        r = requests.get(url, stream=True, timeout=timeout)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0

        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 / total
                    mb_done = downloaded / 1024 / 1024
                    mb_total = total / 1024 / 1024
                    print(f"\r  {pct:5.1f}%  ({mb_done:.1f} MB / {mb_total:.1f} MB)", end="", flush=True)
        print()

        size_mb = os.path.getsize(dest_path) / 1024 / 1024
        print(f"  [OK] Downloaded: {size_mb:.1f} MB")
        return True

    except Exception as e:
        print(f"\n  [FAIL] Download failed: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


# ============================================================================
#  UNSW-NB15 Download
# ============================================================================

def download_unsw():
    """Download UNSW-NB15 dataset."""
    print("\n" + "=" * 60)
    print("  UNSW-NB15 Dataset")
    print("=" * 60)

    cfg = DATASET_CONFIGS["UNSW"]
    local_dir = cfg["local_dir"]
    os.makedirs(local_dir, exist_ok=True)

    # Check if already downloaded
    existing = glob.glob(os.path.join(local_dir, "*.csv"))
    if existing:
        print(f"  Found {len(existing)} CSV file(s) in {local_dir}")
        return True

    # Also check parent data dir
    parent_csvs = glob.glob(os.path.join(DATA_DIR, "UNSW*.csv")) + \
                  glob.glob(os.path.join(DATA_DIR, "unsw*.csv"))
    if parent_csvs:
        print(f"  Found existing files in data/: {[os.path.basename(f) for f in parent_csvs]}")
        # Copy to local_dir
        import shutil
        for f in parent_csvs:
            dest = os.path.join(local_dir, os.path.basename(f))
            if not os.path.exists(dest):
                shutil.copy2(f, dest)
        return True

    sources = cfg["sources"]
    success = False

    # Try official UNSW sources first
    train_path = os.path.join(local_dir, "UNSW_NB15_training-set.csv")
    test_path = os.path.join(local_dir, "UNSW_NB15_testing-set.csv")

    if download_file(sources["official_train"], train_path) and \
       download_file(sources["official_test"], test_path):
        success = True
    else:
        # Fallback to GitHub mirror
        print("  Trying GitHub mirror...")
        csv_path = os.path.join(local_dir, "UNSW_NB15.csv")
        if download_file(sources["github_csv"], csv_path):
            success = True

        # Also try features file
        feat_path = os.path.join(local_dir, "UNSW_NB15_features.csv")
        download_file(sources["github_features"], feat_path)

    if success:
        print("  [OK] UNSW-NB15 downloaded successfully")
    else:
        print("  [WARN] Could not download UNSW-NB15 automatically.")
        print("         Please download manually from:")
        print("         https://research.unsw.edu.au/projects/unsw-nb15-dataset")
        print(f"         Place CSV files in: {local_dir}/")

    return success


# ============================================================================
#  CSE-CIC-IDS 2018 Download (from AWS S3)
# ============================================================================

def download_cse_2018():
    """Download CSE-CIC-IDS 2018 processed CSV files from AWS S3."""
    print("\n" + "=" * 60)
    print("  CSE-CIC-IDS 2018 Dataset")
    print("=" * 60)

    cfg = DATASET_CONFIGS["CSE"]
    local_dir = cfg["local_dir"]
    os.makedirs(local_dir, exist_ok=True)

    # Check if already downloaded
    existing = glob.glob(os.path.join(local_dir, "*.csv"))
    if len(existing) >= 8:
        print(f"  Found {len(existing)} CSV file(s) - skipping download")
        return True

    bucket = cfg["s3_bucket"]
    prefix = cfg["s3_prefix"]

    # List S3 bucket
    import xml.etree.ElementTree as ET
    try:
        r = requests.get(f"https://{bucket}.s3.amazonaws.com/", timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"  [FAIL] Cannot access S3 bucket: {e}")
        print("  Please download manually from: https://www.unb.ca/cic/datasets/ids-2018.html")
        return False

    root = ET.fromstring(r.text)
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

    csv_files = []
    for item in root.findall(".//s3:Contents", ns):
        key = item.find("s3:Key", ns).text
        size = int(item.find("s3:Size", ns).text)
        if key.startswith(prefix) and key.endswith(".csv") and size > 0:
            csv_files.append((key, size))

    if not csv_files:
        print("  [FAIL] No CSV files found in S3 bucket")
        return False

    total_size = sum(s for _, s in csv_files)
    print(f"  Found {len(csv_files)} CSV files ({total_size / 1024 / 1024 / 1024:.1f} GB total)")

    success_count = 0
    for key, size in csv_files:
        filename = os.path.basename(key)
        dest = os.path.join(local_dir, filename)
        url = f"https://{bucket}.s3.amazonaws.com/{requests.utils.quote(key)}"

        if download_file(url, dest, timeout=600):
            success_count += 1
        else:
            print(f"  [WARN] Failed to download {filename}")

    print(f"\n  Downloaded {success_count}/{len(csv_files)} CSE-CIC-IDS 2018 files")
    return success_count > 0


# ============================================================================
#  CIC-IDS 2017 Download
# ============================================================================

def download_cic_2017():
    """Download CIC-IDS 2017 dataset."""
    print("\n" + "=" * 60)
    print("  CIC-IDS 2017 Dataset")
    print("=" * 60)

    cfg = DATASET_CONFIGS["CIC2017"]
    local_dir = cfg["local_dir"]
    os.makedirs(local_dir, exist_ok=True)

    # Check if already downloaded
    existing = glob.glob(os.path.join(local_dir, "*.csv"))
    if existing:
        print(f"  Found {len(existing)} CSV file(s) - skipping download")
        return True

    # Try official source (zip)
    url = cfg["sources"]["official"]
    zip_path = os.path.join(local_dir, "MachineLearningCSV.zip")

    if download_file(url, zip_path, timeout=600):
        try:
            print("  Extracting ZIP archive...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(local_dir)
            print("  [OK] CIC-IDS 2017 downloaded and extracted")

            # Move CSVs from subdirectories to local_dir
            for root_dir, dirs, files in os.walk(local_dir):
                for f in files:
                    if f.endswith(".csv") and root_dir != local_dir:
                        src = os.path.join(root_dir, f)
                        dst = os.path.join(local_dir, f)
                        if not os.path.exists(dst):
                            os.rename(src, dst)
            return True
        except Exception as e:
            print(f"  [FAIL] Extraction failed: {e}")

    # Fallback: download individual CSVs from GitHub mirror
    individual_csvs = cfg.get("individual_csvs", [])
    if individual_csvs:
        print("  Official source failed. Trying GitHub mirror (individual CSVs)...")
        success_count = 0
        for csv_url in individual_csvs:
            filename = csv_url.split("/")[-1]
            dest = os.path.join(local_dir, filename)
            if download_file(csv_url, dest, timeout=300):
                success_count += 1
            else:
                print(f"  [WARN] Failed: {filename}")
        print(f"\n  Downloaded {success_count}/{len(individual_csvs)} CIC-IDS 2017 files")
        if success_count > 0:
            return True

    print("  [WARN] Could not download CIC-IDS 2017 automatically.")
    print("         Please download the MachineLearningCSV.zip from:")
    print("         https://www.unb.ca/cic/datasets/ids-2017.html")
    print(f"         Extract CSV files to: {local_dir}/")
    return False


# ============================================================================
#  Dataset Card Templates
# ============================================================================

UNSW_CARD = """---
license: other
task_categories:
  - tabular-classification
tags:
  - intrusion-detection
  - network-security
  - cybersecurity
  - IDS
  - UNSW-NB15
  - anomaly-detection
pretty_name: UNSW-NB15 Network Intrusion Detection Dataset
size_categories:
  - 100K<n<1M
---

# UNSW-NB15 Network Intrusion Detection Dataset

## Description

The UNSW-NB15 dataset was created by the Cyber Range Lab of the Australian Centre
for Cyber Security (ACCS) at the University of New South Wales (UNSW). It contains
a hybrid of real modern normal activities and synthetic contemporary attack behaviours.

The raw network packets were captured using the IXIA PerfectStorm tool to generate
9 types of attacks: Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic,
Reconnaissance, Shellcode, and Worms.

## Dataset Details

- **Source**: University of New South Wales (UNSW), Australia
- **Year**: 2015
- **Features**: 49 features including flow-based and packet-based attributes
- **Attack Types**: 9 categories (Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms)
- **Total Records**: ~2.5 million
- **Training Set**: 175,341 records
- **Testing Set**: 82,332 records

## Citation

```bibtex
@article{moustafa2015unsw,
  title={UNSW-NB15: a comprehensive data set for network intrusion detection systems},
  author={Moustafa, Nour and Slay, Jill},
  journal={Military Communications and Information Systems Conference (MilCIS)},
  year={2015},
  publisher={IEEE}
}
```

## License

This dataset is provided for research and educational purposes.
Please cite the original paper when using this dataset.

## Original Source

- https://research.unsw.edu.au/projects/unsw-nb15-dataset
"""

CSE_CARD = """---
license: other
task_categories:
  - tabular-classification
tags:
  - intrusion-detection
  - network-security
  - cybersecurity
  - IDS
  - CSE-CIC-IDS-2018
  - anomaly-detection
  - CICFlowMeter
pretty_name: CSE-CIC-IDS 2018 Network Intrusion Detection Dataset
size_categories:
  - 10M<n<100M
---

# CSE-CIC-IDS 2018 Network Intrusion Detection Dataset

## Description

The CSE-CIC-IDS2018 dataset was developed by the Communications Security
Establishment (CSE) and the Canadian Institute for Cybersecurity (CIC).
It includes seven different attack scenarios: Brute-force, Heartbleed,
Botnet, DoS, DDoS, Web attacks, and infiltration of the network from inside.

Network traffic was captured using CICFlowMeter and processed into
bidirectional flow features.

## Dataset Details

- **Source**: CSE & Canadian Institute for Cybersecurity (CIC), University of New Brunswick
- **Year**: 2018
- **Features**: 80+ CICFlowMeter features
- **Attack Types**: 7 categories (Brute-force, Heartbleed, Botnet, DoS, DDoS, Web attacks, Infiltration)
- **Duration**: 10 days of network traffic capture (Feb 14 - Mar 2, 2018)
- **Infrastructure**: 50 machines for attack, 420 machines + 30 servers for victim network

## Files

Processed CSV files from CICFlowMeter for each capture day:
- Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv
- Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv
- Friday-16-02-2018_TrafficForML_CICFlowMeter.csv
- Tuesday-20-02-2018_TrafficForML_CICFlowMeter.csv (largest file)
- Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv
- Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv
- Friday-23-02-2018_TrafficForML_CICFlowMeter.csv
- Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv
- Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv
- Friday-02-03-2018_TrafficForML_CICFlowMeter.csv

## Citation

```bibtex
@misc{cse-cic-ids2018,
  title={A Realistic Cyber Defense Dataset (CSE-CIC-IDS2018)},
  author={Iman Sharafaldin and Arash Habibi Lashkari and Ali A. Ghorbani},
  year={2018},
  publisher={Canadian Institute for Cybersecurity},
  url={https://registry.opendata.aws/cse-cic-ids2018/}
}
```

## License

This dataset is provided for research and educational purposes.
Please cite the original authors when using this dataset.

## Original Source

- https://www.unb.ca/cic/datasets/ids-2018.html
- AWS S3: s3://cse-cic-ids2018/
"""

CIC2017_CARD = """---
license: other
task_categories:
  - tabular-classification
tags:
  - intrusion-detection
  - network-security
  - cybersecurity
  - IDS
  - CIC-IDS-2017
  - anomaly-detection
  - CICFlowMeter
pretty_name: CIC-IDS 2017 Network Intrusion Detection Dataset
size_categories:
  - 1M<n<10M
---

# CIC-IDS 2017 Network Intrusion Detection Dataset

## Description

The CIC-IDS2017 dataset was generated by the Canadian Institute for Cybersecurity
(CIC). It contains benign traffic and the most up-to-date (at the time) common
attacks resembling true real-world data (PCAPs).

The dataset was captured over 5 days (Monday to Friday) with different attack
scenarios each day.

## Dataset Details

- **Source**: Canadian Institute for Cybersecurity (CIC), University of New Brunswick
- **Year**: 2017
- **Features**: 80+ CICFlowMeter features
- **Attack Types**: Brute Force FTP/SSH, DoS (Hulk, GoldenEye, Slowloris, Slowhttptest), Heartbleed, Web Attack (XSS, SQL Injection, Brute Force), Infiltration, Botnet, DDoS, PortScan
- **Duration**: 5 days (July 3-7, 2017)

## Files

Processed CSV files (MachineLearningCVE format):
- Monday-WorkingHours.pcap_ISCX.csv (Benign only)
- Tuesday-WorkingHours.pcap_ISCX.csv (Brute Force)
- Wednesday-workingHours.pcap_ISCX.csv (DoS/Heartbleed)
- Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
- Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
- Friday-WorkingHours-Morning.pcap_ISCX.csv (Botnet)
- Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
- Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv

## Citation

```bibtex
@inproceedings{sharafaldin2018toward,
  title={Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization},
  author={Sharafaldin, Iman and Lashkari, Arash Habibi and Ghorbani, Ali A.},
  booktitle={4th International Conference on Information Systems Security and Privacy (ICISSP)},
  year={2018}
}
```

## License

This dataset is provided for research and educational purposes.
Please cite the original paper when using this dataset.

## Original Source

- https://www.unb.ca/cic/datasets/ids-2017.html
"""


# ============================================================================
#  Hugging Face Upload
# ============================================================================

def upload_to_huggingface(token, username, dataset_key):
    """Upload a dataset to Hugging Face."""
    cfg = DATASET_CONFIGS[dataset_key]
    repo_name = cfg["repo_name"]
    repo_id = f"{username}/{repo_name}"
    local_dir = cfg["local_dir"]

    print(f"\n{'=' * 60}")
    print(f"  Uploading {repo_name} to HuggingFace: {repo_id}")
    print(f"{'=' * 60}")

    # Check local data exists
    csv_files = glob.glob(os.path.join(local_dir, "*.csv"))
    if not csv_files:
        print(f"  [SKIP] No CSV files found in {local_dir}")
        print(f"         Run download step first.")
        return False

    total_size = sum(os.path.getsize(f) for f in csv_files)
    print(f"  Found {len(csv_files)} CSV files ({total_size / 1024 / 1024:.1f} MB total)")

    api = HfApi(token=token)

    # Create or get repo
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            exist_ok=True,
            private=False,
        )
        print(f"  [OK] Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"  [FAIL] Could not create repo: {e}")
        return False

    # Upload dataset card (README.md)
    cards = {"UNSW": UNSW_CARD, "CSE": CSE_CARD, "CIC2017": CIC2017_CARD}
    card_content = cards.get(dataset_key, "")

    if card_content:
        try:
            api.upload_file(
                path_or_fileobj=card_content.encode("utf-8"),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )
            print("  [OK] Dataset card uploaded")
        except Exception as e:
            print(f"  [WARN] Could not upload dataset card: {e}")

    # Upload CSV files
    for csv_path in sorted(csv_files):
        filename = os.path.basename(csv_path)
        size_mb = os.path.getsize(csv_path) / 1024 / 1024
        print(f"  Uploading {filename} ({size_mb:.1f} MB)...")

        try:
            api.upload_file(
                path_or_fileobj=csv_path,
                path_in_repo=f"data/{filename}",
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )
            print(f"  [OK] {filename} uploaded")
        except Exception as e:
            print(f"  [FAIL] {filename}: {e}")

    # Also upload any feature/metadata files
    meta_files = glob.glob(os.path.join(local_dir, "*.txt")) + \
                 glob.glob(os.path.join(local_dir, "*features*"))
    for meta_path in meta_files:
        filename = os.path.basename(meta_path)
        try:
            api.upload_file(
                path_or_fileobj=meta_path,
                path_in_repo=f"metadata/{filename}",
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )
            print(f"  [OK] Metadata: {filename} uploaded")
        except Exception:
            pass

    print(f"\n  Dataset available at: https://huggingface.co/datasets/{repo_id}")
    return True


# ============================================================================
#  Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download IDS datasets and upload to Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all datasets and upload to HuggingFace:
  python upload_datasets_to_hf.py --token hf_xxxx --username pcy12345BSU

  # Download only (no upload):
  python upload_datasets_to_hf.py --download-only

  # Upload only (data already downloaded):
  python upload_datasets_to_hf.py --upload-only --token hf_xxxx --username pcy12345BSU

  # Specific datasets only:
  python upload_datasets_to_hf.py --token hf_xxxx --username pcy12345BSU --datasets UNSW CSE
        """,
    )
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                        help="Hugging Face API token (or set HF_TOKEN env var)")
    parser.add_argument("--username", default="pcy12345BSU",
                        help="Hugging Face username (default: pcy12345BSU)")
    parser.add_argument("--datasets", nargs="+", default=["UNSW", "CSE", "CIC2017"],
                        choices=["UNSW", "CSE", "CIC2017"],
                        help="Which datasets to process (default: all three)")
    parser.add_argument("--download-only", action="store_true",
                        help="Only download datasets, don't upload")
    parser.add_argument("--upload-only", action="store_true",
                        help="Only upload (assumes data already downloaded)")
    args = parser.parse_args()

    print("=" * 60)
    print("  IDS Dataset Downloader & Hugging Face Uploader")
    print("=" * 60)
    print(f"  Datasets: {', '.join(args.datasets)}")
    print(f"  Data dir: {DATA_DIR}")
    if not args.download_only:
        print(f"  HF User:  {args.username}")
    print()

    os.makedirs(DATA_DIR, exist_ok=True)

    # ---- Download Phase ----
    if not args.upload_only:
        download_funcs = {
            "UNSW": download_unsw,
            "CSE": download_cse_2018,
            "CIC2017": download_cic_2017,
        }

        for ds in args.datasets:
            if ds in download_funcs:
                download_funcs[ds]()

    if args.download_only:
        print("\n[DONE] Download complete. Use --upload-only to upload later.")
        return

    # ---- Upload Phase ----
    if not args.token:
        print("\n[ERROR] HuggingFace token required for upload.")
        print("  Use --token YOUR_TOKEN or set HF_TOKEN environment variable.")
        print("  Create a token at: https://huggingface.co/settings/tokens")
        sys.exit(1)

    # Verify token
    api = HfApi(token=args.token)
    try:
        user_info = api.whoami()
        print(f"\n  Authenticated as: {user_info.get('name', 'unknown')}")
    except Exception as e:
        print(f"\n[ERROR] Authentication failed: {e}")
        sys.exit(1)

    for ds in args.datasets:
        upload_to_huggingface(args.token, args.username, ds)

    print("\n" + "=" * 60)
    print("  All done!")
    print("=" * 60)
    print(f"\n  Your datasets are available at:")
    for ds in args.datasets:
        repo_name = DATASET_CONFIGS[ds]["repo_name"]
        print(f"    https://huggingface.co/datasets/{args.username}/{repo_name}")


if __name__ == "__main__":
    main()
