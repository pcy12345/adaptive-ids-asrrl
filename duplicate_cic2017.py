#!/usr/bin/env python3
"""Duplicate CIC-IDS-2017 from an existing HuggingFace dataset."""

import os
import sys
import traceback
from huggingface_hub import HfApi


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set")
        sys.exit(1)

    print(f"Token length: {len(token)}")
    print(f"Token prefix: {token[:5]}...")

    api = HfApi(token=token)

    # Verify auth
    print("\n=== Step 1: Verify authentication ===")
    try:
        user = api.whoami()
        print(f"Authenticated as: {user}")
    except Exception as e:
        print(f"Auth failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Try duplicate_repo first (simplest approach)
    print("\n=== Step 2: Try duplicate_repo ===")
    source_repos = [
        "rdpahalavan/CIC-IDS2017",
        "c01dsnap/CIC-IDS2017",
        "bvk/CICIDS-2017",
    ]

    for source in source_repos:
        print(f"\nTrying to duplicate {source}...")
        try:
            result = api.duplicate_repo(
                from_id=source,
                to_id="pcy12345BSU/CIC-IDS-2017",
                repo_type="dataset",
                token=token,
                exist_ok=True,
            )
            print(f"SUCCESS! Result: {result}")
            print(f"Dataset: https://huggingface.co/datasets/pcy12345BSU/CIC-IDS-2017")
            return
        except Exception as e:
            print(f"  duplicate_repo failed: {e}")
            traceback.print_exc()

    # Fallback: manual file-by-file copy
    print("\n=== Step 3: Manual file copy fallback ===")
    from huggingface_hub import hf_hub_download, create_repo

    for source in source_repos:
        print(f"\nTrying manual copy from {source}...")
        try:
            files = api.list_repo_files(repo_id=source, repo_type="dataset")
            print(f"  Found {len(files)} files in {source}")
            for f in files[:20]:
                print(f"    {f}")

            if not files:
                continue

            create_repo(
                repo_id="pcy12345BSU/CIC-IDS-2017",
                repo_type="dataset",
                token=token,
                exist_ok=True,
                private=False,
            )

            uploaded = 0
            for f in files:
                if f.startswith("."):
                    continue
                print(f"  Copying: {f} ...", end=" ", flush=True)
                try:
                    local_path = hf_hub_download(
                        repo_id=source, filename=f,
                        repo_type="dataset", token=token,
                    )
                    api.upload_file(
                        path_or_fileobj=local_path, path_in_repo=f,
                        repo_id="pcy12345BSU/CIC-IDS-2017",
                        repo_type="dataset", token=token,
                    )
                    uploaded += 1
                    print("[OK]")
                except Exception as e:
                    print(f"[FAIL] {e}")

            if uploaded > 0:
                print(f"\nSUCCESS: Copied {uploaded} files from {source}")
                return
        except Exception as e:
            print(f"  Error: {e}")
            traceback.print_exc()

    print("\nERROR: Could not duplicate from any source!")
    sys.exit(1)


if __name__ == "__main__":
    main()
