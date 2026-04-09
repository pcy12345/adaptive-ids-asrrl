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
        "eugenesiow/CIC-IDS-2017",
        "akabircs/CIC-IDS2017",
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

    # Fallback: manual copy
    print("\n=== Step 3: Manual file copy fallback ===")
    for source in source_repos:
        print(f"\nTrying manual copy from {source}...")
        try:
            files = api.list_repo_files(repo_id=source, repo_type="dataset")
            print(f"  Files in {source}: {files}")
        except Exception as e:
            print(f"  list_repo_files failed for {source}: {e}")
            traceback.print_exc()
            continue

    # Search HuggingFace
    print("\n=== Step 4: Search for any CIC-IDS 2017 dataset ===")
    try:
        results = list(api.list_datasets(search="CIC-IDS", limit=30))
        print(f"Found {len(results)} datasets matching 'CIC-IDS':")
        for ds in results:
            print(f"  - {ds.id} (downloads: {ds.downloads})")
    except Exception as e:
        print(f"Search failed: {e}")
        traceback.print_exc()

    print("\nERROR: Could not duplicate from any source!")
    sys.exit(1)


if __name__ == "__main__":
    main()
