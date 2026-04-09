#!/usr/bin/env python3
"""Duplicate CIC-IDS-2017 from an existing HuggingFace dataset."""

import os
import sys
from huggingface_hub import HfApi, hf_hub_download, create_repo


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set")
        sys.exit(1)

    api = HfApi(token=token)
    target_repo = "pcy12345BSU/CIC-IDS-2017"

    # Verify auth
    try:
        user = api.whoami()
        print(f"Authenticated as: {user.get('name', 'unknown')}")
    except Exception as e:
        print(f"Auth failed: {e}")
        sys.exit(1)

    # Search for existing CIC-IDS-2017 datasets
    print("\n=== Searching HuggingFace for CIC-IDS-2017 ===")
    try:
        results = list(api.list_datasets(search="CIC-IDS-2017", sort="downloads", direction=-1, limit=20))
        print(f"Found {len(results)} datasets:")
        for ds in results:
            print(f"  - {ds.id} (downloads: {ds.downloads})")
    except Exception as e:
        print(f"Search failed: {e}")
        results = []

    # Build list of source repos to try
    source_repos = [
        "eugenesiow/CIC-IDS-2017",
        "akabircs/CIC-IDS2017",
        "Helios-IIITD/CIC-IDS-2017",
    ]
    for ds in results:
        if ds.id not in source_repos:
            source_repos.append(ds.id)

    print(f"\nWill try: {source_repos[:8]}")

    for source_repo in source_repos[:8]:
        print(f"\n=== Trying {source_repo} ===")
        try:
            files = api.list_repo_files(repo_id=source_repo, repo_type="dataset")
            print(f"  Found {len(files)} files")
            for f in files[:20]:
                print(f"    {f}")

            if not files:
                print("  Empty repo, skipping")
                continue

            # Create target repo
            create_repo(
                repo_id=target_repo,
                repo_type="dataset",
                token=token,
                exist_ok=True,
                private=False,
            )
            print(f"  Target repo ready: {target_repo}")

            # Copy files
            uploaded = 0
            for f in files:
                if f.startswith("."):
                    continue
                print(f"  Copying: {f} ...", end=" ", flush=True)
                try:
                    local_path = hf_hub_download(
                        repo_id=source_repo,
                        filename=f,
                        repo_type="dataset",
                        token=token,
                    )
                    api.upload_file(
                        path_or_fileobj=local_path,
                        path_in_repo=f,
                        repo_id=target_repo,
                        repo_type="dataset",
                        token=token,
                    )
                    uploaded += 1
                    print("[OK]")
                except Exception as e:
                    print(f"[FAIL] {e}")

            if uploaded > 0:
                print(f"\n=== SUCCESS ===")
                print(f"Copied {uploaded} files from {source_repo}")
                print(f"Dataset: https://huggingface.co/datasets/{target_repo}")
                return
            else:
                print("  No files copied, trying next source")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    print("\nERROR: Could not duplicate from any source!")
    sys.exit(1)


if __name__ == "__main__":
    main()
