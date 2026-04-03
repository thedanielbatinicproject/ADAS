from huggingface_hub import snapshot_download
import argparse


def download_dataset(repo_id: str, local_dir: str) -> None:
    print(f"Downloading dataset '{repo_id}' to '{local_dir}'...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        max_workers=4,
    )
    print("Download complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download dataset from Hugging Face Hub"
    )
    parser.add_argument("--repo-id", type=str, default="DBatinic/DADA2000")
    parser.add_argument("--local-dir", type=str, default="data/raw")
    args = parser.parse_args()

    download_dataset(args.repo_id, args.local_dir)
