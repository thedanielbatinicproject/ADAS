import os
import subprocess
import sys
import time
from pathlib import Path

#GUIDE: create ".env" file inside scripts folder of a project and create variable in one line like HF_TOKEN=your_token

# ===============================
# CONFIGURATION
# ===============================
DATASET_PATH = Path("data/raw")    # path to your dataset
BATCH_SIZE_MB = 500                        # max batch size per commit
REMOTE_BRANCH = "master"                   # branch to push
REMOTE_NAME = "origin"                     # git remote name
MAX_RETRIES = 5                            # max push retries
RETRY_DELAY = 60                           # seconds between retries

# ===============================
# HELPER FUNCTIONS
# ===============================
def run(cmd, cwd=None):
    """Run a shell command and return output, raise exception on failure."""
    print(f"\n[RUN] {cmd}")
    process = subprocess.Popen(cmd, shell=True, cwd=cwd,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout_lines = []
    stderr_lines = []
    # Print output in real time
    while True:
        out = process.stdout.readline()
        err = process.stderr.readline()
        if out:
            print(out, end='')
            stdout_lines.append(out)
        if err:
            print(err, end='', file=sys.stderr)
            stderr_lines.append(err)
        if out == '' and err == '' and process.poll() is not None:
            break
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{''.join(stderr_lines)}")
    return ''.join(stdout_lines).strip()

def get_folder_size_mb(folder):
    """Calculate folder size in MB."""
    total = 0
    for root, _, files in os.walk(folder):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / (1024 * 1024)

def init_git_lfs():
    """Initialize Git LFS if not already."""
    print("Initializing Git LFS...")
    run("git lfs install")
    run("git lfs track '*.png'")
    run("git add .gitattributes")

def commit_and_push_batch(batch_paths, batch_index):
    """Commit and push a batch of files."""
    print(f"\n[Batch {batch_index}] Adding {len(batch_paths)} files...")
    for path in batch_paths:
        run(f"git add '{path}'")
    
    run(f'git commit -m "Upload batch {batch_index}"')
    
    # Push with retry logic
    retries = 0
    while retries < MAX_RETRIES:
        try:
            print(f"[Batch {batch_index}] Pushing to remote...")
            run(f"git lfs push {REMOTE_NAME} {REMOTE_BRANCH}")
            run(f"git push {REMOTE_NAME} {REMOTE_BRANCH}")
            print(f"[Batch {batch_index}] Push successful!")
            return
        except RuntimeError as e:
            retries += 1
            print(f"[Batch {batch_index}] Push failed. Retry {retries}/{MAX_RETRIES} in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)
    print(f"[Batch {batch_index}] FAILED to push after {MAX_RETRIES} retries.")
    sys.exit(1)

def batch_files_by_size(root_folder, max_batch_mb):
    """Split files into batches not exceeding max_batch_mb."""
    batch = []
    batch_size = 0
    batches = []
    
    for folder, _, files in os.walk(root_folder):
        for f in files:
            fpath = os.path.join(folder, f)
            fsize_mb = os.path.getsize(fpath) / (1024 * 1024)
            if batch_size + fsize_mb > max_batch_mb and batch:
                batches.append(batch)
                batch = []
                batch_size = 0
            batch.append(fpath)
            batch_size += fsize_mb
    if batch:
        batches.append(batch)
    return batches

# ===============================
# MAIN SCRIPT
# ===============================
def main():
    if not DATASET_PATH.exists():
        print(f"Dataset path {DATASET_PATH} does not exist.")
        sys.exit(1)

    init_git_lfs()

    batches = batch_files_by_size(DATASET_PATH, BATCH_SIZE_MB)
    print(f"Total batches to commit: {len(batches)}")

    for i, batch in enumerate(batches, start=1):
        commit_and_push_batch(batch, i)

    print("\nAll batches committed and pushed successfully!")

if __name__ == "__main__":
    main()