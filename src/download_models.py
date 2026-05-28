import os

from huggingface_hub import snapshot_download


def main():
    os.makedirs('./checkpoints', exist_ok=True)
    snapshot_download(
        repo_id="Erosinho/M2D2-Llama-115M-fixed-attn-experts",
        local_dir="./checkpoints",
        local_dir_use_symlinks=False
    )  # type: ignore
    

if __name__ == '__main__':
    main()
