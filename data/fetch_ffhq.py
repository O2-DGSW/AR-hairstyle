"""FFHQ 1024 샤드 1개(약 13GB, 약 10,000장)를 받는다.

전체 89GB 중 1/7 만 받는다. rotate_gen 기본 데이터셋 크기가 10,000장이라
샤드 하나로 논문과 같은 규모가 나온다.
"""
import sys
from huggingface_hub import hf_hub_download

if __name__ == "__main__":
    shard = sys.argv[1] if len(sys.argv) > 1 else "shard_1_of_7.zip"
    p = hf_hub_download(repo_id="pravsels/FFHQ_1024", filename=shard,
                        repo_type="dataset",
                        local_dir=r"C:\Users\rr444\Documents\projects\heddy-v2\data\ffhq")
    print("DONE", p)
