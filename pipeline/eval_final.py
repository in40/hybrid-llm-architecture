"""
Frozen primary-endpoint evaluation: val_clean_bpb / val_masked_bpb on the FINAL
checkpoint (step 38147) of each seed, using the same harness as training
(make_val + C0Backend.eval_bpb, 256 x 2048 subset of frozen V-A).

Run on the training node:
  python3 eval_final.py --config c0_config.yaml --ckpts ckpt/c0_seed42_final.pt \
      ckpt/c0_seed123_final.pt ckpt/c0_seed456_final.pt [--device cpu]
"""
import argparse
import torch
import yaml
from tokenizers import Tokenizer
from model_backend import C0Backend
from run_c0 import make_val

ROOT = "/mnt/data2/c0_corpus"

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=f"{ROOT}/pipeline/c0_config.yaml")
ap.add_argument("--ckpts", nargs="+", required=True)
ap.add_argument("--device", default="cpu")
a = ap.parse_args()

cfg = yaml.safe_load(open(a.config))
cfg["training"]["max_steps"] = 100000  # scheduler irrelevant for eval
tok = Tokenizer.from_file(f"{ROOT}/tokenizer_50k.json")
d = cfg["data"]
val = make_val(f"{ROOT}/tokens/val.bin", d["seq_len"], d.get("val_seqs", 256))
print(f"[eval] val_seqs={len(val)} device={a.device}", flush=True)

for path in a.ckpts:
    be = C0Backend(cfg, tok, device=a.device)
    ck = torch.load(path, map_location=a.device, weights_only=False)
    be.model.load_state_dict(ck["model"])
    be.model.eval()
    torch.manual_seed(0)
    clean = be.eval_bpb(val, masked=False)
    torch.manual_seed(0)
    masked = be.eval_bpb(val, masked=True)
    print(f"[eval] {path.split('/')[-1]} step={ck['step']} "
          f"val_clean_bpb={clean:.4f} val_masked_bpb={masked:.4f}", flush=True)
