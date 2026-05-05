#!/usr/bin/env python3
"""Quick utility to inspect PyTorch checkpoint contents.

Usage: python vim/inspect_checkpoint.py --ckpt /path/to/best_model.pth
"""
import argparse
import torch

def main():
    p = argparse.ArgumentParser(description="Inspect checkpoint keys")
    p.add_argument('--ckpt', required=True, help='Path to checkpoint file')
    args = p.parse_args()

    data = torch.load(args.ckpt, map_location='cpu')
    print(f"Loaded checkpoint type: {type(data)}")

    if not isinstance(data, dict):
        print("Checkpoint is not a dict; nothing to inspect further.")
        return

    top_keys = list(data.keys())
    print(f"Top-level keys ({len(top_keys)}): {top_keys}")

    # If it contains a model state, inspect those keys
    for candidate in ('model_ema_state_dict','model_ema','model_state_dict','model','state_dict'):
        if candidate in data and data[candidate] is not None:
            sd = data[candidate]
            print(f"\nFound '{candidate}' with {len(sd)} keys.")
            # look for fuser-related params
            fuser_keys = [k for k in sd.keys() if 'fuser' in k or 'fuser_q' in k or 'fuser_k' in k or 'fuser_v' in k]
            print(f"Example model keys (first 20): {list(sd.keys())[:20]}")
            print(f"fuser-related keys ({len(fuser_keys)}): {fuser_keys}")
            break
    else:
        print("No model state dict found under common keys. You may be saving a raw state_dict differently.")

    # Also list any unexpected nested structures for debugging
    non_tensor_values = {k: type(v) for k,v in data.items() if not isinstance(v, dict) and not torch.is_tensor(v) }
    if non_tensor_values:
        print('\nNon-dict / non-tensor top-level entries:')
        for k, t in non_tensor_values.items():
            print(f" - {k}: {t}")

if __name__ == '__main__':
    main()
