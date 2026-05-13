#!/usr/bin/env python3
"""
Smoke-test all AttentionX3 ablation fusers by running a single forward pass
for each variant with a dummy CLIP-shaped input.

Run from vim/:
    python scripts/smoke_test_ablation_fusers.py

If Mamba CUDA extensions are unavailable on CPU, the script will warn and exit
rather than crash silently.
"""

import sys
import traceback
import torch

FUSERS_TO_TEST = [
    'sum',
    'attention_x3',
    'attention_x3_no_attn',
    'attention_x3_no_ffn',
    'attention_x3_bn',
    'attention_x3_shared_common',
]

# Dummy input: B=2, C=768 (CLIP-L/14), T=32
B, C_IN, T = 2, 768, 32
NUM_CLASSES = 51

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cpu':
    print(
        '[WARNING] CUDA not available. Mamba ops require CUDA kernels and will '
        'likely fail on CPU. Install CUDA or run on a GPU node.',
        file=sys.stderr,
    )


def run_fuser(fuser_name: str) -> str:
    """Instantiate MSTemba with the given fuser and run one forward pass."""
    try:
        from models_MSTemba import MSTemba
    except ImportError as e:
        return f'IMPORT ERROR: {e}'

    try:
        model = MSTemba(
            in_feat_dim=C_IN,
            num_classes=NUM_CLASSES,
            embed_dims=[256, 384, 576],
            depths=[1, 1, 1],
            d_state=16,
            fuser=fuser_name,
        ).to(device)
        model.eval()

        # Input: (B, C, T)  — same convention as MSTemba_main DataLoader
        x = torch.randn(B, C_IN, T, device=device)

        with torch.no_grad():
            out = model(x)

        # Unpack — forward returns (logits, block_preds, diversity_loss, fusion_weights)
        if isinstance(out, (tuple, list)):
            logits = out[0]
            block_preds = out[1] if len(out) > 1 else None
            diversity_loss = out[2] if len(out) > 2 else None
            fusion_weights = out[3] if len(out) > 3 else None
        else:
            logits = out
            block_preds = None
            diversity_loss = None
            fusion_weights = None

        # Checks
        assert logits.shape == (B, T, NUM_CLASSES), \
            f'Bad logits shape: {logits.shape}, expected ({B}, {T}, {NUM_CLASSES})'

        if block_preds is not None:
            assert len(block_preds) == 3, \
                f'Expected 3 block predictions, got {len(block_preds)}'

        if diversity_loss is not None:
            assert isinstance(diversity_loss, torch.Tensor), \
                'diversity_loss should be a tensor'

        fw_ok = fusion_weights is None or isinstance(fusion_weights, torch.Tensor)
        assert fw_ok, f'fusion_weights has unexpected type: {type(fusion_weights)}'

        return 'OK'

    except Exception:
        return f'FAILED\n{traceback.format_exc()}'


def main():
    print(f'Device: {device}')
    print(f'Input shape: B={B}, C={C_IN}, T={T}, num_classes={NUM_CLASSES}\n')

    results = {}
    for fuser in FUSERS_TO_TEST:
        print(f'  Testing {fuser} ... ', end='', flush=True)
        status = run_fuser(fuser)
        first_line = status.splitlines()[0]
        print(first_line)
        if status != 'OK':
            for line in status.splitlines()[1:]:
                print(f'    {line}')
        results[fuser] = status

    print('\n--- Summary ---')
    all_ok = True
    for fuser, status in results.items():
        icon = '✓' if status == 'OK' else '✗'
        print(f'  {icon}  {fuser}: {status.splitlines()[0]}')
        if status != 'OK':
            all_ok = False

    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
