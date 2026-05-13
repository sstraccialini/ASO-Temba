#!/usr/bin/env python3
"""
Parse AttentionX3 ablation training logs and produce CSV + Markdown tables.

Usage:
    python scripts/parse_ablation_results.py \
        --root /path/to/attx3_ablations \
        --reference-root /path/to/attx3_tuned_v2 \
        --out /path/to/output_dir

The script scans both roots for training.log files, extracts Final Full-val-map
and Final sampled-val-map metrics, and writes:
    <out>/ablation_results.csv
    <out>/ablation_results.md
"""

import argparse
import ast
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Metric ordering for the Markdown table
# ---------------------------------------------------------------------------
FUSER_ORDER = [
    'sum',
    'attention_x3_no_attn',
    'attention_x3_no_ffn',
    'attention_x3_bn',
    'attention_x3_shared_common',
    'attention_x3',
]

# Patterns for the two final metrics we care about (ignore Block N lines)
RE_FINAL_FULL = re.compile(r'Epoch\s+(\d+),\s+Final Full-val-map:\s+([\d.]+)')
RE_FINAL_SAMPLED = re.compile(r'Epoch\s+(\d+),\s+Final sampled-val-map:\s+([\d.]+)')

# Errors that mark a crashed/incomplete run
ERROR_PATTERNS = [
    'Traceback',
    'RuntimeError',
    'ValueError',
    'CUDA out of memory',
    'num_samples=0',
]

# Arguments we parse from the Namespace line
NAMESPACE_KEYS = [
    'fuser', 'dataset', 'backbone', 'rgb_root', 'output_dir',
    'epochs', 'lr', 'weight_decay', 'drop', 'drop_path',
    'warmup_epochs', 'diversity_loss_weight', 'block_loss_weight',
]


def _parse_namespace_line(line: str) -> dict:
    """Extract key=value pairs from 'Arguments: Namespace(...)' lines."""
    m = re.search(r'Namespace\((.+)\)', line)
    if not m:
        return {}
    inner = m.group(1)
    result = {}
    # tokenize as key=value pairs, values may be strings/numbers/booleans
    for part in re.finditer(r'(\w+)=(\'[^\']*\'|"[^"]*"|[^,]+)', inner):
        k, v = part.group(1), part.group(2).strip()
        if k in NAMESPACE_KEYS:
            try:
                result[k] = ast.literal_eval(v)
            except Exception:
                result[k] = v
    return result


def parse_log(log_path: Path) -> dict:
    """Parse a single training.log and return a result dict."""
    text = log_path.read_text(errors='replace')
    lines = text.splitlines()

    row = {
        'log_path': str(log_path),
        'run_name': log_path.parent.name,
        'fuser': None,
        'dataset': None,
        'backbone': None,
        'rgb_root': None,
        'output_dir': None,
        'epochs': None,
        'lr': None,
        'weight_decay': None,
        'drop': None,
        'drop_path': None,
        'warmup_epochs': None,
        'diversity_loss_weight': None,
        'block_loss_weight': None,
        'completed_epochs': 0,
        'best_epoch_by_final_full_map': None,
        'best_final_full_map': None,
        'final_sampled_map_at_best_full_epoch': None,
        'best_final_sampled_map': None,
        'last_epoch': None,
        'last_final_full_map': None,
        'last_final_sampled_map': None,
        'crashed_or_incomplete': False,
        'error_summary': '',
        'delta_vs_baseline': None,
        'delta_vs_final_attx3': None,
    }

    # Parse namespace
    for line in lines:
        if 'Arguments: Namespace' in line or 'Namespace(' in line:
            ns = _parse_namespace_line(line)
            for k in NAMESPACE_KEYS:
                if k in ns:
                    row[k] = ns[k]
            break

    # Collect per-epoch metrics
    full_by_epoch: dict[int, float] = {}
    sampled_by_epoch: dict[int, float] = {}

    error_lines = []
    for line in lines:
        mf = RE_FINAL_FULL.search(line)
        if mf:
            ep, val = int(mf.group(1)), float(mf.group(2))
            full_by_epoch[ep] = val

        ms = RE_FINAL_SAMPLED.search(line)
        if ms:
            ep, val = int(ms.group(1)), float(ms.group(2))
            sampled_by_epoch[ep] = val

        for pat in ERROR_PATTERNS:
            if pat in line:
                error_lines.append(line.strip()[:120])

    row['error_summary'] = ' | '.join(error_lines[:3]) if error_lines else ''

    if full_by_epoch:
        best_ep = max(full_by_epoch, key=full_by_epoch.__getitem__)
        last_ep = max(full_by_epoch)
        row['completed_epochs'] = len(full_by_epoch)
        row['best_epoch_by_final_full_map'] = best_ep
        row['best_final_full_map'] = full_by_epoch[best_ep]
        row['final_sampled_map_at_best_full_epoch'] = sampled_by_epoch.get(best_ep)
        row['best_final_sampled_map'] = max(sampled_by_epoch.values()) if sampled_by_epoch else None
        row['last_epoch'] = last_ep
        row['last_final_full_map'] = full_by_epoch[last_ep]
        row['last_final_sampled_map'] = sampled_by_epoch.get(last_ep)

        declared_epochs = row.get('epochs')
        if declared_epochs is not None:
            # epochs is 0-indexed in logs (0 … epochs-1)
            row['crashed_or_incomplete'] = last_ep < int(declared_epochs) - 1
        else:
            row['crashed_or_incomplete'] = bool(error_lines)
    else:
        row['crashed_or_incomplete'] = True

    if error_lines:
        row['crashed_or_incomplete'] = True

    return row


def scan_root(root: Path) -> list[dict]:
    rows = []
    for log_path in sorted(root.rglob('training.log')):
        rows.append(parse_log(log_path))
    return rows


def add_deltas(rows: list[dict]) -> None:
    """Compute delta_vs_baseline and delta_vs_final_attx3 in-place."""
    # Build lookup: (dataset, backbone) → best_final_full_map for baseline fusers
    baseline: dict[tuple, float] = {}
    attx3: dict[tuple, float] = {}
    for r in rows:
        key = (r.get('dataset'), r.get('backbone'))
        val = r.get('best_final_full_map')
        if val is None:
            continue
        if r.get('fuser') == 'sum':
            baseline[key] = val
        elif r.get('fuser') == 'attention_x3':
            attx3[key] = val

    for r in rows:
        key = (r.get('dataset'), r.get('backbone'))
        val = r.get('best_final_full_map')
        if val is None:
            continue
        if key in baseline:
            r['delta_vs_baseline'] = round(val - baseline[key], 4)
        if key in attx3:
            r['delta_vs_final_attx3'] = round(val - attx3[key], 4)


FIELDNAMES = [
    'fuser', 'dataset', 'backbone', 'run_name', 'log_path',
    'completed_epochs', 'best_epoch_by_final_full_map',
    'best_final_full_map', 'final_sampled_map_at_best_full_epoch',
    'best_final_sampled_map',
    'last_epoch', 'last_final_full_map', 'last_final_sampled_map',
    'crashed_or_incomplete', 'error_summary',
    'delta_vs_baseline', 'delta_vs_final_attx3',
    'epochs', 'lr', 'weight_decay', 'drop', 'drop_path',
    'warmup_epochs', 'diversity_loss_weight', 'block_loss_weight',
]


def write_csv(rows: list[dict], out_path: Path) -> None:
    with out_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _fmtf(v, digits=4) -> str:
    if v is None:
        return '-'
    try:
        return f'{float(v):.{digits}f}'
    except (TypeError, ValueError):
        return str(v)


def _fmtdelta(v) -> str:
    if v is None:
        return '-'
    try:
        f = float(v)
        sign = '+' if f >= 0 else ''
        return f'{sign}{f:.4f}'
    except (TypeError, ValueError):
        return str(v)


def write_markdown(rows: list[dict], out_path: Path) -> None:
    # Group by (dataset, backbone)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r.get('dataset', '?'), r.get('backbone', '?'))].append(r)

    lines = ['# AttentionX3 Ablation Results\n',
             '_Primary metric: **Final Full-val-map** (best epoch)._\n']

    header = (
        '| Fuser | Best Full mAP | Sampled@BestFull | Best Sampled mAP '
        '| Last Full mAP | Δ vs sum | Δ vs attx3 | Epochs | Crashed |\n'
        '|---|---|---|---|---|---|---|---|---|\n'
    )

    for (dataset, backbone), group_rows in sorted(groups.items()):
        lines.append(f'\n## {dataset} / {backbone}\n\n')
        lines.append(header)

        fuser_order_map = {f: i for i, f in enumerate(FUSER_ORDER)}
        sorted_rows = sorted(
            group_rows,
            key=lambda r: fuser_order_map.get(r.get('fuser', ''), 999)
        )

        for r in sorted_rows:
            crashed = '**YES**' if r.get('crashed_or_incomplete') else 'no'
            lines.append(
                f"| {r.get('fuser', '?')} "
                f"| {_fmtf(r.get('best_final_full_map'))} "
                f"| {_fmtf(r.get('final_sampled_map_at_best_full_epoch'))} "
                f"| {_fmtf(r.get('best_final_sampled_map'))} "
                f"| {_fmtf(r.get('last_final_full_map'))} "
                f"| {_fmtdelta(r.get('delta_vs_baseline'))} "
                f"| {_fmtdelta(r.get('delta_vs_final_attx3'))} "
                f"| {r.get('completed_epochs', '-')}/{r.get('epochs', '?')} "
                f"| {crashed} |\n"
            )

    out_path.write_text(''.join(lines))


def main():
    parser = argparse.ArgumentParser(description='Parse AttentionX3 ablation logs')
    parser.add_argument('--root', required=True, help='Root directory of ablation outputs')
    parser.add_argument('--reference-root', default=None,
                        help='Optional root with reference runs (e.g. tuned_v2 with sum/attention_x3)')
    parser.add_argument('--out', required=True, help='Output directory for CSV and Markdown')
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = scan_root(root)
    print(f'Found {len(rows)} logs under {root}', file=sys.stderr)

    if args.reference_root:
        ref_root = Path(args.reference_root)
        ref_rows = scan_root(ref_root)
        print(f'Found {len(ref_rows)} reference logs under {ref_root}', file=sys.stderr)
        rows = ref_rows + rows  # reference first so baselines are found first

    add_deltas(rows)

    csv_path = out_dir / 'ablation_results.csv'
    md_path = out_dir / 'ablation_results.md'
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)

    print(f'CSV:      {csv_path}')
    print(f'Markdown: {md_path}')


if __name__ == '__main__':
    main()
