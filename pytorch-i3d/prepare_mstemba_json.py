import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # CSV Column Names
    "col_video_id": "video_id",
    "col_label": "event",
    "col_start": "start_frame",
    "col_end": "end_frame",

    # Annotation / feature pipeline
    "annotation_fps": 25.0,
    "feature_fps": 10.0,
    "frames_per_segment": 16,
    "max_segments": 2500,
}

SECONDS_PER_SEGMENT = CONFIG["frames_per_segment"] / CONFIG["feature_fps"]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def determine_valid_segments(features: np.ndarray) -> int:
    """
    Infer the number of valid (non-padded) segments.
    Assumes zero-padding may have been applied at the end.

    Expected feature shape:
      - [T, D] for segment features
    """
    if features.ndim != 2:
        raise ValueError(f"Expected feature array with shape [T, D], got {features.shape}")

    if features.shape[0] == 0:
        return 0

    non_zero_rows = np.any(features != 0, axis=1)
    if not np.any(non_zero_rows):
        return 0

    last_non_zero_idx = np.where(non_zero_rows)[0][-1]
    return int(last_non_zero_idx + 1)


def load_feature_info(feat_path: str) -> tuple[np.ndarray, int]:
    """
    Load feature file and return:
      - feature array
      - valid number of raw segments

    Priority:
      1. If .npz and raw_features exists -> use len(raw_features)
      2. If .npz and num_segments_raw exists -> use it directly
      3. Otherwise infer valid segments from trailing zero padding
    """
    if feat_path.endswith(".npz"):
        data = np.load(feat_path)

        if "raw_features" in data:
            features = data["raw_features"]
            valid_segments = int(features.shape[0])
            return features, valid_segments

        if "num_segments_raw" in data:
            if "features" in data:
                features = data["features"]
            else:
                first_key = list(data.keys())[0]
                features = data[first_key]
            valid_segments = int(data["num_segments_raw"])
            return features, valid_segments

        if "features" in data:
            features = data["features"]
        else:
            first_key = list(data.keys())[0]
            features = data[first_key]

        valid_segments = determine_valid_segments(features)
        return features, valid_segments

    else:
        features = np.load(feat_path)
        valid_segments = determine_valid_segments(features)
        return features, valid_segments


def convert_frame_to_feature_frame(frame_idx: float,
                                   annotation_fps: float,
                                   feature_fps: float) -> float:
    """
    Convert frame index from annotation FPS domain to feature-video FPS domain.
    """
    return frame_idx * feature_fps / annotation_fps


def interval_to_segment_indices(start_frame_anno: float,
                                end_frame_anno: float,
                                annotation_fps: float,
                                feature_fps: float,
                                frames_per_segment: int) -> tuple[int, int]:
    """
    Convert an action interval from annotation-frame indices to segment indices.

    Convention:
      start_seg = floor(start_frame_10fps / frames_per_segment)
      end_seg   = ceil((end_frame_10fps + 1) / frames_per_segment) - 1

    This covers all segments that intersect the action interval.
    """
    start_frame_feat = convert_frame_to_feature_frame(
        start_frame_anno, annotation_fps, feature_fps
    )
    end_frame_feat = convert_frame_to_feature_frame(
        end_frame_anno, annotation_fps, feature_fps
    )

    start_seg = int(np.floor(start_frame_feat / frames_per_segment))
    end_seg = int(np.ceil((end_frame_feat + 1.0) / frames_per_segment) - 1)

    return start_seg, end_seg


def clamp_segment_interval(start_seg: int, end_seg: int, valid_segments: int):
    """
    Clamp [start_seg, end_seg] to valid segment range [0, valid_segments-1].

    Returns None if the interval becomes empty/outside.
    """
    if valid_segments <= 0:
        return None

    start_seg = max(0, start_seg)
    end_seg = min(valid_segments - 1, end_seg)

    if end_seg < start_seg:
        return None

    return start_seg, end_seg


def normalize_subset_name(subset: str) -> str:
    """
    Normalize split names to MS-Temba style if needed.
    """
    subset = str(subset).strip().lower()
    if subset in {"train", "training"}:
        return "training"
    if subset in {"test", "testing", "val", "validation"}:
        return "testing"
    return subset


# =============================================================================
# MAIN LOGIC
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert TSU CSVs to MS-Temba JSON format using SEGMENT indices."
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        required=True,
        help="Directory containing .npy/.npz feature files"
    )
    parser.add_argument(
        "--annotations_dir",
        type=str,
        required=True,
        help="Directory containing nested annotation CSVs"
    )
    parser.add_argument(
        "--class_mapping",
        type=str,
        required=True,
        help="Path to class_mapping.json"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Output JSON path (e.g., MS-Temba/data/smarthome.json)"
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default=None,
        help="Optional CSV mapping video_id to subset (training/testing)"
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Load class mapping
    # -------------------------------------------------------------------------
    print(f"Loading class mapping from {args.class_mapping}...")
    with open(args.class_mapping, "r") as f:
        class_map = json.load(f)

    # -------------------------------------------------------------------------
    # 2. Discover features
    # -------------------------------------------------------------------------
    print(f"Scanning features in {args.features_dir}...")
    feat_paths = glob.glob(os.path.join(args.features_dir, "*.npy"))
    feat_paths += glob.glob(os.path.join(args.features_dir, "*.npz"))

    feat_dict = {Path(p).stem: p for p in feat_paths}
    print(f"Found {len(feat_dict)} feature files.")

    if not feat_dict:
        raise ValueError(f"No feature files found in {args.features_dir}")

    # -------------------------------------------------------------------------
    # 3. Load split file if provided
    # -------------------------------------------------------------------------
    split_dict = {}
    if args.split_file and os.path.exists(args.split_file):
        print(f"Loading splits from {args.split_file}...")
        split_df = pd.read_csv(args.split_file)

        if "video_id" in split_df.columns and "subset" in split_df.columns:
            split_dict = {
                str(v).strip(): normalize_subset_name(s)
                for v, s in zip(split_df["video_id"], split_df["subset"])
            }
        else:
            print("Warning: split_file must contain 'video_id' and 'subset' columns. Ignoring.")
    else:
        print("No split_file provided. Defaulting all videos to 'training'.")

    # -------------------------------------------------------------------------
    # 4. Discover and load CSV annotations
    # -------------------------------------------------------------------------
    print(f"Scanning annotations in {args.annotations_dir}...")
    csv_paths = glob.glob(os.path.join(args.annotations_dir, "**", "*.csv"), recursive=True)

    if not csv_paths:
        raise ValueError(f"No CSV files found in {args.annotations_dir}")

    dfs = []
    for csv_path in csv_paths:
        try:
            df_part = pd.read_csv(csv_path)
            video_name = Path(csv_path).stem

            if CONFIG["col_video_id"] not in df_part.columns:
                df_part[CONFIG["col_video_id"]] = video_name

            dfs.append(df_part)
        except Exception as e:
            print(f"Warning: Failed to read {csv_path}: {e}")

    if not dfs:
        raise ValueError("No valid annotation CSVs could be loaded.")

    df = pd.concat(dfs, ignore_index=True)

    required_cols = [
        CONFIG["col_video_id"],
        CONFIG["col_label"],
        CONFIG["col_start"],
        CONFIG["col_end"],
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in loaded annotation dataframe.")

    df[CONFIG["col_video_id"]] = df[CONFIG["col_video_id"]].astype(str).str.strip()
    df[CONFIG["col_label"]] = df[CONFIG["col_label"]].astype(str).str.strip()

    # -------------------------------------------------------------------------
    # 5. Build JSON
    # -------------------------------------------------------------------------
    output_data = {}
    stats = {
        "videos_written": 0,
        "actions_written": 0,
        "missing_features": 0,
        "unknown_labels": 0,
        "invalid_rows": 0,
        "actions_dropped_outside_range": 0,
    }

    csv_video_ids = set(df[CONFIG["col_video_id"]].unique())
    unknown_labels_seen = set()
    video_groups = df.groupby(CONFIG["col_video_id"])

    print("\nProcessing videos...")
    debug_examples = []

    for video_id, feat_path in feat_dict.items():
        try:
            features, valid_segs = load_feature_info(feat_path)
        except Exception as e:
            print(f"Error reading feature file for {video_id}: {e}")
            continue

        subset = split_dict.get(video_id, "training")
        subset = normalize_subset_name(subset)

        video_actions = []

        if video_id in video_groups.groups:
            video_annos = video_groups.get_group(video_id)

            for _, row in video_annos.iterrows():
                start_val = row[CONFIG["col_start"]]
                end_val = row[CONFIG["col_end"]]
                action = row[CONFIG["col_label"]]

                if pd.isna(start_val) or pd.isna(end_val):
                    stats["invalid_rows"] += 1
                    continue

                try:
                    start_frame = float(start_val)
                    end_frame = float(end_val)
                except Exception:
                    stats["invalid_rows"] += 1
                    continue

                if end_frame < start_frame:
                    stats["invalid_rows"] += 1
                    continue

                if action not in class_map:
                    stats["unknown_labels"] += 1
                    unknown_labels_seen.add(action)
                    continue

                class_id = class_map[action]

                start_seg, end_seg = interval_to_segment_indices(
                    start_frame_anno=start_frame,
                    end_frame_anno=end_frame,
                    annotation_fps=CONFIG["annotation_fps"],
                    feature_fps=CONFIG["feature_fps"],
                    frames_per_segment=CONFIG["frames_per_segment"],
                )

                clamped = clamp_segment_interval(start_seg, end_seg, valid_segs)
                if clamped is None:
                    stats["actions_dropped_outside_range"] += 1
                    continue

                start_seg_clamped, end_seg_clamped = clamped

                video_actions.append([
                    int(class_id),
                    int(start_seg_clamped),
                    int(end_seg_clamped)
                ])

        output_data[video_id] = {
            "subset": subset,
            "duration": int(valid_segs),
            "actions": video_actions
        }

        if len(debug_examples) < 5:
            debug_examples.append({
                "video_id": video_id,
                "valid_segments": int(valid_segs),
                "duration_seconds_equivalent": round(valid_segs * SECONDS_PER_SEGMENT, 2),
                "num_actions": len(video_actions)
            })

        stats["videos_written"] += 1
        stats["actions_written"] += len(video_actions)

    # -------------------------------------------------------------------------
    # 6. Validation / reporting
    # -------------------------------------------------------------------------
    videos_missing_features = csv_video_ids - set(feat_dict.keys())
    stats["missing_features"] = len(videos_missing_features)

    unreferenced_features = set(feat_dict.keys()) - csv_video_ids

    with open(args.output_json, "w") as f:
        json.dump(output_data, f, indent=4)

    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    print(f"Target JSON saved to: {args.output_json}")
    print(f"Videos written: {stats['videos_written']}")
    print(f"Action annotations written: {stats['actions_written']}")
    print(f"Invalid rows skipped: {stats['invalid_rows']}")
    print(f"Unknown labels skipped: {stats['unknown_labels']}")
    print(f"Actions dropped outside valid feature range: {stats['actions_dropped_outside_range']}")

    print("\n" + "-" * 60)
    print("VALIDATION REPORT")
    print("-" * 60)

    if videos_missing_features:
        preview = sorted(list(videos_missing_features))[:5]
        print(f"[{len(videos_missing_features)}] Videos in CSV missing features (e.g. {preview})")
    else:
        print("[0] Videos in CSV missing features")

    if unreferenced_features:
        preview = sorted(list(unreferenced_features))[:5]
        print(f"[{len(unreferenced_features)}] Feature files found with NO matching annotations (added as empty) (e.g. {preview})")
    else:
        print("[0] Unreferenced feature files")

    if unknown_labels_seen:
        print(f"\n[{len(unknown_labels_seen)}] Labels in CSV missing from class_mapping.json:")
        for ul in sorted(unknown_labels_seen):
            print(f"  - '{ul}'")

    print("\nSample duration sanity check:")
    for ex in debug_examples:
        print(
            f"  {ex['video_id']}: "
            f"duration={ex['valid_segments']} segs "
            f"(~{ex['duration_seconds_equivalent']} s), "
            f"actions={ex['num_actions']}"
        )


if __name__ == "__main__":
    main()
