"""Build final annotation JSON for Stage2 training/val from atomic clips.

Reads the intermediate annotation from prepare_atomic_clips.py and matches
with tokenized pose files from BodyTokenize inference_atomic.py.

Generates entries for both ego and exo views (as separate training samples).

Usage:
    python build_annotation_atomic.py [--split {train,val}] [--exo-only]
    python build_annotation_atomic.py --split val --output annotation_atomic_val.json
    python build_annotation_atomic.py --exo-only  # exo view only
"""

import argparse
import json
import os
from pathlib import Path


_DATA_ROOT_BASE = os.environ.get("DATA_ROOT", "/work/narus/data")


def _resolve_video_rel(
    video_rel: str,
    data_root: str,
    allow_mp4_fallback: bool = False,
) -> tuple[str | None, str]:
    """Resolve a legacy/new video path and prefer frames_atomic when available.

    Supports custom frame directory names (e.g. frames_atomic_320) as well as
    the standard frames_atomic and legacy videos_atomic paths.

    Returns:
        (resolved_rel_or_none, source_tag)
        source_tag in {"frames", "videos", "missing"}
    """
    if not video_rel:
        return None, "missing"

    candidates = []

    if video_rel.startswith("videos_atomic/") and video_rel.endswith(".mp4"):
        # Legacy MP4 path -> prefer frame directory if present.
        frame_rel = video_rel.replace("videos_atomic/", "frames_atomic/", 1)[:-4]
        candidates = [(frame_rel, "frames")]
        if allow_mp4_fallback:
            candidates.append((video_rel, "videos"))
    elif video_rel.startswith("frames_atomic"):
        # Frame-dir path (frames_atomic, frames_atomic_320, etc.) -> check as-is,
        # then optional fallback to legacy mp4.
        candidates = [(video_rel, "frames")]
        if allow_mp4_fallback:
            # Extract suffix after frames_atomic* prefix for mp4 fallback
            rest = video_rel.split("/", 1)[1] if "/" in video_rel else ""
            if rest:
                mp4_rel = os.path.join("videos_atomic", rest) + ".mp4"
                candidates.append((mp4_rel, "videos"))
    else:
        # Unknown format; keep as-is only when mp4 fallback is enabled.
        candidates = [(video_rel, "videos")] if allow_mp4_fallback else []

    for rel, tag in candidates:
        if os.path.exists(os.path.join(data_root, rel)):
            return rel, tag
    return None, "missing"


def _pick_default_intermediate_json(here: str, split: str) -> str:
    """Pick default intermediate JSON path, supporting new and legacy names."""
    candidates = []
    # New naming (current scripts): annotation_atomic_{split}_intermediate.json
    candidates.append(os.path.join(here, f"annotation_atomic_{split}_intermediate.json"))
    # Legacy naming: annotation_atomic_intermediate.json / annotation_atomic_intermediate_val.json
    if split == "train":
        candidates.append(os.path.join(here, "annotation_atomic_intermediate.json"))
    else:
        candidates.append(os.path.join(here, "annotation_atomic_intermediate_val.json"))

    for p in candidates:
        if os.path.isfile(p):
            return p
    # Fallback to new naming even if not found; caller will fail clearly on open().
    return candidates[0]


def build_annotation(
    intermediate_json: str,
    output_path: str,
    data_root: str,
    exo_only: bool = False,
    allow_mp4_fallback: bool = False,
):
    with open(intermediate_json, "r") as f:
        entries = json.load(f)

    annotations = []
    stats = {
        "total_entries": len(entries),
        "ego_matched": 0,
        "exo_matched": 0,
        "no_tok_pose": 0,
        "no_video": 0,
        "video_frames": 0,
        "video_mp4": 0,
    }

    tok_pose_dir_base = os.path.join(data_root, "tok_pose_atomic_40")

    for entry in entries:
        take_name = entry["take_name"]
        sample_id = entry["sample_id"]
        caption = entry["caption"]

        # Find tokenized pose chunks for this sample
        tok_pose_dir = os.path.join(tok_pose_dir_base, take_name)
        tok_pose_files = []
        if os.path.isdir(tok_pose_dir):
            for fname in sorted(os.listdir(tok_pose_dir)):
                if fname.startswith(f"{sample_id}_") and fname.endswith(".npz"):
                    tok_pose_files.append(fname)

        if not tok_pose_files:
            stats["no_tok_pose"] += 1
            continue

        # Create entries for each tok_pose chunk x each view
        for tok_fname in tok_pose_files:
            tok_rel = os.path.join("tok_pose_atomic_40", take_name, tok_fname)

            # Ego view
            if not exo_only and "video_ego" in entry:
                resolved_rel, source_tag = _resolve_video_rel(
                    entry["video_ego"], data_root, allow_mp4_fallback=allow_mp4_fallback
                )
                if resolved_rel is not None:
                    annotations.append({
                        "video": resolved_rel,
                        "tok_pose": tok_rel,
                        "caption": caption,
                    })
                    stats["ego_matched"] += 1
                    if source_tag == "frames":
                        stats["video_frames"] += 1
                    else:
                        stats["video_mp4"] += 1
                else:
                    stats["no_video"] += 1

            # Exo view
            if "video_exo" in entry:
                resolved_rel, source_tag = _resolve_video_rel(
                    entry["video_exo"], data_root, allow_mp4_fallback=allow_mp4_fallback
                )
                if resolved_rel is not None:
                    annotations.append({
                        "video": resolved_rel,
                        "tok_pose": tok_rel,
                        "caption": caption,
                    })
                    stats["exo_matched"] += 1
                    if source_tag == "frames":
                        stats["video_frames"] += 1
                    else:
                        stats["video_mp4"] += 1
                else:
                    stats["no_video"] += 1

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    print(f"Total intermediate entries: {stats['total_entries']}")
    print(f"Ego matched:    {stats['ego_matched']}")
    print(f"Exo matched:    {stats['exo_matched']}")
    print(f"No tok_pose:    {stats['no_tok_pose']}")
    print(f"No video file:  {stats['no_video']}")
    print(f"Video source - frames_atomic: {stats['video_frames']}")
    print(f"Video source - videos_atomic: {stats['video_mp4']}")
    print(f"Total annotations: {len(annotations)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "val"], default="train",
                        help="Dataset split (default: train)")
    parser.add_argument("--intermediate-json", type=str, default=None,
                        help="Path to intermediate JSON from prepare_atomic_clips.py")
    parser.add_argument("--output", type=str, default=None,
                        help="Output annotation JSON path")
    parser.add_argument("--exo-only", action="store_true", default=True,
                        help="Only include exo view, skip ego (default: True)")
    parser.add_argument("--include-ego", action="store_true",
                        help="Also include ego view (overrides --exo-only)")
    parser.add_argument(
        "--allow-mp4-fallback",
        action="store_true",
        help="Allow legacy videos_atomic/*.mp4 path when frames_atomic path is missing (default: False).",
    )
    args = parser.parse_args()

    _here = os.path.dirname(__file__)

    if args.intermediate_json is None:
        args.intermediate_json = _pick_default_intermediate_json(_here, args.split)

    if args.output is None:
        name = f"annotation_atomic_{args.split}.json"
        args.output = os.path.join(_here, name)

    data_root = os.path.join(_DATA_ROOT_BASE, "EgoExo4D/processed", args.split)

    exo_only = args.exo_only and not args.include_ego
    build_annotation(
        args.intermediate_json,
        args.output,
        data_root,
        exo_only=exo_only,
        allow_mp4_fallback=args.allow_mp4_fallback,
    )
