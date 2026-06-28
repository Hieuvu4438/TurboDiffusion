"""
Extract the best frontal face from a TurboDiffusion restored video.

Since the video typically ends with a frontal view, this script focuses on
the last portion of frames, detects faces using InsightFace, and selects
the best frontal-view face based on detection confidence, face size, pose
estimation (yaw close to 0), and image sharpness.

Supports:
 - Single video mode   : --video_path path/to/video.mp4
 - Batch mode (folder) : --video_dir  path/to/folder_of_mp4s
 - Batch mode (CSV)    : --csv_path   path/to/csv  (uses output video paths)

Usage:
    python scripts/extract_face_from_video.py \
        --video_path output_full/side/blurred3/Amanda_Bynes_side_blurred3.mp4

    python scripts/extract_face_from_video.py \
        --video_dir output_full/side/blurred3/

    python scripts/extract_face_from_video.py \
        --video_dir output_full/ --recursive
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict

# ─── Project root ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "face_recognition_output"

# ─── How much of the video tail to scan ──────────────────────────────────────
DEFAULT_TAIL_RATIO = 0.25       # last 25% of frames
MIN_TAIL_FRAMES = 10            # at least 10 frames
MAX_TAIL_FRAMES = 40            # at most 40 frames (avoid wasting time)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_sharpness(image: np.ndarray) -> float:
    """Laplacian variance – higher means sharper."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def estimate_yaw(kps: np.ndarray) -> float:
    """
    Rough yaw estimation from 5-point landmarks.
    kps: shape (5, 2) – left_eye, right_eye, nose, left_mouth, right_mouth
    Returns absolute yaw angle in degrees (0 = perfect frontal).
    """
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    eye_width = abs(right_eye[0] - left_eye[0])
    if eye_width < 1e-6:
        return 90.0
    # nose offset from eye center, normalized
    nose_offset = (nose[0] - eye_center_x) / eye_width
    # approximate: offset of 0 → frontal, offset of ±0.5 → ~90°
    yaw_deg = abs(nose_offset) * 180.0
    return min(yaw_deg, 90.0)


def extract_tail_frames(video_path: str,
                        tail_ratio: float = DEFAULT_TAIL_RATIO) -> List[Tuple[int, np.ndarray]]:
    """
    Read frames from the tail portion of a video.
    Returns list of (frame_index, bgr_image).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError(f"Video has 0 frames: {video_path}")

    # Determine how many tail frames to read
    n_tail = int(total_frames * tail_ratio)
    n_tail = max(min(n_tail, MAX_TAIL_FRAMES), min(MIN_TAIL_FRAMES, total_frames))
    start_frame = max(0, total_frames - n_tail)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    idx = start_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append((idx, frame))
        idx += 1

    cap.release()
    return frames


def score_face(face, frame: np.ndarray, frame_idx: int, total_frames: int) -> Dict:
    """
    Score a detected face on multiple criteria.
    Returns a dict with individual scores and a combined score.
    """
    h, w = frame.shape[:2]
    bbox = face.bbox  # [x1, y1, x2, y2]
    face_w = bbox[2] - bbox[0]
    face_h = bbox[3] - bbox[1]

    # 1. Detection confidence
    det_score = float(face.det_score) if hasattr(face, 'det_score') else 0.5

    # 2. Face size (larger = better, normalized by frame area)
    area_ratio = (face_w * face_h) / (w * h)

    # 3. Yaw angle (frontal = 0 → higher score)
    yaw = 90.0
    if hasattr(face, 'kps') and face.kps is not None and len(face.kps) >= 3:
        yaw = estimate_yaw(face.kps)
    frontal_score = max(0.0, 1.0 - yaw / 45.0)  # 0° → 1.0, 45°+ → 0.0

    # 4. Sharpness of face region
    x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
    x2, y2 = min(w, int(bbox[2])), min(h, int(bbox[3]))
    face_crop = frame[y1:y2, x1:x2]
    sharpness = compute_sharpness(face_crop) if face_crop.size > 0 else 0.0

    # 5. Frame position bonus (later frames slightly preferred)
    position_score = frame_idx / total_frames if total_frames > 0 else 0.5

    # Combined score (weighted)
    combined = (
        det_score     * 0.25 +
        area_ratio    * 0.15 +
        frontal_score * 0.35 +
        min(sharpness / 500.0, 1.0) * 0.15 +
        position_score * 0.10
    )

    return {
        "det_score": det_score,
        "area_ratio": area_ratio,
        "yaw": yaw,
        "frontal_score": frontal_score,
        "sharpness": sharpness,
        "position_score": position_score,
        "combined": combined,
        "frame_idx": frame_idx,
        "face": face,
        "bbox": (x1, y1, x2, y2),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Main extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_best_face(video_path: str,
                      face_app,
                      tail_ratio: float = DEFAULT_TAIL_RATIO,
                      save_full_frame: bool = False) -> Optional[Dict]:
    """
    Extract the best frontal face from the tail of a video.

    Returns dict with keys: face_image, full_frame, scores, frame_idx
    or None if no face detected.
    """
    video_path = Path(video_path)

    # 1. Read tail frames
    frames = extract_tail_frames(str(video_path), tail_ratio)
    if not frames:
        print(f"    ⚠ No frames extracted from {video_path.name}")
        return None

    total_frames_in_video = frames[-1][0] + 1  # approximate

    # 2. Detect faces in each frame and score them
    all_candidates = []

    for frame_idx, frame in frames:
        # InsightFace expects BGR
        faces = face_app.get(frame)
        if not faces:
            continue

        # Pick the best face in this frame (largest + centered)
        best_in_frame = faces[0]
        best_area = 0
        for f in faces:
            bx = f.bbox
            area = (bx[2] - bx[0]) * (bx[3] - bx[1])
            if area > best_area:
                best_area = area
                best_in_frame = f

        scored = score_face(best_in_frame, frame, frame_idx, total_frames_in_video)
        scored["frame"] = frame
        all_candidates.append(scored)

    if not all_candidates:
        print(f"    ⚠ No faces detected in last {len(frames)} frames of {video_path.name}")
        return None

    # 3. Select the best candidate
    best = max(all_candidates, key=lambda c: c["combined"])

    # 4. Crop the face
    x1, y1, x2, y2 = best["bbox"]
    frame = best["frame"]

    # Add some padding around the face (20%)
    face_w = x2 - x1
    face_h = y2 - y1
    pad_x = int(face_w * 0.2)
    pad_y = int(face_h * 0.2)
    h, w = frame.shape[:2]
    px1 = max(0, x1 - pad_x)
    py1 = max(0, y1 - pad_y)
    px2 = min(w, x2 + pad_x)
    py2 = min(h, y2 + pad_y)
    face_image = frame[py1:py2, px1:px2]

    return {
        "face_image": face_image,
        "full_frame": frame if save_full_frame else None,
        "frame_idx": best["frame_idx"],
        "scores": {k: v for k, v in best.items() if k not in ("face", "frame", "bbox")},
        "video_name": video_path.stem,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Processing functions
# ─────────────────────────────────────────────────────────────────────────────

def init_face_app(det_size: Tuple[int, int] = (640, 640),
                  det_thresh: float = 0.5,
                  use_gpu: bool = False):
    """Initialize InsightFace FaceAnalysis."""
    from insightface.app import FaceAnalysis

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
    app = FaceAnalysis(
        name='buffalo_l',
        providers=providers,
        allowed_modules=['detection', 'recognition']
    )
    app.prepare(ctx_id=0 if use_gpu else -1, det_size=det_size, det_thresh=det_thresh)
    return app


def process_single_video(video_path: str, output_dir: str, face_app,
                         tail_ratio: float = DEFAULT_TAIL_RATIO,
                         save_full_frame: bool = False) -> bool:
    """Process a single video and save the extracted face."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output filename: same as video but .jpg
    out_name = video_path.stem + "_face.jpg"
    out_path = output_dir / out_name

    result = extract_best_face(str(video_path), face_app, tail_ratio, save_full_frame)

    if result is None:
        print(f"    ✗ No face found: {video_path.name}")
        return False

    # Save face image
    cv2.imwrite(str(out_path), result["face_image"])

    scores = result["scores"]
    print(f"    ✓ {out_name}  "
          f"(frame={result['frame_idx']}, "
          f"yaw={scores['yaw']:.1f}°, "
          f"sharp={scores['sharpness']:.0f}, "
          f"det={scores['det_score']:.3f}, "
          f"score={scores['combined']:.3f})")

    # Optionally save full frame
    if save_full_frame and result["full_frame"] is not None:
        frame_name = video_path.stem + "_fullframe.jpg"
        cv2.imwrite(str(output_dir / frame_name), result["full_frame"])

    return True


def process_video_dir(video_dir: str, output_dir: str, face_app,
                      recursive: bool = False,
                      tail_ratio: float = DEFAULT_TAIL_RATIO,
                      save_full_frame: bool = False):
    """Process all .mp4 files in a directory."""
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)

    if recursive:
        video_files = sorted(video_dir.rglob("*.mp4"))
    else:
        video_files = sorted(video_dir.glob("*.mp4"))

    if not video_files:
        print(f"ERROR: No .mp4 files found in {video_dir}")
        return

    print(f"Found {len(video_files)} video files\n")

    success = 0
    failed = 0

    for i, vf in enumerate(video_files, 1):
        # Mirror directory structure if recursive
        if recursive:
            rel = vf.parent.relative_to(video_dir)
            out_sub = output_dir / rel
        else:
            out_sub = output_dir

        print(f"  [{i}/{len(video_files)}] {vf.name}")
        if process_single_video(str(vf), str(out_sub), face_app, tail_ratio, save_full_frame):
            success += 1
        else:
            failed += 1

    print(f"\nDone: {success} succeeded, {failed} failed out of {len(video_files)}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract the best frontal face from TurboDiffusion restored video(s)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Single video
  python scripts/extract_face_from_video.py \\
      --video_path output_full/side/blurred3/Amanda_Bynes_side_blurred3.mp4

  # All videos in a folder
  python scripts/extract_face_from_video.py \\
      --video_dir output_full/side/blurred3/

  # All videos recursively (mirrors directory structure)
  python scripts/extract_face_from_video.py \\
      --video_dir output_full/ --recursive

  # Custom output directory
  python scripts/extract_face_from_video.py \\
      --video_dir output_full/side/blurred3/ \\
      --output_dir my_faces/
""",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video_path", type=str, help="Path to a single video")
    input_group.add_argument("--video_dir", type=str, help="Directory containing .mp4 videos")

    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--recursive", action="store_true",
                        help="Search for videos recursively (mirrors subdir structure)")
    parser.add_argument("--tail_ratio", type=float, default=DEFAULT_TAIL_RATIO,
                        help=f"Fraction of video tail to scan (default: {DEFAULT_TAIL_RATIO})")
    parser.add_argument("--save_full_frame", action="store_true",
                        help="Also save the full frame (not just cropped face)")
    parser.add_argument("--det_thresh", type=float, default=0.5,
                        help="Face detection confidence threshold (default: 0.5)")
    parser.add_argument("--use_gpu", action="store_true",
                        help="Use GPU for InsightFace (default: CPU)")

    args = parser.parse_args()

    # Initialize InsightFace
    print("Loading InsightFace model...")
    face_app = init_face_app(det_thresh=args.det_thresh, use_gpu=args.use_gpu)
    print("InsightFace ready!\n")

    if args.video_path:
        # Single video mode
        vp = Path(args.video_path)
        if not vp.exists():
            print(f"ERROR: Video not found: {vp}")
            sys.exit(1)

        print(f"Processing: {vp.name}")
        process_single_video(str(vp), args.output_dir, face_app,
                             args.tail_ratio, args.save_full_frame)

    elif args.video_dir:
        # Directory mode
        vd = Path(args.video_dir)
        if not vd.is_dir():
            print(f"ERROR: Not a directory: {vd}")
            sys.exit(1)

        print(f"Video directory: {vd}")
        print(f"Output directory: {args.output_dir}")
        print(f"Recursive: {args.recursive}\n")

        process_video_dir(str(vd), args.output_dir, face_app,
                          args.recursive, args.tail_ratio, args.save_full_frame)


if __name__ == "__main__":
    main()
