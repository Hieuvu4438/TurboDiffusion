"""
Run TurboDiffusion Wan2.2 I2V on random side-pose samples from
`gaussian_blur_ref_side.csv`.

 - Chọn ngẫu nhiên N (mặc định: 3) dòng trong CSV.
 - Dùng cột `side_deg` (ảnh side pose đã Gaussian blur) làm input cho I2V.
 - Gọi script `turbodiffusion/inference/wan2.2_i2v_infer.py` giống mẫu trong `run.txt`.
 - Lưu video output vào thư mục `output/i2v_gaussian/`.
"""

import os
import argparse
import subprocess
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CSV_PATH = PROJECT_ROOT / "degradation_experiment" / "output_ref_side" / "csv" / "gaussian_blur_ref_side.csv"


def run_i2v_for_sample(row, idx: int, dry_run: bool = False):
    """
    Run TurboDiffusion I2V for a single CSV row (one person / one blur setting).
    """
    person_name = row["person_name"]
    side_deg = row["side_deg"]
    sigma = row["sigma"]
    k = row["kernel_size"]

    # Đường dẫn ảnh input (side pose sau Gaussian blur)
    image_path = side_deg

    # Thư mục & tên file output
    output_dir = PROJECT_ROOT / "output" / "i2v_gaussian"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{person_name}_sigma{sigma}_k{k}_sample{idx+1}.mp4"

    # Prompt "đẹp" mặc định (bạn có thể sửa lại trong file này nếu muốn)
    prompt = (
        "A high-quality portrait video strictly preserving the identity of the person and original biological gender of the subject in the source image. The person smoothly turns their head from a side-profile to a full frontal view, ending with them looking directly into the camera lens. Maintain consistent facial structure throughout the rotation."
    )

    cmd = [
        "python",
        "turbodiffusion/inference/wan2.2_i2v_infer.py",
        "--model",
        "Wan2.2-A14B",
        "--low_noise_model_path",
        "checkpoints/TurboWan2.2-I2V-A14B-low-720P-quant.pth",
        "--high_noise_model_path",
        "checkpoints/TurboWan2.2-I2V-A14B-high-720P-quant.pth",
        "--resolution",
        "720p",
        "--adaptive_resolution",
        "--image_path",
        str(image_path),
        "--prompt",
        prompt,
        "--num_samples",
        "1",
        "--num_steps",
        "4",
        "--quant_linear",
        "--attention_type",
        "sagesla",
        "--sla_topk",
        "0.1",
        "--ode",
        "--save_path",
        str(save_path),
    ]

    print(f"\n=== Sample {idx+1} ===")
    print(f"person_name : {person_name}")
    print(f"side_deg    : {image_path}")
    print(f"sigma, k    : {sigma}, {k}")
    print(f"save_path   : {save_path}")
    print("Command:")
    print(" ".join(str(c) for c in cmd))

    if dry_run:
        return

    # Chạy lệnh
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="Run TurboDiffusion Wan2.2 I2V on random side-pose samples from gaussian_blur_ref_side.csv"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=str(DEFAULT_CSV_PATH),
        help=f"Path to gaussian_blur_ref_side.csv (default: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of random samples to run (default: 3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print selected samples and commands, do not run TurboDiffusion",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError(f"CSV is empty: {csv_path}")

    # Lấy ngẫu nhiên N dòng
    n = min(args.num_samples, len(df))
    samples = df.sample(n=n, random_state=None)

    print(f"Loaded CSV: {csv_path}")
    print(f"Total rows: {len(df)}, sampling: {n}")

    for idx, (_, row) in enumerate(samples.iterrows()):
        run_i2v_for_sample(row, idx, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

