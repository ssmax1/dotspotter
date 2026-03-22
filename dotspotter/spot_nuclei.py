import argparse
import multiprocessing
import os
from pathlib import Path
import pandas as pd

from dotspotter.spotter import count_spots
from tqdm.contrib.concurrent import process_map


if __name__ == "__main__":
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Spot and quantify nuclei.")

    parser.add_argument(
        "--img_dir",
        required=True,
        help="Path to the directory containing images.",
    )

    parser.add_argument(
        "--output_dir",
        default=Path(os.getcwd()) / "results",
        help="Save directory for results files.",
    )

    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Include this flag to save images overlaid with detected spots and masked areas for QC / review.",
    )

    parser.add_argument(
        "--output_count_filename",
        default="spot_counts.csv",
        help="Output CSV filename (default: spot_counts.csv)",
    )

    parser.add_argument(
        "--nucleus_size",
        type=float,
        default=1.5,
        help="Approximate nucleus radius in pixels.",
    )

    parser.add_argument(
        "--sensitivity",
        type=float,
        default=1.0,
        help="Detection sensitivity. Higher = more faint nuclei detected.",
    )

    parser.add_argument(
        "--preprocess_strength",
        type=float,
        default=1.0,
        help="Strength of preprocessing enhancement (1–3 recommended).",
    )

    # Masking is ON by default — user must explicitly disable it
    parser.add_argument(
        "--no_mask",
        action="store_true",
        help="Disable artefact masking.",
    )

    parser.add_argument(
        "--min_artifact_area",
        type=int,
        default=5000,
        help="Minimum area (px) for an artefact to be masked.",
    )

    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    output_file_name = args.output_count_filename
    save_path = Path(args.output_dir)

    save_images = args.save_images
    nucleus_size = args.nucleus_size
    sensitivity = args.sensitivity
    preprocess_strength = args.preprocess_strength

    # Masking logic
    mask_artifacts = not args.no_mask
    min_artifact_area = args.min_artifact_area

    file_names = os.listdir(img_dir)
    os.makedirs(save_path, exist_ok=True)

    args_list = [
        (
            f,
            img_dir,
            save_path,
            save_images,
            nucleus_size,
            sensitivity,
            preprocess_strength,
            mask_artifacts,
            min_artifact_area,
        )
        for f in file_names
    ]

    num_cores = multiprocessing.cpu_count()

    results = process_map(
        count_spots,
        args_list,
        max_workers=num_cores,
        chunksize=10,
    )

    df = pd.DataFrame([result for result in results if result is not None])
    df.to_csv(save_path / output_file_name, index=False)