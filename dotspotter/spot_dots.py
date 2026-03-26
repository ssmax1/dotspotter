import argparse
import multiprocessing
import os
from pathlib import Path
import pandas as pd

from tqdm.contrib.concurrent import process_map
from dotspotter.spotter import count_dots
from dotspotter.logging_utils import setup_logger


def run_spot_dots(
    img_dir,
    output_dir,
    save_images,
    output_count_filename,
    dot_size,
    sensitivity,
    preprocess_strength,
    mask_artifacts,
    min_artifact_area,
):
    logger = setup_logger()

    img_dir = Path(img_dir)
    save_path = Path(output_dir)
    output_file_name = output_count_filename

    valid_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

    file_names = sorted(
        f for f in os.listdir(img_dir)
        if Path(f).suffix.lower() in valid_exts
    )

    total = len(file_names)


    total = len(file_names)

    logger.info(f"Found {total} images in {img_dir}")
    logger.info("Starting multiprocessing detection…")

    os.makedirs(save_path, exist_ok=True)

    args_list = [
        (
            f,
            img_dir,
            save_path,
            save_images,
            dot_size,
            sensitivity,
            preprocess_strength,
            mask_artifacts,
            min_artifact_area,
        )
        for f in file_names
    ]

    num_cores = multiprocessing.cpu_count()

    results = process_map(
        count_dots,
        args_list,
        max_workers=num_cores,
        chunksize=10,
        desc="Processing images",
    )

    df = pd.DataFrame([result for result in results if result is not None])
    df.to_csv(save_path / output_file_name, index=False)

    # Final summary
    detected = len(df)
    skipped = total - detected

    logger.info(f"Completed processing {total} images")
    logger.info(f"Detected nuclei in {detected} images")
    if skipped > 0:
        logger.info(f"Skipped {skipped} images (no valid output)")
    logger.info(f"Results saved to: {save_path / output_file_name}")


def main():
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Spot and quantify fluorescent “dot‑like” signals like nuclei.")

    parser.add_argument("--img_dir", required=True, help="Directory containing images.")
    parser.add_argument(
        "--output_dir",
        default=Path(os.getcwd()) / "spot_results",
        help="Directory for spotter output files.",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Save QC images with detected spots and masked areas.",
    )
    parser.add_argument(
        "--output_count_filename",
        default="dot_counts.csv",
        help="Output CSV filename.",
    )
    parser.add_argument(
        "--dot_size",
        type=float,
        default=1.5,
        help="Approximate dot or nucleus radius in pixels.",
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

    run_spot_dots(
        img_dir=args.img_dir,
        output_dir=args.output_dir,
        save_images=args.save_images,
        output_count_filename=args.output_count_filename,
        dot_size=args.dot_size,
        sensitivity=args.sensitivity,
        preprocess_strength=args.preprocess_strength,
        mask_artifacts=not args.no_mask,
        min_artifact_area=args.min_artifact_area,
    )


if __name__ == "__main__":
    main()