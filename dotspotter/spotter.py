import logging
import cv2
import numpy as np
from skimage.feature import blob_log
from skimage.filters import gaussian
from skimage.exposure import equalize_adapthist, rescale_intensity

VALID_FILE_EXTENSIONS = [".png", ".jpg", ".tif"]

logger = logging.getLogger(__file__)


def count_dots(args):
    (
        file_name,
        img_dir,
        save_path,
        save_images,
        dot_size,
        sensitivity,
        preprocess_strength,
        mask_artifacts,
        min_artifact_area,
    ) = args

    if not any(file_name.endswith(ext) for ext in VALID_FILE_EXTENSIONS):
        return None

    try:
        # --- Load image ---
        img = cv2.imread(str(img_dir / file_name))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        total_pixels = h * w

        # --- Normalise ---
        gray_norm = gray.astype(np.float32) / 255.0

        # ============================================================
        #   OPTIONAL ARTEFACT MASKING (GREY BLOB ONLY)
        # ============================================================

        artifact_mask = np.ones_like(gray_norm, dtype=np.float32)
        masking_applied = False
        masked_pixels = 0

        if mask_artifacts:

            mid_mask = cv2.inRange(gray, 30, 220)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mid_mask)

            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < min_artifact_area:
                    continue

                region_pixels = gray[labels == i].astype(np.float32)
                mean_intensity = region_pixels.mean()
                variance = region_pixels.var()

                bright_pixels = (region_pixels > 230).sum()
                bright_fraction = bright_pixels / float(area)

                if 40 < mean_intensity < 200 and variance < 300 and bright_fraction < 0.001:
                    artifact_mask[labels == i] = 0.0
                    masking_applied = True
                    masked_pixels += area

            gray_norm = gray_norm * artifact_mask

        # % of image masked
        percent_masked = (masked_pixels / total_pixels) * 100.0

        # ============================================================
        #   PREPROCESSING
        # ============================================================

        clip_limit = 0.01 * preprocess_strength
        gray_eq = equalize_adapthist(gray_norm, clip_limit=clip_limit)
        gray_eq = rescale_intensity(gray_eq, in_range=(0, 0.95))
        smooth = gaussian(gray_eq, sigma=0.8)

        # ============================================================
        #   BLOB DETECTION
        # ============================================================

        min_sigma = max(0.5, dot_size * 0.8)
        max_sigma = dot_size * 1.2

        base_threshold = 0.02
        effective_threshold = base_threshold / sensitivity

        blobs = blob_log(
            smooth,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=10,
            threshold=effective_threshold,
        )

        # Remove blobs inside artefact region
        filtered = []
        for y, x, s in blobs:
            if artifact_mask[int(y), int(x)] > 0:
                filtered.append((y, x, s))
        blobs = filtered

        num_objects = len(blobs)

        # ============================================================
        #   ESTIMATED TOTAL COUNT (density extrapolation)
        # ============================================================

        unmasked_pixels = total_pixels - masked_pixels

        if unmasked_pixels > 0:
            density = num_objects / unmasked_pixels
            estimated_total = int(round(density * total_pixels))
        else:
            density = 0
            estimated_total = num_objects

        # ============================================================
        #   QC OVERLAY
        # ============================================================
        if save_images:
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            if masking_applied:
                artefact_pixels = np.where(artifact_mask == 0)
                vis[artefact_pixels] = (0, 0, 255)

            for y, x, s in blobs:
                cv2.circle(vis, (int(x), int(y)), int(2 * s), (0, 255, 0), 1)

            cv2.imwrite(str(save_path / f"spottedQC_{file_name}"), vis)

        return {
            "file_name": file_name,
            "count": num_objects,
            "masking_used": "yes" if masking_applied else "no",
            "percent_masked": percent_masked,
            "estimated_total_count": estimated_total,
            "dot_size": dot_size,
            "sensitivity": sensitivity,
            "preprocess_strength": preprocess_strength,
            "masking_enabled": "yes" if mask_artifacts else "no",
            "min_artifact_area": min_artifact_area,
        }

    except Exception as e:
        logger.error(f"Exception: {e}, {file_name}")
        return None