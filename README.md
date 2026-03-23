# dotspotter

**dotspotter** is a tool for detecting and counting fluorescent “dot‑like” signals in microscopy images. It is designed for high‑throughput workflows where nuclei or puncta appear as small, faint peaks rather than well‑defined multi‑pixel objects. The tool uses a Laplacian‑of‑Gaussian (LoG) scale‑space detector with tunable preprocessing, sensitivity, and nucleus‑size parameters, making it suitable for DAPI, Hoechst, and other nuclear stains, as well as general fluorescence spot detection.

dotspotter also includes an artefact‑masking system that automatically removes large artifacts that may distort counts. Masking is **enabled by default** and can be disabled with `--no_mask`.

---

## Features

- Fast, parallelised batch processing using all CPU cores  
- Robust detection of faint, tiny nuclei or puncta  
- Adjustable nucleus size, sensitivity, and preprocessing strength  
- Automatic masking of large smooth artefacts (default ON)  
- Optional saving of marked images for QC  
- CSV output summarising per‑image counts  
- Additional metrics:  
  - **percent_masked** — % of image removed due to artefact  
  - **estimated_total_count** — density‑based extrapolated count  

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/ssmax1/dotspotter.git
cd dotspotter
```

### 2. Create and activate a Python environment

Using a dedicated environment avoids dependency conflicts and keeps workflows reproducible.

**Using venv (built‑in Python):**
```bash
python3 -m venv dotspotter-env
source dotspotter-env/bin/activate   # Linux / macOS
# OR
dotspotter-env\Scripts\activate      # Windows
```

### 3. Install dotspotter & dependencies

```bash
pip3 install .
```

---


## Usage

### Basic usage  
```bash
python spot_nuclei.py \  
    --img_dir {image folder path}/ \  
    --output_dir {results folder path}/ \  
    --save_images
```
### With tuning parameters  
```bash
python spot_nuclei.py \  
    --img_dir images/ \  
    --output_dir results/ \  
    --save_images \  
    --nucleus_size 1.9 \  
    --sensitivity 1.8 \  
    --preprocess_strength 2.0
```
### Disable artefact masking  
```bash
python spot_nuclei.py \  
    --img_dir images/ \  
    --output_dir results/ \  
    --no_mask
```
---

## Command‑line Options

| Parameter | Type | Default | Description |
|----------|------|---------|-------------|
| `--img_dir` | path | required | Directory containing input images. Only `.png`, `.jpg`, and `.tif` files are processed. |
| `--output_dir` | path | `./results` | Directory where the CSV file and optional marked images are saved. |
| `--output_count_filename` | str | `spotted_counts.csv` | Name of the output CSV file. |
| `--save_images` | flag | off | Saves QC images with detected spots overlaid in green. |
| `--dot_size` | float | 1.5 | Approximate radius (px) of expected nuclei or puncta. |
| `--sensitivity` | float | 1.0 | Controls detection aggressiveness. Higher values detect fainter spots. |
| `--preprocess_strength` | float | 1.0 | Strength of adaptive histogram equalisation and highlight compression. |
| `--no_mask` | flag | off | Disable artefact masking. Masking is ON by default. |
| `--min_artifact_area` | int | 5000 | Minimum connected‑component area (px) to be considered an artefact. |

---

## How dotspotter Works

### 1. Image loading and normalisation  
Images are loaded with OpenCV, converted to grayscale, and normalised to 0–1.

### 2. Artefact masking (default ON)  
Large, smooth, mid‑grey features are detected using:  
- mid‑intensity thresholding  
- connected components  
- texture (variance) filtering  
- bright‑pixel density filtering  

Masked regions are excluded from detection and highlighted in **red** in QC images.

### 3. Preprocessing  
The preprocessing pipeline includes:  
- adaptive histogram equalisation  
- highlight compression  
- light Gaussian smoothing  

### 4. Scale‑space LoG detection  
The LoG detector searches across a **narrow sigma band** tuned to the expected nucleus size:  
min_sigma = max(0.5, nucleus_size × 0.8)  
max_sigma = nucleus_size × 1.2  

### 5. Sensitivity control  
Higher sensitivity lowers the LoG threshold:  
effective_threshold = base_threshold / sensitivity

### 6. QC image generation  
If `--save_images` is enabled:  
- masked artefacts are shown in **red**  
- detected nuclei are shown in **green**

### 7. Parallel processing  
All images are processed in parallel using `process_map`.

### 8. Output  
The CSV includes:  
- file_name  
- count  
- masking_used  
- percent_masked  
- estimated_total_count  

---

## Output Files

dotspotter produces:  
- A CSV file summarising counts and masking metrics  
- Optional QC images with detected spots and masked regions  

Example CSV:  
file_name,count,masking_used,percent_masked,estimated_total_count  
img001.png,142,yes,12.4,162  
img002.png,158,no,0.0,158

---

## Recommended Parameter Ranges

| Scenario | Suggested Settings |
|---------|--------------------|
| Very faint nuclei | sensitivity 2.0–3.0, preprocess_strength 2.0–3.0 |
| Very small puncta | nucleus_size 1.0–1.5 |
| Larger nuclei | nucleus_size 2.0–3.0 |
| High‑contrast images | preprocess_strength 1.0 |

---

## License

MIT

---

## Author

Scott Maxwell