# MVAA 2026 Challenge — Complete AI Agent Instructions

> **Competition:** Mitral Valve Anatomy Analysis Using Multimodal Imaging Data (MVAA 2026)
> **Platform:** CodaBench — https://www.codabench.org/competitions/15662/
> **Baseline Repo:** https://github.com/db0725/MVAA
> **Conda Environment (MANDATORY):** `megha_env`
> **All work MUST be done inside:** `~/megha_workspace/` (created in Step 0)

---

## CRITICAL RULES BEFORE ANYTHING ELSE

1. **NEVER install any package outside of the `megha_env` conda environment.** Always activate it first.
2. **NEVER place any file outside of `~/megha_workspace/`.** All data, code, outputs, checkpoints, and submissions go here.
3. **READ every source file before modifying it.** Do not assume what a file contains.
4. **DO NOT skip the analysis phase.** Every dataset folder must be inspected and reported before training begins.
5. **ALL Python execution must be done with the Python binary from `megha_env`**, which is at:
   `$(conda run -n megha_env which python)`
   or after activation: `conda activate megha_env && python ...`
6. **Submission day limit:** CodaBench allows a maximum of **2 submissions per day** during the validation phase. Plan accordingly.
7. If any step fails, **stop, log the error to `~/megha_workspace/logs/agent_errors.log`**, diagnose, fix, and retry. Do not continue past a broken step.

---

## WORKSPACE LAYOUT (Enforced)

```
~/megha_workspace/
├── MVAA/                         ← git cloned baseline repo
│   ├── task1/
│   │   ├── train.py
│   │   ├── generate_task1_predictions.py
│   │   ├── dataset.py
│   │   ├── model_factory.py
│   │   └── utils.py
│   ├── task2/
│   │   ├── train.py
│   │   ├── generate_task2_predictions.py
│   │   ├── dataset.py
│   │   ├── model_factory.py
│   │   └── utils.py
│   ├── task3/
│   │   ├── train.py
│   │   ├── generate_task3_predictions.py
│   │   ├── dataset.py
│   │   ├── model_factory.py
│   │   └── utils.py
│   ├── requirements.txt
│   ├── submission.zip            ← reference submission format
│   └── README.md
├── data/
│   ├── t1_ct/
│   │   ├── train/
│   │   │   ├── images/           ← 1067 total: 27 labeled + 1040 unlabeled
│   │   │   └── labels/           ← 27 .nii.gz label files (label=1: foreground)
│   │   └── val/
│   │       └── images/           ← 30 .nii.gz files WITHOUT labels
│   ├── t2_tee/
│   │   ├── train/
│   │   │   ├── images/           ← 105 .nii.gz ultrasound volumes
│   │   │   └── labels/           ← 105 .nii.gz label files (labels 1 and 2)
│   │   └── val/
│   │       └── images/           ← 20 .nii.gz files WITHOUT labels
│   └── t3_vid/
│       ├── train/
│       │   ├── images/           ← 1619 frames total (240 labeled, 1379 unlabeled)
│       │   └── labels/           ← 240 binary PNGs or tar files (*_png_Label.tar)
│       └── val/
│           └── images/           ← 48 frames WITHOUT labels (all contain MV)
├── runs/
│   ├── task1/                    ← Task 1 training outputs
│   ├── task2/                    ← Task 2 training outputs
│   └── task3/                    ← Task 3 training outputs
├── submission/
│   ├── t1_ct/
│   │   ├── task1_predictions.json
│   │   └── *.nii.gz
│   ├── t2_tee/
│   │   ├── task2_predictions.json
│   │   └── *.nii.gz
│   └── t3_vid/
│       ├── task3_predictions.json
│       └── <video_folder_N>/
│           └── *_label_bin.png
├── logs/
│   └── agent_errors.log
└── analysis_reports/
    ├── task1_data_analysis.txt
    ├── task2_data_analysis.txt
    └── task3_data_analysis.txt
```

---

## STEP 0 — Create Workspace and Verify Conda Environment

### 0.1 — Verify `megha_env` exists

```bash
conda env list
```

**Expected output must include a line like:**
```
megha_env    /home/<user>/miniconda3/envs/megha_env
```

**If `megha_env` does NOT appear in the list, stop immediately and report:**
```
ERROR: conda environment 'megha_env' does not exist.
Please create it with: conda create -n megha_env python=3.9 -y
Then re-run from Step 0.
```

Do NOT proceed until `megha_env` is confirmed to exist.

### 0.2 — Create workspace directory

```bash
mkdir -p ~/megha_workspace/data
mkdir -p ~/megha_workspace/runs/task1
mkdir -p ~/megha_workspace/runs/task2
mkdir -p ~/megha_workspace/runs/task3
mkdir -p ~/megha_workspace/submission/t1_ct
mkdir -p ~/megha_workspace/submission/t2_tee
mkdir -p ~/megha_workspace/submission/t3_vid
mkdir -p ~/megha_workspace/logs
mkdir -p ~/megha_workspace/analysis_reports
```

### 0.3 — Verify Python version in megha_env

```bash
conda run -n megha_env python --version
```

**Expected:** Python 3.9.x or 3.10.x or 3.11.x
**If Python version is below 3.9**, stop and report. Python 3.9+ is required.

### 0.4 — Verify CUDA availability (optional but important for training speed)

```bash
conda run -n megha_env python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count())"
```

Log the output. If CUDA is not available, training will run on CPU (much slower). Note this in the logs and continue — the code supports CPU fallback.

---

## STEP 1 — Clone the Baseline Repository

### 1.1 — Clone into workspace

```bash
cd ~/megha_workspace
git clone https://github.com/db0725/MVAA.git
```

**Verify the clone succeeded:**
```bash
ls ~/megha_workspace/MVAA/
```

**Expected output must include:**
```
task1/  task2/  task3/  requirements.txt  submission.zip  README.md  .gitignore
```

**If any of these are missing**, re-run the clone command. If git is not installed, run:
```bash
conda run -n megha_env conda install -c conda-forge git -y
```
or
```bash
sudo apt-get install git -y
```

### 1.2 — Inspect the reference submission.zip immediately

```bash
unzip -l ~/megha_workspace/MVAA/submission.zip
```

Study and log the exact directory structure shown. This is the gold-standard format your final submission must match. Copy this output to `~/megha_workspace/analysis_reports/reference_submission_structure.txt`:

```bash
unzip -l ~/megha_workspace/MVAA/submission.zip > ~/megha_workspace/analysis_reports/reference_submission_structure.txt
cat ~/megha_workspace/analysis_reports/reference_submission_structure.txt
```

### 1.3 — Read ALL source files before proceeding

The agent MUST read (print/cat) every Python file in the repository BEFORE making any modifications or running any training. This is non-negotiable.

```bash
# Read all Python files in task1
echo "====== task1/train.py ======" && cat ~/megha_workspace/MVAA/task1/train.py
echo "====== task1/dataset.py ======" && cat ~/megha_workspace/MVAA/task1/dataset.py
echo "====== task1/model_factory.py ======" && cat ~/megha_workspace/MVAA/task1/model_factory.py
echo "====== task1/utils.py ======" && cat ~/megha_workspace/MVAA/task1/utils.py
echo "====== task1/generate_task1_predictions.py ======" && cat ~/megha_workspace/MVAA/task1/generate_task1_predictions.py

# Read all Python files in task2
echo "====== task2/train.py ======" && cat ~/megha_workspace/MVAA/task2/train.py
echo "====== task2/dataset.py ======" && cat ~/megha_workspace/MVAA/task2/dataset.py
echo "====== task2/model_factory.py ======" && cat ~/megha_workspace/MVAA/task2/model_factory.py
echo "====== task2/utils.py ======" && cat ~/megha_workspace/MVAA/task2/utils.py
echo "====== task2/generate_task2_predictions.py ======" && cat ~/megha_workspace/MVAA/task2/generate_task2_predictions.py

# Read all Python files in task3
echo "====== task3/train.py ======" && cat ~/megha_workspace/MVAA/task3/train.py
echo "====== task3/dataset.py ======" && cat ~/megha_workspace/MVAA/task3/dataset.py
echo "====== task3/model_factory.py ======" && cat ~/megha_workspace/MVAA/task3/model_factory.py
echo "====== task3/utils.py ======" && cat ~/megha_workspace/MVAA/task3/utils.py
echo "====== task3/generate_task3_predictions.py ======" && cat ~/megha_workspace/MVAA/task3/generate_task3_predictions.py

# Read requirements
cat ~/megha_workspace/MVAA/requirements.txt
```

Save the complete output to a log file:
```bash
(
  cat ~/megha_workspace/MVAA/task1/train.py
  cat ~/megha_workspace/MVAA/task1/dataset.py
  cat ~/megha_workspace/MVAA/task1/model_factory.py
  cat ~/megha_workspace/MVAA/task1/utils.py
  cat ~/megha_workspace/MVAA/task1/generate_task1_predictions.py
  cat ~/megha_workspace/MVAA/task2/train.py
  cat ~/megha_workspace/MVAA/task2/dataset.py
  cat ~/megha_workspace/MVAA/task2/model_factory.py
  cat ~/megha_workspace/MVAA/task2/utils.py
  cat ~/megha_workspace/MVAA/task2/generate_task2_predictions.py
  cat ~/megha_workspace/MVAA/task3/train.py
  cat ~/megha_workspace/MVAA/task3/dataset.py
  cat ~/megha_workspace/MVAA/task3/model_factory.py
  cat ~/megha_workspace/MVAA/task3/utils.py
  cat ~/megha_workspace/MVAA/task3/generate_task3_predictions.py
) > ~/megha_workspace/analysis_reports/all_source_code.txt
```

After reading, identify and note:
- What default CLI arguments each `train.py` accepts
- What path constants exist in each `generate_*_predictions.py` that must be edited
- What data format each `dataset.py` expects (file extensions, folder structure)
- What model architecture each `model_factory.py` defines
- Any hardcoded paths that may cause failures

---

## STEP 2 — Install All Dependencies into megha_env

The `requirements.txt` specifies: `torch`, `monai`, `numpy`, `pillow`, `nibabel`, `segmentation-models-pytorch`, `timm`

All installations happen **exclusively inside `megha_env`**.

### 2.1 — Install PyTorch (with CUDA if available)

First check your CUDA version:
```bash
nvidia-smi | grep "CUDA Version"
```

**If CUDA 11.8:**
```bash
conda run -n megha_env pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**If CUDA 12.1 or 12.x:**
```bash
conda run -n megha_env pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**If no CUDA / CPU only:**
```bash
conda run -n megha_env pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 2.2 — Install all remaining requirements

```bash
conda run -n megha_env pip install \
    monai \
    numpy \
    pillow \
    nibabel \
    segmentation-models-pytorch \
    timm \
    scipy \
    scikit-image \
    tqdm \
    pandas \
    matplotlib \
    SimpleITK \
    connected-components-3d
```

### 2.3 — Verify all packages installed correctly

```bash
conda run -n megha_env python -c "
import torch; print('torch:', torch.__version__)
import monai; print('monai:', monai.__version__)
import numpy; print('numpy:', numpy.__version__)
import PIL; print('pillow:', PIL.__version__)
import nibabel; print('nibabel:', nibabel.__version__)
import segmentation_models_pytorch as smp; print('smp:', smp.__version__)
import timm; print('timm:', timm.__version__)
import scipy; print('scipy:', scipy.__version__)
import skimage; print('scikit-image:', skimage.__version__)
import SimpleITK; print('SimpleITK:', SimpleITK.Version.VersionString())
print('All packages verified successfully.')
"
```

**If any import fails**, install the missing package specifically:
```bash
conda run -n megha_env pip install <package_name>
```
Re-run verification until all packages import successfully. Do not proceed to Step 3 until verification passes.

---

## STEP 3 — Download the Dataset

The dataset is split across two Google Drive links. Both must be downloaded to get the complete dataset.

### 3.1 — Install gdown for Google Drive downloads

```bash
conda run -n megha_env pip install gdown
```

Verify:
```bash
conda run -n megha_env gdown --version
```

### 3.2 — Download both dataset archives

```bash
cd ~/megha_workspace/data

# File 1
conda run -n megha_env gdown "https://drive.google.com/file/d/14WneBUBZ1X4p69tRdRzximNb0IsWuh2B/view?usp=sharing" -O mvaa_data_part1.zip --fuzzy

# File 2
conda run -n megha_env gdown "https://drive.google.com/file/d/1K4vFo-4pdL1_xt-8msO-32bc2aEdJ7wS/view?usp=sharing" -O mvaa_data_part2.zip --fuzzy
```

**If gdown fails** (e.g., quota exceeded), try:
```bash
conda run -n megha_env pip install "gdown>=4.6.0"
conda run -n megha_env gdown --id 14WneBUBZ1X4p69tRdRzximNb0IsWuh2B -O mvaa_data_part1.zip
conda run -n megha_env gdown --id 1K4vFo-4pdL1_xt-8msO-32bc2aEdJ7wS -O mvaa_data_part2.zip
```

Verify both files downloaded and are non-zero size:
```bash
ls -lh ~/megha_workspace/data/mvaa_data_part1.zip
ls -lh ~/megha_workspace/data/mvaa_data_part2.zip
```

Both files must be greater than 0 bytes. If either is 0 bytes or download failed, retry.

### 3.3 — Extract the archives

```bash
cd ~/megha_workspace/data

# Determine the archive format first (could be .zip or other)
file mvaa_data_part1.zip
file mvaa_data_part2.zip

# If .zip format:
unzip -o mvaa_data_part1.zip -d ~/megha_workspace/data/
unzip -o mvaa_data_part2.zip -d ~/megha_workspace/data/

# If .tar.gz format instead:
# tar -xzf mvaa_data_part1.zip -C ~/megha_workspace/data/
# tar -xzf mvaa_data_part2.zip -C ~/megha_workspace/data/
```

### 3.4 — Locate and map the extracted folders

After extraction, list everything:
```bash
find ~/megha_workspace/data/ -maxdepth 3 -type d | sort
```

Identify which folders correspond to:
- Task 1 CT data (NIfTI volumes, `.nii.gz`)
- Task 2 TEE data (NIfTI volumes, `.nii.gz`)
- Task 3 Surgical video frames (2D image files, `.png` or `.jpg`)

**Then create the canonical folder structure** by moving or symlinking:
```bash
# Adjust <actual_extracted_folder_name> based on what find shows
# For example:
mv ~/megha_workspace/data/<actual_task1_folder>  ~/megha_workspace/data/t1_ct
mv ~/megha_workspace/data/<actual_task2_folder>  ~/megha_workspace/data/t2_tee
mv ~/megha_workspace/data/<actual_task3_folder>  ~/megha_workspace/data/t3_vid
```

If the extracted structure already matches `t1_ct/`, `t2_tee/`, `t3_vid/`, just verify:
```bash
ls ~/megha_workspace/data/t1_ct/
ls ~/megha_workspace/data/t2_tee/
ls ~/megha_workspace/data/t3_vid/
```

---

## STEP 4 — Data Analysis (MANDATORY — Do NOT Skip)

Before any training, the agent must perform a thorough analysis of the dataset. This prevents failures from unexpected data formats.

### 4.1 — Task 1 (CT) Analysis

```bash
conda run -n megha_env python - << 'EOF'
import os, nibabel as nib, numpy as np

t1_root = os.path.expanduser("~/megha_workspace/data/t1_ct")
report_path = os.path.expanduser("~/megha_workspace/analysis_reports/task1_data_analysis.txt")

lines = []
lines.append("=== TASK 1 CT DATA ANALYSIS ===\n")

# Check folder structure
for split in ["train", "val"]:
    for subfolder in ["images", "labels"]:
        p = os.path.join(t1_root, split, subfolder)
        if os.path.exists(p):
            files = sorted(os.listdir(p))
            lines.append(f"\n[{split}/{subfolder}] Found {len(files)} files")
            lines.append(f"  Extensions: {set(os.path.splitext(f)[1] for f in files)}")
            lines.append(f"  First 5 files: {files[:5]}")
            # Inspect first file in detail
            if files:
                fpath = os.path.join(p, files[0])
                try:
                    img = nib.load(fpath)
                    data = img.get_fdata()
                    lines.append(f"  Shape of first file ({files[0]}): {data.shape}")
                    lines.append(f"  Dtype: {data.dtype}")
                    lines.append(f"  Voxel spacing (zooms): {img.header.get_zooms()}")
                    lines.append(f"  Value range: min={data.min():.4f}, max={data.max():.4f}")
                    if "label" in subfolder:
                        unique_vals = np.unique(data)
                        lines.append(f"  Unique label values: {unique_vals}")
                except Exception as e:
                    lines.append(f"  ERROR reading file: {e}")
        else:
            lines.append(f"\n[{split}/{subfolder}] FOLDER DOES NOT EXIST: {p}")

report = "\n".join(lines)
print(report)
with open(report_path, "w") as f:
    f.write(report)
print(f"\nReport saved to {report_path}")
EOF
```

### 4.2 — Task 2 (TEE) Analysis

```bash
conda run -n megha_env python - << 'EOF'
import os, nibabel as nib, numpy as np

t2_root = os.path.expanduser("~/megha_workspace/data/t2_tee")
report_path = os.path.expanduser("~/megha_workspace/analysis_reports/task2_data_analysis.txt")

lines = []
lines.append("=== TASK 2 TEE DATA ANALYSIS ===\n")

for split in ["train", "val"]:
    for subfolder in ["images", "labels"]:
        p = os.path.join(t2_root, split, subfolder)
        if os.path.exists(p):
            files = sorted(os.listdir(p))
            lines.append(f"\n[{split}/{subfolder}] Found {len(files)} files")
            lines.append(f"  Extensions: {set(os.path.splitext(f)[1] for f in files)}")
            lines.append(f"  First 5 files: {files[:5]}")
            if files:
                fpath = os.path.join(p, files[0])
                try:
                    img = nib.load(fpath)
                    data = img.get_fdata()
                    lines.append(f"  Shape of first file ({files[0]}): {data.shape}")
                    lines.append(f"  Dtype: {data.dtype}")
                    lines.append(f"  Voxel spacing (zooms): {img.header.get_zooms()}")
                    lines.append(f"  Value range: min={data.min():.4f}, max={data.max():.4f}")
                    if "label" in subfolder:
                        unique_vals = np.unique(data)
                        lines.append(f"  Unique label values: {unique_vals}")
                        lines.append(f"  NOTE: Expected labels 1 and 2 (multi-class TEE task)")
                except Exception as e:
                    lines.append(f"  ERROR reading file: {e}")
        else:
            lines.append(f"\n[{split}/{subfolder}] FOLDER DOES NOT EXIST: {p}")

report = "\n".join(lines)
print(report)
with open(report_path, "w") as f:
    f.write(report)
print(f"\nReport saved to {report_path}")
EOF
```

### 4.3 — Task 3 (Surgical Video Frames) Analysis

```bash
conda run -n megha_env python - << 'EOF'
import os
from PIL import Image
import numpy as np

t3_root = os.path.expanduser("~/megha_workspace/data/t3_vid")
report_path = os.path.expanduser("~/megha_workspace/analysis_reports/task3_data_analysis.txt")

lines = []
lines.append("=== TASK 3 SURGICAL VIDEO FRAMES DATA ANALYSIS ===\n")

for split in ["train", "val"]:
    for subfolder in ["images", "labels"]:
        p = os.path.join(t3_root, split, subfolder)
        if os.path.exists(p):
            # Walk entire subtree — may have subfolders per video
            all_files = []
            for root, dirs, files in os.walk(p):
                for fname in files:
                    all_files.append(os.path.join(root, fname))
            lines.append(f"\n[{split}/{subfolder}] Found {len(all_files)} files total")
            
            # Get extensions
            exts = set(os.path.splitext(f)[1] for f in all_files)
            lines.append(f"  Extensions found: {exts}")
            
            # Check for tar files (label archives)
            tar_files = [f for f in all_files if f.endswith(".tar")]
            lines.append(f"  Tar archives found: {len(tar_files)}")
            if tar_files:
                lines.append(f"  Tar archive names: {[os.path.basename(t) for t in tar_files[:5]]}")
            
            # Inspect first image
            img_files = [f for f in all_files if f.endswith((".png", ".jpg", ".jpeg"))]
            if img_files:
                try:
                    img = Image.open(img_files[0])
                    arr = np.array(img)
                    lines.append(f"  First image ({os.path.basename(img_files[0])}): shape={arr.shape}, dtype={arr.dtype}")
                    lines.append(f"  Value range: min={arr.min()}, max={arr.max()}")
                    if "label" in subfolder:
                        unique_vals = np.unique(arr)
                        lines.append(f"  Unique label pixel values: {unique_vals}")
                        lines.append(f"  NOTE: target_label=10 is the default for mitral valve in task3")
                except Exception as e:
                    lines.append(f"  ERROR reading image: {e}")
            
            # List subdirectory structure (video folders)
            subdirs = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]
            if subdirs:
                lines.append(f"  Subdirectories (video folders): {len(subdirs)}")
                lines.append(f"  First 5 subdirs: {sorted(subdirs)[:5]}")
        else:
            lines.append(f"\n[{split}/{subfolder}] FOLDER DOES NOT EXIST: {p}")

report = "\n".join(lines)
print(report)
with open(report_path, "w") as f:
    f.write(report)
print(f"\nReport saved to {report_path}")
EOF
```

### 4.4 — Print Summary of All Analysis Reports

```bash
echo "=== DATA ANALYSIS COMPLETE ==="
echo ""
echo "--- TASK 1 REPORT ---"
cat ~/megha_workspace/analysis_reports/task1_data_analysis.txt
echo ""
echo "--- TASK 2 REPORT ---"
cat ~/megha_workspace/analysis_reports/task2_data_analysis.txt
echo ""
echo "--- TASK 3 REPORT ---"
cat ~/megha_workspace/analysis_reports/task3_data_analysis.txt
```

**After reading all reports, verify:**
- [ ] Task 1 train/images has ~1067 `.nii.gz` files
- [ ] Task 1 train/labels has ~27 `.nii.gz` files (label values include 1)
- [ ] Task 1 val/images has ~30 `.nii.gz` files (no labels)
- [ ] Task 2 train/images has ~105 `.nii.gz` files
- [ ] Task 2 train/labels has ~105 `.nii.gz` files (label values include 1 and 2)
- [ ] Task 2 val/images has ~20 `.nii.gz` files (no labels)
- [ ] Task 3 train has ~1619 frame files total (240 labeled, 1379 unlabeled)
- [ ] Task 3 val/images has ~48 image files (no labels)

If counts are significantly off, check the extraction in Step 3 and re-extract as needed.

---

## STEP 5 — Configure Path Constants in Prediction Scripts

Each `generate_task*_predictions.py` file has a section marked `# ===== Config (edit here) =====` that contains hardcoded path constants. These MUST be updated before training or inference.

**IMPORTANT: Read each file first (done in Step 1.3), then edit only the config constants section.**

### 5.1 — Edit Task 1 prediction config

Open `~/megha_workspace/MVAA/task1/generate_task1_predictions.py` and change the constants under `# ===== Config (edit here) =====` to:

```python
CKPT_PATH        = os.path.expanduser("~/megha_workspace/runs/task1/checkpoints/best_model.pt")
DATA_DIR         = os.path.expanduser("~/megha_workspace/data/t1_ct/val/images")
SUBMISSION_TASK_DIR = os.path.expanduser("~/megha_workspace/submission/t1_ct")
PRED_DIR         = os.path.expanduser("~/megha_workspace/submission/t1_ct")
OUTPUT_JSON      = os.path.expanduser("~/megha_workspace/submission/t1_ct/task1_predictions.json")
```

Use sed or a Python script to make this edit programmatically. Example with Python:
```bash
conda run -n megha_env python - << 'EOF'
import re, os

fpath = os.path.expanduser("~/megha_workspace/MVAA/task1/generate_task1_predictions.py")
with open(fpath, "r") as f:
    content = f.read()

# Print the config section to verify
config_start = content.find("# ===== Config (edit here) =====")
if config_start == -1:
    print("WARNING: Config section marker not found. Read the file and identify path constants manually.")
    print("Full file content:")
    print(content)
else:
    print("Config section found at character", config_start)
    print("Config section preview:")
    print(content[config_start:config_start+800])
EOF
```

After reading the actual content, make targeted replacements using Python string replacement. Do not blindly substitute — verify the variable names match what you see in the actual file.

### 5.2 — Edit Task 2 prediction config

Same process as 5.1 for `~/megha_workspace/MVAA/task2/generate_task2_predictions.py`:

```python
CKPT_PATH        = os.path.expanduser("~/megha_workspace/runs/task2/checkpoints/best_model.pt")
DATA_DIR         = os.path.expanduser("~/megha_workspace/data/t2_tee/val/images")
SUBMISSION_TASK_DIR = os.path.expanduser("~/megha_workspace/submission/t2_tee")
PRED_DIR         = os.path.expanduser("~/megha_workspace/submission/t2_tee")
OUTPUT_JSON      = os.path.expanduser("~/megha_workspace/submission/t2_tee/task2_predictions.json")
```

### 5.3 — Edit Task 3 prediction config

Same process for `~/megha_workspace/MVAA/task3/generate_task3_predictions.py`:

```python
CKPT_PATH        = os.path.expanduser("~/megha_workspace/runs/task3/checkpoints/best.pt")
DATA_DIR         = os.path.expanduser("~/megha_workspace/data/t3_vid/val/images")
VIDEO_FOLDERS    = []  # Empty = infer all folders/images under DATA_DIR
SUBMISSION_TASK_DIR = os.path.expanduser("~/megha_workspace/submission/t3_vid")
PRED_DIR         = os.path.expanduser("~/megha_workspace/submission/t3_vid")
OUTPUT_JSON      = os.path.expanduser("~/megha_workspace/submission/t3_vid/task3_predictions.json")
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## STEP 6 — Training

### IMPORTANT NOTES ON TRAINING STRATEGIES:
- **Task 1 (CT):** SEMI-SUPERVISED — MUST use the 1040 unlabeled images alongside 27 labeled. The baseline `train.py` handles this natively.
- **Task 2 (TEE):** FULLY SUPERVISED — all 105 training images have labels. Standard supervised training.
- **Task 3 (Video Frames):** SEMI-SUPERVISED — 240 labeled + 1379 unlabeled frames. MUST use unlabeled data.

The baseline uses a `target_label=10` default for Task 3 labels stored in tar archives. Verify this matches your data from the analysis in Step 4.

### 6.1 — Train Task 1 (CT Segmentation — Semi-Supervised)

```bash
cd ~/megha_workspace/MVAA

conda run -n megha_env python task1/train.py \
    --data-root ~/megha_workspace/data/t1_ct \
    --output-dir ~/megha_workspace/runs/task1 \
    2>&1 | tee ~/megha_workspace/logs/task1_train.log
```

**Monitor training:**
- Log is written to `~/megha_workspace/logs/task1_train.log`
- Checkpoints are saved to `~/megha_workspace/runs/task1/checkpoints/`
- The best checkpoint should be named `best_model.pt` (verify this after training)
- Training history CSV saved to `~/megha_workspace/runs/task1/history.csv`

**Check that training is actually running** (not silently failing):
```bash
# After ~5 minutes of training, check the log for progress
tail -50 ~/megha_workspace/logs/task1_train.log
# Look for epoch numbers, loss values, and no Python tracebacks
```

**If training fails** with an OOM (Out Of Memory) error on GPU, read the `train.py` to find the batch size argument and reduce it:
```bash
# Example — only if batch size argument exists in the script:
conda run -n megha_env python task1/train.py \
    --data-root ~/megha_workspace/data/t1_ct \
    --output-dir ~/megha_workspace/runs/task1 \
    --batch-size 1 \
    2>&1 | tee ~/megha_workspace/logs/task1_train.log
```

**After training completes, verify the checkpoint exists:**
```bash
ls -lh ~/megha_workspace/runs/task1/checkpoints/
```
The file `best_model.pt` (or the equivalent checkpoint name from the train.py source) must exist and be > 1 MB. If it doesn't exist, training failed — check the log.

### 6.2 — Train Task 2 (TEE Segmentation — Fully Supervised)

```bash
cd ~/megha_workspace/MVAA

conda run -n megha_env python task2/train.py \
    --data-dir ~/megha_workspace/data/t2_tee/train \
    --output-dir ~/megha_workspace/runs/task2 \
    2>&1 | tee ~/megha_workspace/logs/task2_train.log
```

**Monitor:**
```bash
tail -50 ~/megha_workspace/logs/task2_train.log
```

**Verify checkpoint after completion:**
```bash
ls -lh ~/megha_workspace/runs/task2/checkpoints/
```

### 6.3 — Train Task 3 (Video Frame Segmentation — Semi-Supervised)

```bash
cd ~/megha_workspace/MVAA

conda run -n megha_env python task3/train.py \
    --labeled-root ~/megha_workspace/data/t3_vid/train \
    --unlabeled-root ~/megha_workspace/data/t3_vid/train \
    --output-dir ~/megha_workspace/runs/task3 \
    2>&1 | tee ~/megha_workspace/logs/task3_train.log
```

**NOTE on unlabeled-root:** The `--unlabeled-root` should point to the directory containing the unlabeled image frames. Based on the baseline README, this is separate from the labeled training root. After Step 4 analysis, you will know the exact path. Adjust if needed.

**Monitor:**
```bash
tail -50 ~/megha_workspace/logs/task3_train.log
```

**Verify checkpoint:**
```bash
ls -lh ~/megha_workspace/runs/task3/checkpoints/
```
The checkpoint for task3 is named `best.pt` (NOT `best_model.pt` — this is different from tasks 1 and 2).

---

## STEP 7 — Generate Predictions on Validation Set

After all three tasks have trained successfully and all checkpoints exist, generate predictions.

### 7.1 — Generate Task 1 Predictions

```bash
cd ~/megha_workspace/MVAA

conda run -n megha_env python task1/generate_task1_predictions.py \
    2>&1 | tee ~/megha_workspace/logs/task1_predict.log
```

**Verify output:**
```bash
ls -lh ~/megha_workspace/submission/t1_ct/
```
Must contain:
- `task1_predictions.json` (not empty, not zero bytes)
- Multiple `.nii.gz` files (one per validation case, should be ~30 files)

**Sanity-check the JSON:**
```bash
conda run -n megha_env python -c "
import json
with open('$HOME/megha_workspace/submission/t1_ct/task1_predictions.json') as f:
    d = json.load(f)
cases = d.get('cases', [])
print(f'Number of cases in JSON: {len(cases)}')
print('First 3 cases:', cases[:3])
# Verify required fields
for c in cases:
    assert 'case_id' in c, f'Missing case_id in: {c}'
    assert 'segmentation' in c, f'Missing segmentation in: {c}'
print('All required fields (case_id, segmentation) present.')
"
```

### 7.2 — Generate Task 2 Predictions

```bash
cd ~/megha_workspace/MVAA

conda run -n megha_env python task2/generate_task2_predictions.py \
    2>&1 | tee ~/megha_workspace/logs/task2_predict.log
```

**Verify output:**
```bash
ls -lh ~/megha_workspace/submission/t2_tee/
```
Must contain:
- `task2_predictions.json`
- Multiple `.nii.gz` files (~20 files for validation)

**Sanity-check:**
```bash
conda run -n megha_env python -c "
import json
with open('$HOME/megha_workspace/submission/t2_tee/task2_predictions.json') as f:
    d = json.load(f)
cases = d.get('cases', [])
print(f'Number of cases in JSON: {len(cases)}')
print('First 3 cases:', cases[:3])
for c in cases:
    assert 'case_id' in c
    assert 'segmentation' in c
print('All required fields present.')
"
```

### 7.3 — Generate Task 3 Predictions

```bash
cd ~/megha_workspace/MVAA

conda run -n megha_env python task3/generate_task3_predictions.py \
    2>&1 | tee ~/megha_workspace/logs/task3_predict.log
```

**Verify output:**
```bash
ls -lh ~/megha_workspace/submission/t3_vid/
find ~/megha_workspace/submission/t3_vid/ -name "*_label_bin.png" | wc -l
```
Must contain:
- `task3_predictions.json`
- PNG binary mask files named `*_label_bin.png` organized in video subfolders
- Should have ~48 PNG files (one per validation frame)

**Sanity-check:**
```bash
conda run -n megha_env python -c "
import json
with open('$HOME/megha_workspace/submission/t3_vid/task3_predictions.json') as f:
    d = json.load(f)
cases = d.get('cases', [])
print(f'Number of cases in JSON: {len(cases)}')
print('First 3 cases:', cases[:3])
for c in cases:
    assert 'case_id' in c
    assert 'segmentation' in c
print('All required fields present.')
"
```

---

## STEP 8 — Build and Verify the Submission ZIP

### 8.1 — Verify exact directory structure before zipping

```bash
find ~/megha_workspace/submission/ -type f | sort
```

**Expected structure:**
```
~/megha_workspace/submission/t1_ct/task1_predictions.json
~/megha_workspace/submission/t1_ct/<case_id>.nii.gz
~/megha_workspace/submission/t1_ct/<case_id>.nii.gz
...
~/megha_workspace/submission/t2_tee/task2_predictions.json
~/megha_workspace/submission/t2_tee/<case_id>.nii.gz
...
~/megha_workspace/submission/t3_vid/task3_predictions.json
~/megha_workspace/submission/t3_vid/<video_folder>/<frame_id>_label_bin.png
...
```

**CRITICAL:** The zip root must contain `t1_ct/`, `t2_tee/`, `t3_vid/` DIRECTLY. There must be NO extra parent directory inside the zip.

### 8.2 — Create the submission zip

```bash
cd ~/megha_workspace/submission

zip -r ~/megha_workspace/final_submission.zip t1_ct t2_tee t3_vid
```

### 8.3 — Verify the zip structure matches the reference

```bash
echo "=== YOUR SUBMISSION STRUCTURE ==="
unzip -l ~/megha_workspace/final_submission.zip | head -60

echo ""
echo "=== REFERENCE SUBMISSION STRUCTURE ==="
unzip -l ~/megha_workspace/MVAA/submission.zip | head -60
```

Compare the two outputs. The top-level entries in your zip should be `t1_ct/`, `t2_tee/`, `t3_vid/` — exactly like the reference.

**If your zip has an extra parent directory (e.g., `submission/t1_ct/...` instead of `t1_ct/...`), rebuild it:**
```bash
rm ~/megha_workspace/final_submission.zip
cd ~/megha_workspace/submission
zip -r ~/megha_workspace/final_submission.zip t1_ct/ t2_tee/ t3_vid/
unzip -l ~/megha_workspace/final_submission.zip | head -10
# Verify first entry is "t1_ct/" not "submission/t1_ct/"
```

### 8.4 — Final zip size check

```bash
ls -lh ~/megha_workspace/final_submission.zip
```

The zip should be at least several MB (not 0 bytes or a few KB which would indicate empty predictions).

---

## STEP 9 — Submit to CodaBench

### 9.1 — Login

Navigate to: https://www.codabench.org/competitions/15662/

Log in to your account. If you don't have an account, register at https://www.codabench.org/accounts/signup

### 9.2 — Find the Submission page

Click on the "Submit" or "Participate" tab on the competition page.

### 9.3 — Upload the zip file

Upload `~/megha_workspace/final_submission.zip` to the CodaBench validation phase submission form.

**Submission limit:** Maximum **2 submissions per day**. Do not waste submissions on unverified zips.

### 9.4 — Check evaluation results

After submission, go to the "My Submissions" or "Results" tab:
1. Find your submission
2. Click on the **LOGS** tab
3. Go to **Prediction Logs**
4. Check **Ingestion stdout**

You will see per-task evaluation results in the format:
```
Task1_DSC: <value>   Task1_HD: <value>   Task1_ASD: <value>
Task2_DSC: <value>   Task2_HD: <value>   Task2_ASD: <value>
Task3_DSC: <value>   Task3_HD: <value>   Task3_ASD: <value>
```

**Interpretation:**
- DSC: Higher is better (range 0–1, perfect = 1.0)
- HD (Hausdorff Distance): Lower is better (in mm)
- ASD (Average Surface Distance): Lower is better (in mm)

If you see `DSC = 0` for all tasks, your submission zip has a structural problem. Re-check Step 8.

---

## STEP 10 — Improvement Loop

After getting initial baseline scores, the agent should iterate to improve performance. Record all scores.

### 10.1 — Log your results

```bash
cat >> ~/megha_workspace/logs/submission_history.txt << EOF
$(date)
Submission: final_submission.zip
Task1_DSC: __  Task1_HD: __  Task1_ASD: __
Task2_DSC: __  Task2_HD: __  Task2_ASD: __
Task3_DSC: __  Task3_HD: __  Task3_ASD: __
Notes: Baseline model, no modifications
EOF
```

### 10.2 — Priority improvements to consider

**Task 1 (CT, semi-supervised):**
- Increase training epochs (check `train.py` for `--epochs` argument)
- Add stronger data augmentation (random flips, rotations, intensity shifts)
- Improve pseudo-label generation for unlabeled CT volumes
- Try larger patch size if GPU memory allows

**Task 2 (TEE, supervised):**
- Focus on handling ultrasound artifacts (speckle noise, dropout)
- Use multi-scale features or attention mechanisms
- Ensure class-balanced training since label=1 and label=2 may be imbalanced

**Task 3 (Video frames, semi-supervised):**
- Apply ImageNet-pretrained encoders (already supported via `segmentation_models_pytorch`)
- Handle challenging conditions: reflections, blood occlusion, non-rigid motion
- Use strong augmentation for 2D frames (color jitter, blur, elastic transforms)

### 10.3 — Re-run the full pipeline after changes

After any code modification:
1. Re-run Step 6 (training) for the modified task
2. Re-run Step 7 (prediction) for that task
3. Re-run Step 8 (zip creation) — always start fresh:
   ```bash
   rm ~/megha_workspace/final_submission.zip
   ```
4. Re-submit (Step 9)

---

## EVALUATION METRIC REFERENCE

| Metric | Direction | Computation |
|--------|-----------|-------------|
| DSC (Dice Similarity Coefficient) | Higher = Better (0–1) | 2×\|pred∩GT\| / (\|pred\|+\|GT\|) |
| HD (Hausdorff Distance) | Lower = Better (mm) | max(directed_surface_distances) |
| ASD (Average Surface Distance) | Lower = Better (mm) | symmetric mean of surface distances |

**Task-specific label definitions:**
- Task 1: Binary (label 1 = foreground, i.e., mitral valve)
- Task 2: Multi-class (label 1 = leaflet class A, label 2 = leaflet class B); metrics averaged over classes
- Task 3: Binary (target_label = 10 in training masks; prediction saved as binary PNG)

**Penalty for missing/invalid predictions:**
- DSC = 0, HD = large penalty value, ASD = large penalty value
- This means every validation case MUST have a corresponding entry in the JSON and a prediction file

**Empty mask rule:** If both prediction AND ground truth are empty for a class → perfect score (DSC=1, HD=0, ASD=0) for that class.

---

## TROUBLESHOOTING GUIDE

| Problem | Diagnosis | Solution |
|---------|-----------|----------|
| `ModuleNotFoundError` during training | Package not installed in megha_env | `conda run -n megha_env pip install <package>` |
| `CUDA out of memory` | Batch size too large | Find `--batch-size` arg in train.py and set to 1 or 2 |
| `FileNotFoundError` on data paths | Wrong path to data directory | Check Step 3 extraction and re-verify paths |
| Training script has no `--batch-size` arg | Need to read train.py | `cat ~/megha_workspace/MVAA/task1/train.py \| grep batch` |
| Zip submission gets DSC=0 | Wrong zip structure | Re-check Step 8 — root must be `t1_ct/`, `t2_tee/`, `t3_vid/` directly |
| `task1_predictions.json` missing `cases` key | Wrong JSON format | Read generate script output and re-run |
| TEE labels show values other than 1,2 | Data format mismatch | Check actual unique values from Step 4 analysis and adjust label IDs |
| Task 3 tar labels not loading | Tar extraction issue | Check `dataset.py` tar-reading logic; manually test with `tar -tf <file>.tar` |
| gdown fails for Google Drive | File quota/auth issue | Try `gdown --id <FILE_ID>` or download manually and upload via scp |
| CodaBench submission stuck | Platform delay | Wait 10 minutes and refresh. If still stuck, contact tyt6xx@163.com |

---

## COMPETITION CONSTRAINTS SUMMARY

| Constraint | Detail |
|------------|--------|
| Max submissions/day | 2 (validation phase) |
| Inference time limit | 10 seconds per case on 12 GB GPU |
| External data | Only PUBLIC datasets and PUBLIC pre-trained models allowed |
| Private datasets | STRICTLY PROHIBITED |
| Private annotations of public data | STRICTLY PROHIBITED |
| Award eligibility prerequisite | GitHub repo URL + conference paper submission |
| Final evaluation | Docker container submission (not zip) |
| Task 1 & 3 learning strategy | Semi-supervised (unlabeled data MUST be used) |
| Task 2 learning strategy | Supervised only |
| Data license | CC BY-NC (non-commercial use only) |

---

## KEY CONTACTS

| Role | Contact |
|------|---------|
| Primary Organizer | Jieyun Bai — jbai996@aucklanduni.ac.cn |
| Technical Support | tyt6xx@163.com |
| Competition Platform | https://www.codabench.org/competitions/15662/ |
| Baseline Code | https://github.com/db0725/MVAA |

---

## TIMELINE (All 23:59 UTC+8)

| Date | Event |
|------|-------|
| April 15, 2026 | ✅ Training data released |
| June 01, 2026 | Validation data released + leaderboard opens |
| July 01, 2026 | Test phase / final evaluation begins |
| August 01, 2026 | **Final submission deadline** |
| September 01, 2026 | Winner announcement |
| September 27 – October 01, 2026 | MICCAI 2026 conference, France |

---

*End of INSTRUCTIONS.md — All steps must be followed in order. No step may be skipped.*
