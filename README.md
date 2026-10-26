# Real-Time Image Sharpening Project

This is the final v2 teacher–student project folder. `research_paper.pdf` is the two-column paper-style report; `report.md` is its editable text version. The trained student checkpoint and exported OpenVINO video model are in `checkpoints/`. The 125-image test results, Intel GPU benchmark, and anonymized MOS study records are in `results/`.

## Run the live webcam demo on Windows

Open PowerShell in this folder. Install packages into a virtual environment if needed:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then run:

```powershell
.venv\Scripts\python.exe demo.py --source 0 --backend openvino --device GPU --synthetic-degradation
```

The model processes 1920×1080 frames and shows smaller side-by-side previews. The left side is the simulated lower-quality input; the right side is the trained student output. Press `D` to toggle the simulated degradation and `Q` to quit. The on-screen FPS counter measures frame processing; it excludes camera capture and display.

## Files

| Location | Purpose |
|---|---|
| `sharpdistill/`, root `.py` files | Dataset, model, training, evaluation, export, speed test, live demo, and MOS analysis code |
| `notebooks/train_v2_in_colab.ipynb` | Colab training workflow used for the final student |
| `checkpoints/distill_best.pt` | Final trained student |
| `checkpoints/student_1920x1080_video.xml` and `.bin` | OpenVINO model for the Intel GPU demo |
| `results/full_test/` | Per-image metrics, category summary, and figures for 125 test images |
| `results/benchmark_gpu.json` | Measured 1080p speed |
| `results/mos/` | Blinded image pairs, anonymized responses, answer key, and MOS summary |
| `research_paper.pdf`, `report.md` | Paper-style PDF and editable text report |
| `figures/` | Actual model output comparisons and measured-result charts used in the paper |

The large DIV2K dataset and the 25 supplemental test images are not bundled in this folder. The official dataset can be obtained with `download_data.py`. The MOS answer key is included here for reproducibility after collection; it was excluded from the volunteer survey package.

The report focuses on the final teacher-guided student, as requested. It does not claim that the teacher guidance alone caused the measured improvement. The MOS study found no clear visual preference for the output, and that limitation is reported.
