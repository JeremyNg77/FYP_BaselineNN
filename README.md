# Final Year Project - Baseline Neural Networks (FYPBaselineNN)

This repository contains the baseline object detection neural networks for my Final Year Project. It includes scripts for training, evaluating, and running inference (with object counting) using two primary architectures: **YOLOv8** and **EfficientDet-D0**. 

Several baseline models have already been trained, and their weights (e.g., `YOLOv8n - FINAL.pt`, `effdet_d0_laptop.pth`) are referenced within the scripts for immediate testing and deployment.

## 📁 File & Folder Overview

### YOLOv8 Scripts
- **`trainBaseline.py`**: The training script for YOLOv8. It downloads the pre-trained `yolov8n.pt` weights and fine-tunes the model on our custom dataset using the `ultralytics` library.
- **`testBaseline.py`**: The evaluation script for a trained YOLOv8 model. It calculates standard object detection metrics (mAP@50, mAP@50-95, Precision, Recall) on the validation dataset.
- **`parameterCount.py`**: A simple utility script that prints out the layer architecture and total parameter count of your trained YOLOv8 model.
- **`runScript.py`**: A real-time video inference script. It runs YOLOv8 tracking on a specified video, detects objects, draws bounding boxes, and increments a counter when tracked objects cross a pre-defined line on the screen.

### EfficientDet-D0 Scripts
- **`trainDet0.py`**: A custom PyTorch training loop for EfficientDet-D0. It includes a custom Dataset class to convert YOLO-formatted labels into the format EfficientDet expects. It is highly optimized for laptop GPUs (using mixed precision, batch size of 4, and utilizing CPU workers).
- **`testDet0.py`**: The evaluation script for EfficientDet-D0. It uses `TorchMetrics` to calculate mAP and Recall (Average Recall) across the validation dataset.
- **`speedEffdet.py`**: A benchmarking script to test the inference latency and FPS capabilities of EfficientDet-D0 on your specific GPU.

---

## 🚀 How to Train the Models

### 1. Training YOLOv8
YOLOv8 training is highly streamlined. 
1. Open `trainBaseline.py`.
2. Ensure the `data` parameter points to your correct `data.yaml` file.
3. Adjust hyperparameters if necessary (e.g., `epochs=150`, `imgsz=640`).
4. Run the script:
   ```bash
   python trainBaseline.py
   ```
   *Note: Training outputs, including the best weights (`best.pt`), will be saved automatically in the `runs/detect/train...` directory.*

### 2. Training EfficientDet-D0
EfficientDet requires a bit more manual setup but is fully configured for a laptop environment in this repo.
1. Open `trainDet0.py`.
2. Update `TRAIN_IMAGES` and `TRAIN_LABELS` with the absolute paths to your dataset.
3. Run the script:
   ```bash
   python trainDet0.py
   ```
   *Note: The script will train for 50 epochs by default and save the final weights as `effdet_d0_laptop.pth` in the root directory.*

---

## 📊 How to Evaluate the Models

If you want to check the mAP, Precision, or Recall of your trained models against a validation dataset:

**For YOLOv8:**
1. Open `testBaseline.py`.
2. Check that the `model = YOLO(...)` path points to your newly trained weights (e.g., `best.pt`).
3. Run `python testBaseline.py`. It will output overall and class-specific metrics to the console.

**For EfficientDet-D0:**
1. Open `testDet0.py`.
2. Verify the `VAL_IMAGES` and `VAL_LABELS` point to your validation data.
3. Run `python testDet0.py`. It uses `TorchMetrics` to output the validation results.

---

## 🎥 Running Video Inference & Counting

To test the YOLOv8 model on a real video and count objects as they pass a certain point:
1. Open `runScript.py`.
2. Ensure `MODEL_PATH` points to your best YOLOv8 weights (e.g., `YOLOv8n - FINAL.pt`).
3. Set `VIDEO_PATH` to your test MP4 file.
4. (Optional) Adjust `LINE_START` and `LINE_END` coordinates depending on your video's resolution and where you want the "counting line" to be drawn.
5. Run the script:
   ```bash
   python runScript.py
   ```
   *A window will pop up showing the real-time tracking. The final processed video will be saved as `counted_output.mp4`. Press 'q' to stop early.*

---

## ⚙️ Additional Utilities

- **Check YOLO Parameter Count**: Run `python parameterCount.py` to see the internal structure and weight size of your YOLO model.
- **Benchmark EfficientDet Speed**: Run `python speedEffdet.py` to do a warm-up and 100-run latency test to calculate the actual FPS your GPU can output for EfficientDet-D0.