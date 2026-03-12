import wandb
from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback

# 1. Initialize W&B
wandb.init(project="FYP_Conveyor_Counting", job_type="training")

# 2. Load the model
model = YOLO("yolov8n.pt")

# 3. Explicitly add the callback
add_wandb_callback(model, enable_model_checkpointing=True)

# 4. Train
model.train(data="data.yaml", epochs=100, imgsz=640)

# 5. Finish
wandb.finish()