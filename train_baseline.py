from ultralytics import YOLO

def main():
    # 1. Load a model
    # By using 'yolov8n.pt', the library will automatically download 
    # the weights for you if they aren't already in your folder.
    model = YOLO("yolov8n.pt") 

    # 2. Train the model
    # 'data' points to your data.yaml file from Roboflow
    # 'epochs' is how many times the model sees the full dataset
    # 'imgsz' is the resolution (640 is standard)
    results = model.train(
        data="datasets/FVP-Baseline.v1-v3.yolov8/data.yaml", 
        epochs=50, 
        imgsz=640,
        device=0, # Change to device='cpu' if you don't have an NVIDIA GPU
        workers=0
    )

if __name__ == "__main__":
    main()