import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from effdet import create_model, DetBenchPredict 
from torchvision.transforms import functional as F
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# ==========================================
# 1. DATASET: YOLO format -> EfficientDet
# ==========================================
class EffDetValDataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_size=512):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self): return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(os.path.join(self.images_dir, img_name)).convert("RGB")
        
        image = image.resize((self.img_size, self.img_size))
        image_tensor = F.to_tensor(image)

        label_path = os.path.join(self.labels_dir, os.path.splitext(img_name)[0] + ".txt")
        boxes, labels = [], []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    cls, xc, yc, bw, bh = map(float, line.strip().split())
                    xmin = (xc - bw/2) * self.img_size
                    xmax = (xc + bw/2) * self.img_size
                    ymin = (yc - bh/2) * self.img_size
                    ymax = (yc + bh/2) * self.img_size
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(cls + 1)

        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            'labels': torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,)),
            'image_id': torch.tensor([idx])
        }
        
        img_info = {
            'img_size': (self.img_size, self.img_size),
            'img_scale': 1.0
        }
        
        return image_tensor, target, img_info

def collate_fn(batch):
    return tuple(zip(*batch))

# ==========================================
# 2. EVALUATION SCRIPT
# ==========================================
def main():
    # --- PATHS & SETTINGS ---
    VAL_IMAGES = r"D:\Jeremy\UoN\Year 4\Final Year Project Coding\FYP_ModifiedNN\datasets\FVP-Baseline.v17-vfinal.yolov8\valid\images"
    VAL_LABELS = r"D:\Jeremy\UoN\Year 4\Final Year Project Coding\FYP_ModifiedNN\datasets\FVP-Baseline.v17-vfinal.yolov8\valid\labels"
    WEIGHTS_PATH = "effdet_d0_laptop.pth"
    
    # EXACT class names from your data.yaml
    CLASS_NAMES = ["Part 1", "Part 2", "Part 3", "Part 4", "Part 5", "Part 6"] 
    NUM_CLASSES = len(CLASS_NAMES) 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 1. LOAD MODEL ---
    print("Loading Google EfficientDet-D0 for Evaluation...")
    
    net = create_model('tf_efficientdet_d0', num_classes=NUM_CLASSES, pretrained=False)
    net.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
    model = DetBenchPredict(net)
    
    model.to(device)
    model.eval()

    # --- 2. LOAD DATA ---
    val_dataset = EffDetValDataset(VAL_IMAGES, VAL_LABELS)
    val_loader = DataLoader(val_dataset, batch_size=4, collate_fn=collate_fn)

    # --- 3. RUN EVALUATION ---
    print("Running Inference with TorchMetrics...")
    metric = MeanAveragePrecision(class_metrics=True)

    with torch.no_grad():
        for images, targets, img_infos in val_loader:
            images = torch.stack(images).to(device)
            
            # FIX: Format img_info exactly as the effdet wrapper expects (a dictionary of tensors)
            img_info_dict = {
                'img_size': torch.tensor([info['img_size'] for info in img_infos], dtype=torch.float32).to(device),
                'img_scale': torch.tensor([info['img_scale'] for info in img_infos], dtype=torch.float32).to(device)
            }
            
            outputs = model(images, img_info=img_info_dict)
            
            preds = []
            for out in outputs:
                if out.shape[0] == 0:
                    preds.append({"boxes": torch.zeros((0, 4)), "scores": torch.zeros((0,)), "labels": torch.zeros((0,))})
                else:
                    preds.append({
                        "boxes": out[:, :4].cpu(),
                        "scores": out[:, 4].cpu(),
                        "labels": out[:, 5].cpu().to(torch.int64)
                    })
            
            metric.update(preds, targets)

    results = metric.compute()

    # --- 4. PRINT OVERALL METRICS ---
    print("\n" + "="*45)
    print("OVERALL METRICS")
    print("="*45)
    print(f"mAP@50:     {results['map_50'].item():.4f}")
    print(f"mAP@50-95:  {results['map'].item():.4f}")
    print(f"Precision*: {results['map_50'].item():.4f}") 
    print(f"Recall*:    {results['mar_100'].item():.4f}")

    # --- 5. PRINT CLASS-SPECIFIC METRICS ---
    print("\n" + "="*45)
    print("CLASS-SPECIFIC METRICS")
    print("="*45)
    print(f"{'Class':<12} | {'Metric':<15} | {'Value':<10}")
    print("-" * 45)
    
    # Check if classes were evaluated
    if 'classes' in results and len(results['classes']) > 0:
        evaluated_classes = results['classes'].tolist()
        
        for i, class_idx in enumerate(evaluated_classes):
            class_idx = int(class_idx)
            actual_class_index = class_idx - 1 
            
            if actual_class_index < 0 or actual_class_index >= len(CLASS_NAMES):
                continue 
                
            class_name = CLASS_NAMES[actual_class_index]
            
            # Using 'map_per_class' as the primary strict metric (mAP@50-95)
            class_map_50_95 = results['map_per_class'][i].item() if 'map_per_class' in results else -1
            
            # Since torchmetrics often omits map_50_per_class, we handle it safely:
            class_map50 = results.get('map_50_per_class', results.get('map_per_class'))[i].item()
            class_r = results.get('mar_100_per_class', results.get('mar_1'))[i].item()
            
            print(f"{class_name:<12} | {'mAP@50-95':<15} | {class_map_50_95:.4f}")
            print(f"{'':<12} | {'Recall (AR)':<15} | {class_r:.4f}")
            print("-" * 45)
    else:
        print("No class-specific data returned by TorchMetrics.")

if __name__ == '__main__':
    main()