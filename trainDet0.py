import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from effdet import create_model, DetBenchTrain
from effdet.config import get_efficientdet_config
from torchvision.transforms import functional as F

# ==========================================
# 1. DATASET: YOLO format -> EfficientDet
# ==========================================
class EffDetDataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_size=512):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self): return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(os.path.join(self.images_dir, img_name)).convert("RGB")
        
        # EfficientDet-D0 is native at 512x512
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
                    boxes.append([ymin, xmin, ymax, xmax])
                    labels.append(cls + 1)

        target = {
            'bbox': torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            'cls': torch.as_tensor(labels, dtype=torch.float32) if labels else torch.zeros((0,)),
            'img_size': torch.tensor([(self.img_size, self.img_size)]),
            'img_scale': torch.tensor([1.0])
        }
        return image_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))

# ==========================================
# 2. LAPTOP-OPTIMIZED TRAINING LOOP
# ==========================================
def train():
    # --- UPDATE PATHS FOR YOUR HOME LAPTOP ---
    TRAIN_IMAGES = r"D:\Jeremy\UoN\Year 4\Final Year Project Coding\FYP_BaselineNN\datasets\FVP-Baseline.v17-vfinal.yolov8\train\images"
    TRAIN_LABELS = r"D:\Jeremy\UoN\Year 4\Final Year Project Coding\FYP_BaselineNN\datasets\FVP-Baseline.v17-vfinal.yolov8\train\labels"
    NUM_CLASSES = 6 
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Running on: {torch.cuda.get_device_name(0)} (Laptop Edition)")

    # Load Model
    config = get_efficientdet_config('tf_efficientdet_d0')
    net = create_model('tf_efficientdet_d0', pretrained=True, num_classes=NUM_CLASSES)
    train_net = DetBenchTrain(net, config)
    train_net.to(device)

    optimizer = torch.optim.AdamW(train_net.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()

    dataset = EffDetDataset(TRAIN_IMAGES, TRAIN_LABELS)
    
    # --- LAPTOP SPECS ---
    # Batch size 4 is the sweet spot for 4GB VRAM.
    # Num_workers 4 matches your 4 physical CPU cores.
    loader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=4, 
        pin_memory=True,
        prefetch_factor=2
    )

    print("🚀 Starting EfficientDet-D0 Training on GTX 1650...")
    for epoch in range(50):
        train_net.train()
        total_loss = 0
        for batch_idx, (images, targets) in enumerate(loader):
            images = torch.stack(images).to(device)
            t_boxes = [t['bbox'].to(device) for t in targets]
            t_labels = [t['cls'].to(device) for t in targets]
            target_dict = {'bbox': t_boxes, 'cls': t_labels}

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = train_net(images, target_dict)
                loss = output['loss']

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")

        print(f"--- Epoch {epoch+1} Finished | Avg Loss: {total_loss/len(loader):.4f} ---")

    torch.save(net.state_dict(), "effdet_d0_laptop.pth")
    print("Training finished! Weights saved as 'effdet_d0_laptop.pth'")

if __name__ == '__main__':
    train()