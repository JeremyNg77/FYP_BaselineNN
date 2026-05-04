import torch
import numpy as np
from effdet import create_model, DetBenchPredict

def benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on: {torch.cuda.get_device_name(0)}")

    # 1. Load Model Architecture
    NUM_CLASSES = 6
    net = create_model('tf_efficientdet_d0', num_classes=NUM_CLASSES, pretrained=False)
    
    # Optional: Load your weights (though random weights take the same amount of time to process)
    try:
        net.load_state_dict(torch.load("effdet_d0_laptop.pth", map_location=device, weights_only=True))
    except FileNotFoundError:
        print("Weights file not found. Using random weights for speed test...")
        
    model = DetBenchPredict(net)
    model.to(device)
    model.eval()

    # 2. Create Dummy Data (EfficientDet-D0 uses 512x512 natively)
    dummy_image = torch.randn(1, 3, 512, 512).to(device)
    
    # Recreate the exact img_info dictionary format the wrapper expects
    dummy_info = {
        'img_size': torch.tensor([[512, 512]], dtype=torch.float32).to(device),
        'img_scale': torch.tensor([[1.0]], dtype=torch.float32).to(device)
    }

    # 3. GPU WARM-UP (Crucial for laptops)
    print("Warming up the GPU...")
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_image, img_info=dummy_info)

    # 4. ACTUAL BENCHMARK
    print("Measuring inference speed over 100 runs...")
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((100, 1))

    with torch.no_grad():
        for i in range(100):
            starter.record()
            _ = model(dummy_image, img_info=dummy_info)
            ender.record()
            
            # Wait for the GPU to actually finish the math
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[i] = curr_time

    avg_time = np.sum(timings) / 100
    
    print("\n" + "="*45)
    print("INFERENCE SPEED RESULT (Batch Size 1)")
    print("="*45)
    print(f"Average Latency: {avg_time:.2f} ms")
    print(f"Equivalent FPS:  {1000/avg_time:.2f} FPS")
    print("="*45)

if __name__ == '__main__':
    benchmark()