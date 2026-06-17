import os
import json
import torch
import numpy as np
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
from src.pipeline import IABFNetCorePipeline
from src.real_dataset import RealNuScenesDataset

def apply_heavy_fog(image):
    img_np = np.array(image).astype(np.float32)
    fog_layer = np.ones_like(img_np) * 210
    foggy_np = (img_np * 0.35 + fog_layer * 0.65).astype(np.uint8)
    foggy_img = Image.fromarray(foggy_np)
    enhancer = ImageEnhance.Contrast(foggy_img)
    return enhancer.enhance(0.25)

def apply_heavy_rain(image):
    img_np = np.array(image).astype(np.float32)
    rain_mask = np.zeros_like(img_np)
    for i in range(0, rain_mask.shape[0], 6):
        for j in range(0, rain_mask.shape[1], 12):
            if np.random.rand() > 0.75:
                rain_mask[max(0, i-15):i, j:j+1, :] = 180
    rainy_np = np.clip(img_np * 0.65 + rain_mask, 0, 255).astype(np.uint8)
    return Image.fromarray(rainy_np)

def run_adverse_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    real_dataset = RealNuScenesDataset(
        data_root="data/nuscenes",
        metadata_file="data/nuscenes/metadata.json",
        image_size=128,
        feature_channels=64
    )
    
    camera_tensor_ref, active_bev, _ = real_dataset[0]
    
    with open("data/nuscenes/metadata.json", "r") as f:
        meta = json.load(f)
    
    base_img_path = os.path.join("data/nuscenes", meta[0]["cam_front_path"])
    raw_img = Image.open(base_img_path).convert("RGB")
    
    foggy_img = apply_heavy_fog(raw_img)
    rainy_img = apply_heavy_rain(raw_img)
    
    os.makedirs("outputs", exist_ok=True)
    foggy_img.save("outputs/simulated_fog.jpg")
    rainy_img.save("outputs/simulated_rain.jpg")
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    clean_tensor = transform(raw_img).unsqueeze(0).to(device)
    fog_tensor = transform(foggy_img).unsqueeze(0).to(device)
    rain_tensor = transform(rainy_img).unsqueeze(0).to(device)
    active_bev = active_bev.unsqueeze(0).to(device)
    
    model = IABFNetCorePipeline(feature_channels=64).to(device)
    if os.path.exists("models/iabf_net_nuscenes.pth"):
        model.load_state_dict(torch.load("models/iabf_net_nuscenes.pth", weights_only=True))
    model.eval()
    
    with torch.no_grad():
        _, alpha_clean = model(clean_tensor, active_bev)
        _, alpha_fog = model(fog_tensor, active_bev)
        _, alpha_rain = model(rain_tensor, active_bev)
        
    print("\n================ WEATHER STRESS TEST RESULTS ================")
    print(f"Normal Reference Frame Alpha : {alpha_clean.item():.4f}")
    print(f"Simulated Heavy Fog Alpha    : {alpha_fog.item():.4f}")
    print(f"Simulated Heavy Rain Alpha   : {alpha_rain.item():.4f}")
    print("=============================================================\n")

if __name__ == "__main__":
    run_adverse_test()