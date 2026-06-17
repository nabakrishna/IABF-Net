import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from src.pipeline import IABFNetCorePipeline
from src.real_dataset import RealNuScenesDataset

def isolate_environmental_proof_cases():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    real_dataset = RealNuScenesDataset(
        data_root="data/nuscenes",
        metadata_file="data/nuscenes/metadata.json",
        image_size=128,
        feature_channels=64
    )
    real_dataloader = DataLoader(real_dataset, batch_size=1, shuffle=False)
    
    model = IABFNetCorePipeline(feature_channels=64).to(device)
    model.eval()
    
    high_alpha_sample = None
    low_alpha_sample = None
    
    with torch.no_grad():
        for idx, (camera_images, active_bev, _) in enumerate(real_dataloader):
            camera_images = camera_images.to(device)
            active_bev = active_bev.to(device)
            
            _, alpha = model(camera_images, active_bev)
            alpha_val = alpha.item()
            
            meta_entry = real_dataset.annotations[idx]
            
            if alpha_val > 0.68 and high_alpha_sample is None:
                high_alpha_sample = {
                    "tensor": camera_images[0],
                    "alpha": alpha_val,
                    "path": meta_entry["cam_front_path"]
                }
                
            if alpha_val < 0.46 and low_alpha_sample is None:
                low_alpha_sample = {
                    "tensor": camera_images[0],
                    "alpha": alpha_val,
                    "path": meta_entry["cam_front_path"]
                }
                
            if high_alpha_sample and low_alpha_sample:
                break
                
    os.makedirs("outputs", exist_ok=True)
    
    if high_alpha_sample:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
        denorm_high = high_alpha_sample["tensor"] * std + mean
        save_image(denorm_high, "outputs/case_high_camera_reliance.jpg")
        print("--- CAMERA RELIANCE PROOF CASE DISCOVERED ---")
        print(f"Target Filename: {high_alpha_sample['path']}")
        print(f"Calculated Alpha Activation Value: {high_alpha_sample['alpha']:.4f}\n")
        
    if low_alpha_sample:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
        denorm_low = low_alpha_sample["tensor"] * std + mean
        save_image(denorm_low, "outputs/case_low_camera_reliance.jpg")
        print("--- SENSOR SHIFT RELIANCE PROOF CASE DISCOVERED ---")
        print(f"Target Filename: {low_alpha_sample['path']}")
        print(f"Calculated Alpha Activation Value: {low_alpha_sample['alpha']:.4f}\n")

if __name__ == "__main__":
    isolate_environmental_proof_cases()