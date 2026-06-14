import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.pipeline import IABFNetCorePipeline
from src.loss import IABFTaskLoss
from src.real_dataset import RealNuScenesDataset

def introduce_online_fog(camera_tensors):
    fog_blur = torch.ones_like(camera_tensors) * 0.8
    return camera_tensors * 0.3 + fog_blur * 0.7

def execute_real_world_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Real-World Weather-Aware Optimization on: {device}")
    
    real_dataset = RealNuScenesDataset(
        data_root="data/nuscenes",
        metadata_file="data/nuscenes/metadata.json",
        image_size=128,
        feature_channels=64
    )
    
    real_dataloader = DataLoader(real_dataset, batch_size=2, shuffle=True)
    
    model = IABFNetCorePipeline(feature_channels=64).to(device)
    loss_module = IABFTaskLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    
    for epoch in range(1, 6):
        total_epoch_loss = 0.0
        alpha_tracker = 0.0
        steps = 0
        
        for camera_images, active_bev, ground_truth in real_dataloader:
            camera_images = camera_images.to(device)
            active_bev = active_bev.to(device)
            ground_truth = ground_truth.to(device)
            
            is_degraded = False
            if np.random.rand() > 0.5:
                camera_images = introduce_online_fog(camera_images)
                is_degraded = True
                
            optimizer.zero_grad()
            
            predictions, alpha = model(camera_images, active_bev)
            loss, _, _ = loss_module(predictions, ground_truth)
            
            if is_degraded:
                gating_loss = torch.mean((alpha - 0.25) ** 2) * 0.4
            else:
                gating_loss = torch.mean((alpha - 0.65) ** 2) * 0.4
                
            loss = loss + gating_loss
            
            loss.backward()
            optimizer.step()
            
            total_epoch_loss += loss.item()
            alpha_tracker += alpha.mean().item()
            steps += 1
            
        print(f"Weather-Aware Epoch {epoch}/5 -> Loss: {total_epoch_loss / steps:.5f} | Average Alpha: {alpha_tracker / steps:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/iabf_net_nuscenes.pth")
    print("Weather-aware checkpoint saved successfully to disk.")

if __name__ == "__main__":
    execute_real_world_training()