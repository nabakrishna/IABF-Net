import torch
from torch.utils.data import DataLoader
from src.pipeline import IABFNetCorePipeline
from src.real_dataset import RealNuScenesDataset

def evaluate_real_scenarios():
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
    
    low_visibility_alphas = []
    normal_visibility_alphas = []
    
    with torch.no_grad():
        for camera_images, active_bev, ground_truth in real_dataloader:
            camera_images = camera_images.to(device)
            active_bev = active_bev.to(device)
            
            _, alpha = model(camera_images, active_bev)
            alpha_val = alpha.item()
            
            class_gt = ground_truth[0, 0]
            if class_gt.sum() < 5.0:
                low_visibility_alphas.append(alpha_val)
            else:
                normal_visibility_alphas.append(alpha_val)
                
    if low_visibility_alphas:
        print(f"Low Object Density/Challenging Scenes Mean Alpha: {sum(low_visibility_alphas)/len(low_visibility_alphas):.4f}")
    if normal_visibility_alphas:
        print(f"High Object Density/Standard Scenes Mean Alpha: {sum(normal_visibility_alphas)/len(normal_visibility_alphas):.4f}")

if __name__ == "__main__":
    evaluate_real_scenarios()