import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from src.pipeline import IABFNetCorePipeline
from src.real_dataset import RealNuScenesDataset

def test_external_image(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(image_path):
        print(f"Error: Could not find the file at {image_path}")
        return
        
    real_dataset = RealNuScenesDataset(
        data_root="data/nuscenes",
        metadata_file="data/nuscenes/metadata.json",
        image_size=128,
        feature_channels=64
    )
    _, active_bev, _ = real_dataset[0]
    active_bev = active_bev.unsqueeze(0).to(device)
    
    raw_img = Image.open(image_path).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    camera_tensor = transform(raw_img).unsqueeze(0).to(device)
    
    model = IABFNetCorePipeline(feature_channels=64).to(device)
    checkpoint_path = "models/iabf_net_nuscenes.pth"
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()
    
    with torch.no_grad():
        _, alpha = model(camera_tensor, active_bev)
        
    print("\n================ EXTERNAL IMAGE TEST RESULTS ================")
    print(f"Target Image: {image_path}")
    print(f"Calculated Alpha Activation Value: {alpha.item():.4f}")
    print("=============================================================\n")

if __name__ == "__main__":
    test_external_image("outputs/my_foggy_road.jpg")