import torch
from torch.utils.data import DataLoader
from src.pipeline import IABFNetCorePipeline
from src.dataset import SyntheticAutonomousDataset

def run_scenario_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running scenario analysis on: {device}")
    
    dataset = SyntheticAutonomousDataset(num_samples=90, image_size=128, feature_channels=64)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = IABFNetCorePipeline(feature_channels=64).to(device)
    model.eval()
    
    day_alpha = 0.0
    night_alpha = 0.0
    rain_alpha = 0.0
    
    day_count = 0
    night_count = 0
    rain_count = 0
    
    with torch.no_grad():
        for idx, (camera_images, active_bev, _) in enumerate(dataloader):
            camera_images = camera_images.to(device)
            active_bev = active_bev.to(device)
            
            _, alpha = model(camera_images, active_bev)
            alpha_val = alpha.item()
            
            scenario = idx % 3
            if scenario == 0:
                day_alpha += alpha_val
                day_count += 1
            elif scenario == 1:
                night_alpha += alpha_val
                night_count += 1
            else:
                rain_alpha += alpha_val
                rain_count += 1
                
    print(f"\nAverage Alpha [Clear Day Scenario]: {day_alpha / day_count:.4f}")
    print(f"Average Alpha [Extreme Night Scenario]: {night_alpha / night_count:.4f}")
    print(f"Average Alpha [Heavy Rain/Fog Scenario]: {rain_alpha / rain_count:.4f}")

if __name__ == "__main__":
    run_scenario_evaluation()