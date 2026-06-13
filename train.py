import torch
from torch.utils.data import DataLoader
from src.pipeline import IABFNetCorePipeline
from src.loss import IABFTaskLoss
from src.dataset import SyntheticAutonomousDataset

def execute_training_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing optimization on device: {device}")
    
    dataset = SyntheticAutonomousDataset(num_samples=64, image_size=128, feature_channels=64)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = IABFNetCorePipeline(feature_channels=64).to(device)
    loss_module = IABFTaskLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Beginning validation training execution loop...")
    
    for epoch in range(1, 51):
        epoch_loss = 0.0
        epoch_cls = 0.0
        epoch_reg = 0.0
        alpha_accumulator = 0.0
        iterations = 0
        
        for camera_images, active_bev, ground_truth in dataloader:
            camera_images = camera_images.to(device)
            active_bev = active_bev.to(device)
            ground_truth = ground_truth.to(device)
            
            optimizer.zero_grad()
            
            predictions, alpha = model(camera_images, active_bev)
            
            total_loss, cls_loss, reg_loss = loss_module(predictions, ground_truth)
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_cls += cls_loss.item()
            epoch_reg += reg_loss.item()
            alpha_accumulator += alpha.mean().item()
            iterations += 1
            
        avg_loss = epoch_loss / iterations
        avg_cls = epoch_cls / iterations
        avg_reg = epoch_reg / iterations
        avg_alpha = alpha_accumulator / iterations
        
        print(f"Epoch {epoch}/50 -> Loss: {avg_loss:.4f} [Cls: {avg_cls:.4f}, Reg: {avg_reg:.4f}] | Mean Gating Alpha: {avg_alpha:.4f}")

if __name__ == "__main__":
    execute_training_pipeline()