

import torch
from src.pipeline import IABFNetCorePipeline

def run_local_pipeline_verification():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Execution Device: {device}")
    
    model = IABFNetCorePipeline(feature_channels=64).to(device)
    
    mock_camera_images = torch.randn(2, 3, 128, 128).to(device)
    mock_active_bev = torch.randn(2, 64, 128, 128).to(device)
    mock_ground_truth = torch.randn(2, 6, 128, 128).to(device)
    
    print("\nExecuting Forward Pass...")
    predictions, alpha = model(mock_camera_images, mock_active_bev)
    
    print(f"Output Matrix Shape: {predictions.shape}")
    print(f"Gating Alpha Scalar (First Batch Element): {alpha[0].item():.4f}")
    
    print("\nExecuting Backward Pass Optimization Check...")
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    loss = loss_function(predictions, mock_ground_truth)
    loss.backward()
    optimizer.step()
    
    print(f"Loss Compute Successful. Initial Loss Value: {loss.item():.6f}")
    print("Pipeline Structure Verified.")

if __name__ == "__main__":
    run_local_pipeline_verification()