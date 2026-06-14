import torch
from src.pipeline import IABFNetCorePipeline
from src.loss import IABFTaskLoss

def run_local_pipeline_verification():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Execution Device: {device}")
    
    model = IABFNetCorePipeline(feature_channels=64).to(device)
    loss_module = IABFTaskLoss().to(device)
    
    mock_camera_images = torch.randn(2, 3, 128, 128).to(device)
    mock_active_bev = torch.randn(2, 64, 128, 128).to(device)
    
    mock_class_ground_truth = torch.randint(0, 2, (2, 1, 128, 128)).float().to(device)
    mock_reg_ground_truth = torch.randn(2, 5, 128, 128).to(device)
    mock_ground_truth = torch.cat([mock_class_ground_truth, mock_reg_ground_truth], dim=1)
    
    print("\nExecuting Forward Pass...")
    predictions, alpha = model(mock_camera_images, mock_active_bev)
    
    print(f"Output Matrix Shape: {predictions.shape}")
    print(f"Gating Alpha Scalar (First Batch Element): {alpha[0].item():.4f}")
    
    print("\nExecuting Backward Pass Optimization Check...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    total_loss, cls_loss, reg_loss = loss_module(predictions, mock_ground_truth)
    total_loss.backward()
    optimizer.step()
    
    print(f"Total Combined Loss Value: {total_loss.item():.6f}")
    print(f"Classification Component Loss: {cls_loss.item():.6f}")
    print(f"Regression Component Loss: {reg_loss.item():.6f}")
    print("Pipeline and Task Loss Structure Verified.")

if __name__ == "__main__":
    run_local_pipeline_verification()



