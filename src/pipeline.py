import torch
import torch.nn as nn
from src.modules import IlluminationGuidedAttention, EnvironmentalGating

class IABFNetCorePipeline(nn.Module):
    def __init__(self, feature_channels=64):
        super().__init__()
        self.camera_encoder = nn.Sequential(
            nn.Conv2d(3, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.illumination_block = IlluminationGuidedAttention(feature_channels)
        self.gating_block = EnvironmentalGating(feature_channels)
        
        self.cam_to_bev_projection = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        self.active_to_bev_projection = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        
        self.detection_head = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_channels, 6, kernel_size=1)
        )

    def forward(self, camera_images, active_sensor_bev):
        raw_camera_features = self.camera_encoder(camera_images)
        
        enhanced_camera_features = self.illumination_block(raw_camera_features)
        alpha = self.gating_block(enhanced_camera_features)
        
        camera_bev_space = self.cam_to_bev_projection(enhanced_camera_features)
        active_bev_space = self.active_to_bev_projection(active_sensor_bev)
        
        fused_bev_plane = (alpha * camera_bev_space) + ((1.0 - alpha) * active_bev_space)
        
        bounding_box_predictions = self.detection_head(fused_bev_plane)
        
        return bounding_box_predictions, alpha