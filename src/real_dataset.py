import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class RealNuScenesDataset(Dataset):
    def __init__(self, data_root, metadata_file, image_size=128, feature_channels=64):
        self.data_root = data_root
        self.image_size = image_size
        self.feature_channels = feature_channels
        
        with open(metadata_file, "r") as f:
            self.annotations = json.load(f)
            
        self.image_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.annotations)

    def _load_pseudo_active_bev(self, binary_path):
        if os.path.exists(binary_path):
            raw_bin = np.fromfile(binary_path, dtype=np.float32)
            elements = self.feature_channels * self.image_size * self.image_size
            if len(raw_bin) >= elements:
                bev_data = raw_bin[:elements].reshape(self.feature_channels, self.image_size, self.image_size)
                return torch.tensor(bev_data, dtype=torch.float32)
        return torch.randn(self.feature_channels, self.image_size, self.image_size)

    def __getitem__(self, idx):
        entry = self.annotations[idx]
        
        cam_path = os.path.join(self.data_root, entry["cam_front_path"])
        if os.path.exists(cam_path):
            raw_image = Image.open(cam_path).convert("RGB")
            camera_tensor = self.image_transform(raw_image)
        else:
            camera_tensor = torch.randn(3, self.image_size, self.image_size)
            
        lidar_path = os.path.join(self.data_root, entry["lidar_top_path"])
        active_bev_tensor = self._load_pseudo_active_bev(lidar_path)
        
        class_matrix = torch.tensor(entry["gt_classification"], dtype=torch.float32).unsqueeze(0)
        reg_matrix = torch.tensor(entry["gt_regression"], dtype=torch.float32)
        ground_truth_tensor = torch.cat([class_matrix, reg_matrix], dim=0)
        
        return camera_tensor, active_bev_tensor, ground_truth_tensor