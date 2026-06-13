# import torch
# from torch.utils.data import Dataset

# class SyntheticAutonomousDataset(Dataset):
#     def __init__(self, num_samples=100, image_size=128, feature_channels=64):
#         self.num_samples = num_samples
#         self.image_size = image_size
#         self.feature_channels = feature_channels

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, idx):
#         is_night = idx % 2 == 1
        
#         if is_night:
#             camera_image = torch.randn(3, self.image_size, self.image_size) * 0.1
#         else:
#             camera_image = torch.randn(3, self.image_size, self.image_size) * 1.0
            
#         active_bev = torch.randn(self.feature_channels, self.image_size, self.image_size)
        
#         class_gt = torch.randint(0, 2, (1, self.image_size, self.image_size)).float()
#         reg_gt = torch.randn(5, self.image_size, self.image_size)
#         ground_truth = torch.cat([class_gt, reg_gt], dim=0)
        
#         return camera_image, active_bev, ground_truth



#new code ------------------------------------------------------------------------------------------
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

class SyntheticAutonomousDataset(Dataset):
    def __init__(self, num_samples=90, image_size=128, feature_channels=64):
        self.num_samples = num_samples
        self.image_size = image_size
        self.feature_channels = feature_channels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        scenario = idx % 3
        base_image = torch.rand(3, self.image_size, self.image_size)
        
        if scenario == 0:
            camera_image = F.adjust_brightness(base_image, 1.2)
            camera_image = F.adjust_contrast(camera_image, 1.1)
        elif scenario == 1:
            camera_image = F.adjust_brightness(base_image, 0.12)
            camera_image = F.adjust_contrast(camera_image, 0.25)
            noise = torch.randn(3, self.image_size, self.image_size) * 0.08
            camera_image = torch.clamp(camera_image + noise, 0.0, 1.0)
        else:
            camera_image = F.adjust_brightness(base_image, 0.5)
            camera_image = F.adjust_contrast(camera_image, 0.4)
            camera_image = F.gaussian_blur(camera_image, kernel_size=[5, 5], sigma=[1.8, 1.8])
            for _ in range(12):
                x = torch.randint(0, self.image_size - 12, (1,)).item()
                y = torch.randint(0, self.image_size - 2, (1,)).item()
                camera_image[:, y:y+2, x:x+12] = 0.85
                
        active_bev = torch.randn(self.feature_channels, self.image_size, self.image_size)
        if scenario == 2:
            active_bev = active_bev + torch.randn(self.feature_channels, self.image_size, self.image_size) * 0.25
            
        class_gt = torch.randint(0, 2, (1, self.image_size, self.image_size)).float()
        reg_gt = torch.randn(5, self.image_size, self.image_size)
        ground_truth = torch.cat([class_gt, reg_gt], dim=0)
        
        return camera_image, active_bev, ground_truth