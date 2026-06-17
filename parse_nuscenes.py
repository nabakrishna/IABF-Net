import os
import json
import numpy as np
from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes', verbose=True)
real_meta = []

for sample in nusc.sample:
    cam_front_token = sample['data']['CAM_FRONT']
    cam_data = nusc.get('sample_data', cam_front_token)
    cam_path = cam_data['filename']
    
    lidar_top_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_top_token)
    lidar_path = lidar_data['filename']
    
    class_matrix = np.zeros((128, 128))
    reg_matrix = np.zeros((5, 128, 128))
    
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        if 'vehicle' in ann['category_name']:
            x_coord = int(np.clip((ann['translation'][0] + 50) * (128 / 100), 0, 127))
            y_coord = int(np.clip((ann['translation'][1] + 50) * (128 / 100), 0, 127))
            class_matrix[y_coord, x_coord] = 1.0
            reg_matrix[0, y_coord, x_coord] = ann['translation'][0]
            reg_matrix[1, y_coord, x_coord] = ann['translation'][1]
            reg_matrix[2, y_coord, x_coord] = ann['translation'][2]
            reg_matrix[3, y_coord, x_coord] = ann['size'][0]
            reg_matrix[4, y_coord, x_coord] = ann['size'][1]
            
    real_meta.append({
        "cam_front_path": cam_path,
        "lidar_top_path": lidar_path,
        "gt_classification": class_matrix.tolist(),
        "gt_regression": reg_matrix.tolist()
    })

with open("data/nuscenes/metadata.json", "w") as f:
    json.dump(real_meta, f)

print(f"Successfully compiled tracking references for {len(real_meta)} real scenes.")