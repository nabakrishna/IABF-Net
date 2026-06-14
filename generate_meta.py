import os
import json
import numpy as np

os.makedirs("data/nuscenes/samples/CAM_FRONT", exist_ok=True)
os.makedirs("data/nuscenes/samples/LIDAR_TOP", exist_ok=True)

mock_image_path = "data/nuscenes/samples/CAM_FRONT/sample_0.jpg"
from PIL import Image
img = Image.new("RGB", (128, 128), color=(73, 109, 137))
img.save(mock_image_path)

mock_lidar_path = "data/nuscenes/samples/LIDAR_TOP/sample_0.pcd.bin"
mock_bin_data = np.random.randn(64 * 128 * 128).astype(np.float32)
mock_bin_data.tofile(mock_lidar_path)

mock_meta = []
for i in range(12):
    mock_meta.append({
        "cam_front_path": "samples/CAM_FRONT/sample_0.jpg",
        "lidar_top_path": "samples/LIDAR_TOP/sample_0.pcd.bin",
        "gt_classification": np.random.randint(0, 2, (128, 128)).tolist(),
        "gt_regression": np.random.randn(5, 128, 128).tolist()
    })

with open("data/nuscenes/metadata.json", "w") as f:
    json.dump(mock_meta, f)

print("Mock metadata structure generated successfully.")