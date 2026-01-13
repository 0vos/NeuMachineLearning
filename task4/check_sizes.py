import os
from PIL import Image

img_dir = '/home/algo/video_agent_group/qianqian/NeuMachineLearning-main/task4/detection/train'
sizes = set()
for i in range(1, 81):
    fname = f"{i:04d}.jpg"
    path = os.path.join(img_dir, fname)
    if os.path.exists(path):
        with Image.open(path) as img:
            sizes.add(img.size)
    else:
        print(f"Missing {fname}")

print(f"Unique sizes: {sizes}")
