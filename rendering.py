import pybullet as p
import numpy as np
from PIL import Image

def capture_frame(file_path: str):
    width, height, view_matrix, proj_matrix = 320, 240, *p.getDebugVisualizerCamera()[2:4]
    img = p.getCameraImage(width, height, view_matrix, proj_matrix)[2]  # RGB
    img_np = np.reshape(img, (height, width, 4))[:, :, :3]  # Drop alpha
    image = Image.fromarray(img_np.astype(np.uint8))
    image.save(file_path)

