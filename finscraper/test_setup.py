import torch
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import matplotlib

print("PyTorch version:", torch.__version__)
print("TensorFlow version:", tf.__version__)
print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("Matplotlib version:", matplotlib.__version__)

# Check for GPU availability
print("CUDA available:", torch.cuda.is_available())
print("TensorFlow GPU devices:", tf.config.list_physical_devices("GPU"))
