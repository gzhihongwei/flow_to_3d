import cv2
import numpy as np

# Example data
X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)  # 4 3D points
pts2 = np.array([[100, 200], [150, 200], [100, 250], [150, 250]], dtype=np.float32)  # 4 2D points
K = np.eye(3)  # Example intrinsic matrix

# SolvePnP
ret, rvec, tvec = cv2.solvePnP(X, pts2, K, None)

print("Rotation Vector:", rvec)
print("Translation Vector:", tvec)