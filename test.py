import numpy as np
import cv2

# Define 3D points in world coordinates
object_points = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
], dtype=np.float32)

# Define corresponding 2D points in pixel coordinates
image_points = np.array([
    [300, 300],
    [400, 300],
    [300, 400],
    [400, 400],
    [320, 320],
    [420, 320],
    [320, 420],
    [420, 420],
], dtype=np.float32)

# Prepare data for cv2.calibrateCamera
object_points = [object_points]  # OpenCV expects a list of points for multiple images
image_points = [image_points]   # OpenCV expects a list of points for multiple images
image_size = (640, 480)  # Example image size (width, height)

# Provide an initial guess for the intrinsic matrix
initial_camera_matrix = np.array([
    [500, 0, image_size[0] / 2],  # Focal length and principal point x
    [0, 500, image_size[1] / 2],  # Focal length and principal point y
    [0, 0, 1]                     # Fixed
], dtype=np.float32)

# Perform camera calibration with an initial guess
flags = cv2.CALIB_USE_INTRINSIC_GUESS
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    object_points, image_points, image_size, initial_camera_matrix, None, flags=flags
)

# Output results
print("Camera Matrix (Intrinsic Parameters):")
print(camera_matrix)

print("\nDistortion Coefficients:")
print(dist_coeffs)

print("\nRotation Vectors:")
print(rvecs)

print("\nTranslation Vectors:")
print(tvecs)

# Reprojection error calculation
total_error = 0
for i in range(len(object_points)):
    projected_points, _ = cv2.projectPoints(
        object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
    )
    error = cv2.norm(image_points[i], projected_points, cv2.NORM_L2) / len(projected_points)
    total_error += error

print("\nReprojection Error:")
print(total_error / len(object_points))
