import cv2
import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from views3 import poses_from_E, triangulate, F_to_E, GlobalOptimization


img0 = cv2.imread("sanity/rubiks_0.jpg")
img1 = cv2.imread("sanity/rubiks_1.jpg")
img2 = cv2.imread("sanity/rubiks_2.jpg")

pts0 = np.load("sanity/rubiks_0.npy")[:, :-1]
pts1 = np.load("sanity/rubiks_1.npy")[:, :-1]
pts2 = np.load("sanity/rubiks_2.npy")[:, :-1]

lines_img = np.vstack((img0, img1, img2))

pts0_plot = pts0
pts1_plot = pts1 + np.array([[0, img1.shape[0]]])
pts2_plot = pts2 + np.array([[0, 2*img1.shape[0]]])

# for k in range(pts0_plot.shape[0]):
#     cv2.line(lines_img, pts0_plot[k, :].astype(np.int64), pts1_plot[k, :].astype(np.int64), color = [0, 255, 0], thickness = 6)
#     cv2.line(lines_img, pts1_plot[k, :].astype(np.int64), pts2_plot[k, :].astype(np.int64), color = [0, 255, 0], thickness = 6)

# cv2.imwrite("line_img.jpg", lines_img)

F, _ = cv2.findFundamentalMat(pts0, pts1, method=cv2.FM_8POINT)

K = np.load("assets/intrinsics.npy")
K[:2, 2] = K[:2, 2][::-1]

E = K.T @ F @ K

E = F_to_E(E)

poses = poses_from_E(E)

X_list = []
r1_list = []
r2_list = []
C1_list = []
C2_list = []

for pose in poses:
    C1, R1 = pose
    T1 = np.eye(4)
    T1[:3, :3] = R1
    T1[:3, 3] = C1

    P2 = T1[:-1]

    X, geometries = triangulate(K, P2, pts0, pts1)
    X3D = X.cpu().detach().numpy().astype(np.float32)

    # X3D = np.zeros((8, 3))
    # X3D[:] = X[:]
    ret, rvecs, tvecs = cv2.solvePnP(X3D, pts2.astype(np.float32), K, None)
    R2 = cv2.Rodrigues(rvecs)[0]
    C2 = tvecs[:, 0]
    T2 = np.eye(4)
    # T[:3, :3] = P2[0, :3, :3].cpu().detach().numpy()
    # T[:3, 3] = P2[0, :3, 3].cpu().detach().numpy()
    T2[:3, :3] = R2
    T2[:3, 3] = C2

    cheirality = (X3D[:, -1] > 0) & ((X3D - C1)[:, -1] > 0) & ((X3D - C2)[:, -1] > 0)
    print(cheirality.sum())
    print("")

    X_list.append(X)
    r1_list.append(cv2.Rodrigues(R1)[0][:, 0])
    r2_list.append(rvecs[:, 0])
    C1_list.append(C1)
    C2_list.append(C2)

    geometries = []
    frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2., origin=[0, 0, 0])
    frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2., origin=[0, 0, 0])
    frame2.transform(T1.copy())
    frame3 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2., origin=[0, 0, 0])
    frame3.transform(T2.copy())

    geometries.append(frame1)
    geometries.append(frame2)
    geometries.append(frame3)

    pcd = np.zeros((X3D.shape[0], 3))
    pcd[:, :] = X3D[:, :]
    pts_vis = o3d.geometry.PointCloud()
    pts_vis.points = o3d.utility.Vector3dVector(pcd)
    geometries.append(pts_vis)

    # o3d.visualization.draw_geometries(geometries, window_name="Reconstruction")

i = 2 #int(input("Choose Index = "))

X = X_list[i]
r1 = r1_list[i]
r2 = r2_list[i]
C1 = C1_list[i]
C2 = C2_list[i]

global_opt = GlobalOptimization(K, pts0, pts1, pts2, X, r1, r2, C1, C2)
global_opt.to("cuda")
optimizer = torch.optim.AdamW(global_opt.parameters(), lr = 1e-2)

losses = []
for _ in tqdm(range(25000)):
    optimizer.zero_grad()
    loss = global_opt()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

print(loss.item())
plt.plot(losses[1000:])
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

X, R1, R2, t1, t2 = global_opt.get_params()

T1[:3, :3] = R1
T1[:3, 3] = t1

T2[:3, :3] = R2
T2[:3, 3] = t2

geometries = []
frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2., origin=[0, 0, 0])
frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2., origin=[0, 0, 0])
frame2.transform(T1.copy())
frame3 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2., origin=[0, 0, 0])
frame3.transform(T2.copy())

geometries.append(frame1)
geometries.append(frame2)
geometries.append(frame3)

pcd = np.zeros((X.shape[0], 3))
pcd[:, :] = X[:, :]
pts_vis = o3d.geometry.PointCloud()
pts_vis.points = o3d.utility.Vector3dVector(pcd)
geometries.append(pts_vis)

o3d.visualization.draw_geometries(geometries, window_name="Reconstruction")










