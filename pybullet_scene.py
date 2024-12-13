"""
This file contains the pybullet wrapper for the scene generation and object storage.
"""
import os
import time
import open3d as o3d

import cv2
import numpy as np
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc
import yaml
from scipy.spatial.transform import Rotation as R

urdf_root_path = pybullet_data.getDataPath()

COLORS = {
    "blue": np.array([78, 121, 167]) / 255.0,  # blue
    "green": np.array([89, 161, 79]) / 255.0,  # green
    "brown": np.array([156, 117, 95]) / 255.0,  # brown
    "orange": np.array([242, 142, 43]) / 255.0,  # orange
    "yellow": np.array([237, 201, 72]) / 255.0,  # yellow
    "gray": np.array([186, 176, 172]) / 255.0,  # gray
    "red": np.array([255, 87, 89]) / 255.0,  # red
    "purple": np.array([176, 122, 161]) / 255.0,  # purple
    "cyan": np.array([118, 183, 178]) / 255.0,  # cyan
    "pink": np.array([255, 157, 167]) / 255.0,
}  

class SceneCamera:
    def __init__(self, client_id, cam_args):

        self.client_id = client_id

        self.mode = cam_args["mode"]  # or "position"
        self.target = cam_args["target"]

        # For distance mode:
        self.distance = cam_args["distance"]
        self.yaw = cam_args["yaw"]
        self.pitch = cam_args["pitch"]
        self.roll = cam_args["roll"]
        self.up_axis_index = cam_args["up_axis_index"]

        # For position mode:
        self.eye = cam_args["eye"]
        self.up_vec = cam_args["up_vec"]

        # Intrinsics:
        self.width = cam_args["width"]
        self.height = cam_args["height"]
        self.fov = cam_args["fov"]
        self.near = cam_args["near"]
        self.far = cam_args["far"]

        # If camera is already saved somewhere:
        self.view_matrix = cam_args["view_matrix"]
        self.projection_matrix = cam_args["projection_matrix"]

        self.quat = R.from_euler("xyz", [self.yaw, self.pitch, self.roll]).as_quat()
        self.aspect = self.width / self.height

        # Define Intrinsics:
        self.f = 1 / np.tan(self.fov/2 * (np.pi / 180))
        # self.scale = (2 * self.near * self.far) / (self.near - self.far)
        self.K = np.array([[self.f, 0., self.width / 2],
                           [0., self.f, self.height / 2],
                           [0., 0., 1.]])

        if (self.view_matrix is None) or (self.view_matrix == "None"):
            if self.mode == "distance":
                self.view_matrix = self.client_id.computeViewMatrixFromYawPitchRoll(
                    self.target,
                    self.distance,
                    self.yaw,
                    self.pitch,
                    self.roll,
                    self.up_axis_index,
                )
            elif self.mode == "position":
                self.view_matrix = self.client_id.computeViewMatrix(
                    self.eye, self.target, self.up_vec
                )

        if (self.projection_matrix is None) or (self.projection_matrix == "None"):
            self.projection_matrix = self.client_id.computeProjectionMatrixFOV(
                self.fov, self.aspect, self.near, self.far
            )

    def capture(self):
        _, _, rgb, depth, segs = self.client_id.getCameraImage(
            self.width,
            self.height,
            self.view_matrix,
            self.projection_matrix,
        )

        rgb = np.reshape(rgb, (self.width, self.height, 4))
        rgb = rgb[..., :3]

        depth = np.reshape(depth, (self.width, self.height))

        return rgb, depth, segs

    def get_pointcloud(self, depth, seg_img=None):
        """Returns a point cloud and its segmentation from the given depth image

        Args:
        -----
            depth (np.array): depth image
            width (int): width of the image
            height (int): height of the image
            view_matrix (np.array): 4x4 view matrix
            proj_matrix (np.array): 4x4 projection matrix
            seg_img (np.array): segmentation image

        Return:
        -------
            pcd (np.array): Nx3 point cloud
            pcd_seg (np.array): N array, segmentation of the point cloud
        """
        # based on https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer

        # create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
        proj_matrix = np.asarray(self.projection_matrix).reshape([4, 4], order="F")
        view_matrix = np.asarray(self.view_matrix).reshape([4, 4], order="F")
        # tran_pix_world = np.linalg.inv(
        #     np.matmul(proj_matrix, view_matrix)
        # )  # Pixel to 3D transformation
        tran_pix_world = np.linalg.inv(proj_matrix)

        # create a mesh grid with pixel coordinates, by converting 0 to width and 0 to height to -1 to 1
        y, x = np.mgrid[-1 : 1 : 2 / self.height, -1 : 1 : 2 / self.width]
        y *= -1.0  # y is reversed in pixel coordinates

        # Reshape to single dimension arrays
        # x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)

        # Homogenize:
        pixels = np.stack([x, y, depth, np.ones_like(depth)], axis=-1)
        # filter out "infinite" depths:
        # fin_depths = np.where(depth < 0.99)
        # pixels = pixels[fin_depths]

        # Depth z is between 0 to 1, so convert it to -1 to 1 (for backprojection)
        pixels[..., 2] = 2 * pixels[..., 2] - 1

        if seg_img is not None:
            pcd_seg = np.array(seg_img)
            # pcd_seg = seg_img.reshape(-1)[fin_depths]  # filter out "infinite" depths
        else:
            pcd_seg = None

        # turn pixels to world coordinates
        points = np.matmul(tran_pix_world, pixels[..., None])[..., 0]
        points /= points[..., 3:4]  # Homogenize in 3D
        points = points[..., :3]  # Remove last axis ones
        # points[..., 2] = -points[..., 2]
        points = -points[:]

        # geometries = []
        # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2., origin=[0, 0, 0])
        # geometries.append(frame)

        # pcd = points.reshape((-1, 3))
        # # pcd = np.zeros((points.shape[0], 3))
        # # pcd[:, :] = X3D[:, :]
        # pts_vis = o3d.geometry.PointCloud()
        # pts_vis.points = o3d.utility.Vector3dVector(pcd)
        # geometries.append(pts_vis)

        # o3d.visualization.draw_geometries(geometries, window_name="Reconstruction")

        return points, pcd_seg


class Scene:
    def __init__(
        self,
        args,
        seed=None,
        gui=True,
        timestep=1 / 480,
    ):
        self.seed = seed
        self.args = args
        self.max_control_iters = self.args["max_control_iters"]
        self.stability_iters = self.args["stability_iters"]
        self.tol = self.args["tolerance"]
        self.max_frames = self.args["max_frames"]

        self.config_path = (
            self.args["scene_config_folder"] + self.args["scene"] + ".yaml"
        )
        assert os.path.isfile(
            self.config_path
        ), f"Error: {self.config_path} is not a file or does not exist! Check your configs"

        with open(self.config_path, "r") as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

        self.client_id = bc.BulletClient(
            p.GUI if gui else p.DIRECT
        )  # Initialize the bullet client

        self.client_id.setAdditionalSearchPath(
            pybullet_data.getDataPath()
        )  # Add pybullet's data package to path
        self.client_id.setTimeStep(timestep)  # Set simulation timestep
        # self.client_id.configureDebugVisualizer(
        #     p.COV_ENABLE_SHADOWS, 1
        # )  # Enable Shadows
        self.client_id.configureDebugVisualizer(
            p.COV_ENABLE_GUI, 0
        )  # Disable Frame Axes

        self.client_id.resetSimulation()
        self.client_id.setGravity(0, 0, -9.8)  # Set Gravity

        self.plane = self.client_id.loadURDF(
            "plane.urdf", basePosition=(0, 0, 0), useFixedBase=True
        )  # Load a floor

        self.load_objects()

        # Setup Camera:
        self.camera_list = []
        cam_args = self.config["camera"]
        # for i, cam_args in self.config["camera"].items():
        self.cam = SceneCamera(self.client_id, cam_args)
            # self.camera_list.append(cam)

        print("Loading Perception Modules")
        self.prev_press = -1
        self.num_pressed = 1
        self.current_focus = 0

        self.prev_keys = {}

        self.start_state = self.client_id.saveState()

        self.wait_for_stability()

        # self.gsam = grounded_sam()

    def reset(self):

        self.client_id.restoreState(stateId = self.start_state)

    def load_objects(self):
        assert "objects" in self.config, "Error: No objects in the config file"

        self.objects = {}
        self.grasp_obj_names = []
        self.grasp_obj_ids = []
        self.fixed_obj_ids = []
        for obj_name, obj in self.config["objects"].items():
            obj_path = self.args["objects_folder"] + obj["file"]
            obj_id = self.client_id.loadURDF(
                obj_path,
                obj["pos"],
                obj["orn"],
                useFixedBase=obj["fixed_base"],
                globalScaling=obj["scale"],
            )
            self.objects[obj_name] = obj_id
            if not obj["fixed_base"]:
                self.grasp_obj_names.append(obj_name)
                self.grasp_obj_ids.append(obj_id)
                # self.client_id.changeDynamics(obj_id, -1, lateralFriction=10000.)
            else:
                self.fixed_obj_ids.append(obj_id)

        self.num_objs = len(self.grasp_obj_ids)
        self.controlled_obj = 0
        self.controlled_obj_name = self.grasp_obj_names[self.controlled_obj]

        print("Currently controlling: " + self.controlled_obj_name)

        # !!! You also need to add functionality for articulations and constraints, which I am skipping for now

    def move_object(self, obj_id, force_pt, force_vec, vid_path):

        pose = self.get_object_pose(obj_id)
        # obj_trans = self.pose_to_transformation(pose)
        # obj_frame = self.draw_frame(obj_trans)

        # force_pose = pose.copy()
        # force_pose[:3] = force_pose[:3] + force_pt
        # obj_frame = self.draw_frame(obj_trans)

        # self.client_id.applyExternalForce(obj_id, -1, force_vec, force_pt, p.LINK_FRAME)
        self.client_id.applyExternalTorque(obj_id, -1, force_vec, p.LINK_FRAME)

        self.record_motion(vid_path)
        # for _ in range(self.stability_iters):
        #     self.client_id.stepSimulation()

        print("")

    def record_motion(self, vid_path):

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        writer = cv2.VideoWriter(vid_path, fourcc, fps, (self.cam.width, self.cam.height))

        cam_rgb, cam_depth, cam_seg = self.cam.capture()
        cam_rgb = cv2.cvtColor(cam_rgb, cv2.COLOR_BGR2RGB)
        writer.write(cam_rgb)
        cam_pcd, cam_pcd_seg = self.cam.get_pointcloud(cam_depth, cam_seg)
        np.save(vid_path[:-4] + "_pcd.npy", cam_pcd)

        video_seg = [cam_pcd_seg]

        for _ in range(self.max_frames):

            self.client_id.stepSimulation()
            cam_rgb, cam_depth, _ = self.cam.capture()
            cam_rgb = cv2.cvtColor(cam_rgb, cv2.COLOR_BGR2RGB)
            writer.write(cam_rgb)
            cam_pcd, cam_pcd_seg = self.cam.get_pointcloud(cam_depth, cam_seg)
            video_seg.append(cam_pcd_seg)
            
            # time.sleep(0.01)
            obj_vels = np.zeros((self.num_objs, 6))
            for i in range(self.num_objs):
                obj_vels[i] = self.get_object_vel(self.grasp_obj_names[i])

            error = np.abs(obj_vels)
            if np.all(error < self.tol):
                break

        seg = np.stack(video_seg, axis=0)
        np.save(vid_path[:-4] + "_seg.npy", seg)

        writer.release()

    
    def wait_for_stability(self):
        for _ in range(self.stability_iters):
            self.client_id.stepSimulation()
            time.sleep(0.01)
            obj_vels = np.zeros((self.num_objs, 6))
            for i in range(self.num_objs):
                obj_vels[i] = self.get_object_vel(self.grasp_obj_names[i])

            error = np.abs(obj_vels)
            if np.all(error < self.tol):
                break

    def get_object_pose(self, obj_name):
        if type(obj_name) == int:  # If it is object id
            pose = self.client_id.getBasePositionAndOrientation(obj_name)
        else:
            pose = self.client_id.getBasePositionAndOrientation(self.objects[obj_name])
        return np.array([*pose[0], *pose[1]])

    def get_object_vel(self, obj_name):
        vel = self.client_id.getBaseVelocity(self.objects[obj_name])
        return np.array([*vel[0], *vel[1]])

    def combine_poses(self, pose_list):
        """
        Order of the list is the order in which it will be applied
        """

        T = np.eye(4)
        for pose in pose_list:
            T = T @ self.pose_to_transformation(pose)

        final_pose = self.transformation_to_pose(T)

        return final_pose
    
    
    def save_img_and_seg(self):
        for i in range(len(self.camera_list)):
            cam = self.camera_list[i]
            cam_rgb, cam_depth = cam.capture()
            cam_pcd, cam_pcd_seg, cam_pcd_ind = cam.get_pointcloud(cam_depth)

            r = np.max(cam_depth) - np.min(cam_depth)
            m = np.min(cam_depth)
            cam_depth = np.round((255 / r) * (cam_depth - m)).astype(np.uint8)

            cv2.imwrite(
                "vis/Camera_" + str(i + 1) + "RGB_" + str(self.drops + 1) + ".jpg",
                cv2.cvtColor(cam_rgb, cv2.COLOR_BGR2RGB),
            )
            cv2.imwrite(
                "vis/Camera_" + str(i + 1) + "Depth_" + str(self.drops + 1) + ".jpg",
                cam_depth,
            )
            # cv2.imwrite("vis/Camera_" + str(i+1) + "Seg_" + str(self.drops+1) + ".jpg", cam_pcd_seg)

        pcd, rgb = self.get_fused_pcd()
        np.save("vis/pcd_" + str(self.drops) + ".npy", pcd)
        np.save("vis/pcd_rgb_" + str(self.drops) + ".npy", rgb)

    def get_fused_pcd(self):
        rgbs = []
        pcds = []
        pcd_segs = []

        for i in range(len(self.camera_list)):
            cam = self.camera_list[i]
            cam_rgb, cam_depth, cam_seg = cam.capture()
            cam_pcd, cam_pcd_seg, cam_pcd_ind = cam.get_pointcloud(cam_depth, cam_seg)
            rgbs.append(cam_rgb.reshape(-1, 3)[cam_pcd_ind])
            pcds.append(cam_pcd)
            pcd_segs.append(cam_pcd_seg)

        pcd = np.concatenate(pcds, axis=0)  # Fuse point clouds by simply stacking them
        rgb = np.concatenate(rgbs, axis=0)  # Optionally, get colors for each point
        pcd_segs = np.concatenate(pcd_segs, axis=0)

        return pcd, pcd_segs, rgb

    def get_end_effector_pose(self):
        pos, ori = self.client_id.getLinkState(
            self.robot_id, self.end_effector, computeForwardKinematics=1
        )[:2]
        pose = np.array([*pos, *ori])

        return pose

    def plan_motion(self):
        pass

    def quaternion_to_rotation_matrix(self, quat):
        """
        Convert a quaternion to a rotation matrix.

        :param q: Quaternion [w, x, y, z]
        :return: 3x3 rotation matrix
        """
        # w, x, y, z = quat
        # rotation_matrix = np.array([[1 - 2*y**2 - 2*z**2,  2*x*y - 2*z*w,        2*x*z + 2*y*w],
        #                             [2*x*y + 2*z*w,        1 - 2*x**2 - 2*z**2,  2*y*z - 2*x*w],
        #                             [2*x*z - 2*y*w,        2*y*z + 2*x*w,        1 - 2*x**2 - 2*y**2]])

        mat = np.array(self.client_id.getMatrixFromQuaternion(quat))
        rotation_matrix = np.reshape(mat, (3, 3))

        return rotation_matrix

    def pose_to_transformation(self, pose):
        pos = pose[:3]
        quat = pose[3:]

        rotation_matrix = self.quaternion_to_rotation_matrix(quat)

        transform = np.zeros((4, 4))
        transform[:3, :3] = rotation_matrix.copy()
        transform[:3, 3] = pos.copy()
        transform[3, 3] = 1

        return transform

    def transformation_to_pose(self, T):
        trans = T[:3, 3]  # Extract translation (3x1 vector)
        rot = T[:3, :3]  # Extract rotation (3x3 matrix)
        quat = R.from_matrix(rot).as_quat()  # Convert to quaternion (w, x, y, z)

        pose = np.append(trans, quat)

        return pose

    def invert_transform(self, T):
        R = T[:3, :3]
        t = T[:3, 3]
        
        T_inv = np.eye(4)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -R.T @ t

        return T_inv
    
    def draw_frame(self, transform, scale_factor=0.2):
        unit_axes_world = np.array(
            [
                [scale_factor, 0, 0],
                [0, scale_factor, 0],
                [0, 0, scale_factor],
                [1, 1, 1],
            ]
        )
        axis_points = ((transform @ unit_axes_world)[:3, :]).T
        axis_center = transform[:3, 3]

        l1 = self.client_id.addUserDebugLine(
            axis_center, axis_points[0], COLORS["red"], lineWidth=4
        )
        l2 = self.client_id.addUserDebugLine(
            axis_center, axis_points[1], COLORS["green"], lineWidth=4
        )
        l3 = self.client_id.addUserDebugLine(
            axis_center, axis_points[2], COLORS["blue"], lineWidth=4
        )

        frame_id = [l1, l2, l3]

        return frame_id[:]

    def remove_frame(self, frame_id):
        for id in frame_id:
            self.client_id.removeUserDebugItem(id)