import numpy as np
import yaml
import copy
import matplotlib.pyplot as plt

from pybullet_scene import Scene

with open("config/global_config.yaml", "r") as config:
    args = yaml.load(config, Loader=yaml.FullLoader)

env = Scene(args, gui=True)
obj_id = env.objects['object']

# force_vec = np.array([-500., 0., 0.])
force_vec = np.array([0., 0., 10.])
# force_pt = np.array([0.012, 0.02, -0.02])
force_pt = np.array([0., 0., 0.])
vid_path = "assets/videos/simulation/tomato_soup_can.mp4"

env.move_object(obj_id, force_pt, force_vec, vid_path)

print("")