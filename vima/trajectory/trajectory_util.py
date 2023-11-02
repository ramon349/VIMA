import pickle as pkl 
import os 
from PIL import Image 
from vima.utils import * 
from einops import rearrange 
import numpy as np 
def load_trajectory_info(traj_path):
    """ Trajectories are stored in a hierachical fashion 
    traj_directory/obs.pkl  <--  These are the rgb images representing the scene 
    traj_directory/action.pkl  <--  These are the actions taken by the oracle 
    traj_directory/trajectory.pkl  <-- contains number of steps, prmpt, prompt assets etc. useful stuff
    """
    #load  observations 
    with open(os.path.join(traj_path, "obs.pkl"), "rb") as f:
        obs = pkl.load(f)
    #parse them as front and top trajectories 
    rgb_dict = {"front": [], "top": []}
    #the frames will consist of the initial state and then + actions taken by model 
    n_rgb_frames = len(os.listdir(os.path.join(traj_path, f"rgb_front")))
    #load all the frames 
    for view in ["front", "top"]:
        for idx in range(n_rgb_frames):
            #load individual images into a dictionary 
            rgb_dict[view].append(
                rearrange(
                    np.array(
                        Image.open(os.path.join(traj_path, f"rgb_{view}", f"{idx}.jpg")),
                        copy=True,
                        dtype=np.uint8,
                    ),
                    "h w c -> c h w",
                )
            )
    rgb_dict = {k: np.stack(v, axis=0) for k, v in rgb_dict.items()}
    # add the rgb  representation to the observation dict 
    obs['rgb'] = rgb_dict
    #load actions to take 
    with open(os.path.join(traj_path, "action.pkl"), "rb") as f:
        action = pkl.load(f)
    with open(os.path.join(traj_path, "trajectory.pkl"), "rb") as f:
        traj_meta = pkl.load(f)
    #load the prompt the model would see 
    prompt = traj_meta.pop("prompt")
    #these are assets needed for "rendering" by the model 
    prompt_assets = traj_meta.pop("prompt_assets")
    return  {'prompt':prompt,'prompt_assets':prompt_assets,'traj_meta':traj_meta,'action':action,'obs':obs}
