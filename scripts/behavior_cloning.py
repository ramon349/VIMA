from __future__ import annotations
import os
import pdb 
import numpy as np
from einops import rearrange
from vima.utils import *
from vima_bench import make,PARTITION_TO_SPECS
from vima import create_policy_from_ckpt
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from glob import glob 
import pickle as pkl 
from PIL import Image 
from example import prepare_prompt,prepare_obs
import torch 




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

def index_observation(obs_d,index): 
    """ observations have the structure (number_of_steps,h,w,c) 
        we need to return new observations dictionary that is just  (1,h,w,c) 
        index: which time step we will pull from 
    """
    new_dict = dict() 
    for modality in ['rgb','segm']: 
        new_dict[modality]= dict()
        for orientation in ['top','front']: 
            new_dict[modality][orientation] = obs_d[modality][orientation][index] 
    new_dict['ee'] = np.asarray(obs_d['ee'][index])
    return new_dict 
def index_action(action_d,index,device=None): 
    """ Same as obvervations but acting overt the actions dictionary 

    """
    new_dict = dict()
    ##TODO:#Do a similar approach of indixing above where instead of having (num_steps,x,y) we have (x,y) 
    for e in action_d.keys(): 
        if  'position' in e: 
            new_dict[e] = torch.tensor(action_d[e][index,0:2],device=device)
        else: 
            new_dict[e] = torch.tensor(action_d[e][index],device=device)
    return new_dict


def model_train(policy,traj_info,device='cuda:0',opti=None): 
    torch.autograd.set_detect_anomaly(True)
    my_loss =torch.nn.NLLLoss()
    #step our model over a single trajectory
    traj_steps= traj_info['traj_meta']['steps']
    #i make a dummy enviroment just to pull some extra metadata information.  not used elsewhere
    env = make('rearrange',modalities=['segm','rgb'],task_kwargs=PARTITION_TO_SPECS["test"]['placement_generalization']['rearrange'],seed=42,render_prompt=False,display_debug_window=False,hide_arm_rgb=False,record_gui=False)
    env.reset() 
    meta_info = env.meta_info 
    c_step = 0
    inference_cache = dict()  
    inference_cache["obs_tokens"] = []
    inference_cache["obs_masks"] = []
    inference_cache["action_tokens"] = []
    #load the intial observation data 
    prompt,prompt_assets = traj_info['prompt'],traj_info['prompt_assets']
    prompt_token_type, word_batch, image_batch = prepare_prompt(
        prompt=prompt, prompt_assets=prompt_assets, views=["front", "top"]
    )
    #send everything to gpu 
    word_batch = word_batch.to(device)
    image_batch = image_batch.to_torch_tensor(device=device)
    prompt_tokens, prompt_masks = policy.forward_prompt_assembly(
        (prompt_token_type, word_batch, image_batch)
        )
    obs_d = traj_info['obs']
    oracle_action_d = traj_info['action']
    l = torch.tensor([0],device=device)
    for i in range(traj_steps):
        opti.zero_grad() 
        #get the current observation  and preprocess it 
        obs = index_observation(obs_d,c_step)
        obs = add_batch_dim(obs)
        obs = prepare_obs(obs=obs, rgb_dict=None, meta=meta_info).to_torch_tensor(
            device=device
            )
        ################# BEGIN COMPLICATED FORWARD STEP ###########
        obs_token_this_step, obs_mask_this_step = policy.forward_obs_token(obs)
        obs_token_this_step = obs_token_this_step.squeeze(0)
        obs_mask_this_step = obs_mask_this_step.squeeze(0)
        inference_cache["obs_tokens"].append(obs_token_this_step[0].detach())
        inference_cache["obs_masks"].append(obs_mask_this_step[0].detach())
        max_objs = max(x.shape[0] for x in inference_cache["obs_tokens"])
        obs_tokens_to_forward, obs_masks_to_forward = [], []
        obs_tokens_this_env, obs_masks_this_env = [], []
        for idx in range(len(inference_cache["obs_tokens"])):
            obs_this_env_this_step = inference_cache["obs_tokens"][idx]
            obs_mask_this_env_this_step = inference_cache["obs_masks"][idx]
            required_pad = max_objs - obs_this_env_this_step.shape[0]
            obs_tokens_this_env.append(
                any_concat(
                    [
                        obs_this_env_this_step,
                        torch.zeros(
                            required_pad,
                            obs_this_env_this_step.shape[1],
                            device=device,
                            dtype=obs_this_env_this_step.dtype,
                        ),
                    ],
                    dim=0,
                )
            )
            obs_masks_this_env.append(
                any_concat(
                    [
                        obs_mask_this_env_this_step,
                        torch.zeros(
                            required_pad,
                            device=device,
                            dtype=obs_mask_this_env_this_step.dtype,
                        ),
                    ],
                    dim=0,
                )
            )
        obs_tokens_to_forward.append(any_stack(obs_tokens_this_env, dim=0))
        obs_masks_to_forward.append(any_stack(obs_masks_this_env, dim=0))
        if len(obs_tokens_to_forward)>1: 
            obs_tokens_to_forward = [e.detach() for e in obs_tokens_to_forward]
        obs_tokens_to_forward = any_stack(obs_tokens_to_forward, dim=0)
        obs_masks_to_forward = any_stack(obs_masks_to_forward, dim=0)
        obs_tokens_to_forward = obs_tokens_to_forward.transpose(0, 1)
        obs_masks_to_forward = obs_masks_to_forward.transpose(0, 1)
        if c_step ==0: 
            action_tokens_to_forward = None
        else:
            action_tokens_to_forward = any_stack(
                    [any_stack(inference_cache["action_tokens"], dim=0)],
                    dim=0,
                )
            action_tokens_to_forward = action_tokens_to_forward.transpose(0, 1)
        predicted_action_tokens = policy.forward(
            obs_token=obs_tokens_to_forward,
            action_token=action_tokens_to_forward,
            prompt_token=prompt_tokens,
            prompt_token_mask=prompt_masks,
            obs_mask=obs_masks_to_forward,
        )  # (L, B, E)
        #
        predicted_action_tokens = predicted_action_tokens[-1].unsqueeze( 
            0
        )  # (1, B, E) # reading pytorch docs suggest that squeeze operations may be problematic 
        dist_dict = policy.forward_action_decoder(predicted_action_tokens)
        oracle_action = index_action(action_d=oracle_action_d,index=c_step,device=device)  
        with torch.no_grad():
            dec = policy.discretize_action(oracle_action)
        #actions = {k: v.detach().mode() for k, v in dist_dict.items()}  
        action_probs = dict() 
        for k,v in dist_dict.items(): 
            action_probs[k] = list() 
            for  dist in v._dists:
                action_probs[k].append(torch.log(dist.probs))


        ## this al for determing the next action 
        with torch.no_grad(): 
            action_tokens = policy.forward_action_token(oracle_action)  # (1, B, E)
            action_tokens = action_tokens.squeeze(0)  # (B, E)
            inference_cache["action_tokens"].append(action_tokens[0].detach())

        for k in dist_dict.keys(): 
            if 'position' in k:  
                action_vec = dec[k].unsqueeze(0)
                for d in range(2):
                    probs= action_probs[k][d].squeeze(0)
                    if probs.shape[0]>1: 
                        l = l+  my_loss(probs[-1],action_vec[:,d]) 
                    else: 
                        l = l+  my_loss(probs,action_vec[:,d]) 
        c_step = c_step+1 
        l.backward(retain_graph=True) 
        opti.step()
        print(l)
        #TODO use get action function to get the expected function from the oracles action 
        #TODO copy the training loop of the people in stable_baselines 
        #TODO #https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/common/base_class.py#L753
        #TODO calculate the difference between the actions taken by the model and those of the oracle 
        #TODO backproapagate said errorr  
        #TODO step the model loss  using the optimizer 
        #record the loss so we can at least plot it 
    #TODO: maybe return a list of losses 

def main(): 
    from torch.optim import Adam 

    seed = 42
    #TODO: for the purpose of time look up how to unzip a subfolder so you extract the relevant section
    #Path to the trajectories 
    trajectories = glob("/scratch/rlcorrea/vima_v6/rearrange_then_restore/*") 
    #TODO filter the metadata.pkl from trajectories list 
    meta_path = "/scratch/rlcorrea/vima_v6/rearrange_then_restore/metadata.pkl"
    with open(meta_path,'rb') as f: 
        meta = pkl.load(f)
    device = 'cuda:0'
    weight_path ="/home/rlcorrea/CSE574_project_vima/model_weights/2M.ckpt"
    policy = create_policy_from_ckpt(weight_path,device,ignore_statedict=None)
    policy.train() 
    for e in policy.parameters(): 
        e.requries_grad = True 
    for e in policy.xattn_gpt.parameters(): 
        e.requires_grad = True 
    policy = policy.train()
    policy = policy.to(device)
    opti = Adam(policy.parameters(),lr=0.005)
    for traj in trajectories:
        elapsed_steps =0  
        traj_info =  load_trajectory_info(traj)
        model_train(policy=policy,traj_info=traj_info,device='cuda:0',opti=opti)
        


if __name__ == "__main__":
    main()
