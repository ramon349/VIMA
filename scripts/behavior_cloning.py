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
from vima.trajectory.trajectory_util import load_trajectory_info
from torch.optim import Adam 
from vima.policy.vima_policy  import VIMAPolicy 
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
    #step our model over a single trajectory
    #how many steps to take in our trajectory 
    traj_steps= traj_info['traj_meta']['steps']
    #i make a dummy enviroment just to pull some extra metadata information. 
    env = make('rearrange',modalities=['segm','rgb'],task_kwargs=PARTITION_TO_SPECS["test"]['placement_generalization']['rearrange'],seed=42,render_prompt=False,display_debug_window=False,hide_arm_rgb=False,record_gui=False)
    env.reset() 
    meta_info = env.meta_info 
    c_step = 0
    inference_cache = dict()  
    # is used to maintained a list of  observations and actions 
    #TODO: i do not understand what the  purpose of the mask is 
    inference_cache["obs_tokens"] = []
    inference_cache["obs_masks"] = []
    inference_cache["action_tokens"] = []
    #load the intial observation data 
    prompt,prompt_assets = traj_info['prompt'],traj_info['prompt_assets']
    #START THE FORWARD LOOP HERE  

    #pre process prompt
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
        inference_cache = dict()  
        inference_cache["obs_tokens"] = []
        inference_cache["obs_masks"] = []
        inference_cache["action_tokens"] = []
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
        inference_cache["obs_tokens"].append(obs_token_this_step[0])
        inference_cache["obs_masks"].append(obs_mask_this_step[0])
        max_objs = max(x.shape[0] for x in inference_cache["obs_tokens"])
        obs_tokens_to_forward, obs_masks_to_forward = [], []
        obs_tokens_this_env, obs_masks_this_env = [], []
        for idx in range(len(inference_cache["obs_tokens"])):
            obs_this_env_this_step = inference_cache["obs_tokens"][idx]
            obs_mask_this_env_this_step = inference_cache["obs_masks"][idx]
            required_pad = max_objs - obs_this_env_this_step.shape[0]
            #could this be to stack the models?  with different trajectories at once?
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
        #stack the list 
        obs_tokens_to_forward.append(any_stack(obs_tokens_this_env, dim=0))
        obs_masks_to_forward.append(any_stack(obs_masks_this_env, dim=0))

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
        predicted_action_tokens[-1].view((1,1,256)) 
        pred_x,pred_y = predicted_action_tokens[-1].shape 
        predicted_action_tokens = predicted_action_tokens.view(1,pred_x,pred_y)
        #predicted_action_tokens = predicted_action_tokens[-1].unsqueeze( 
        #    0
        #)  # (1, B, E) # reading pytorch docs suggest that squeeze operations may be problematic 
        dist_dict = policy.forward_action_decoder(predicted_action_tokens)
        oracle_action = index_action(action_d=oracle_action_d,index=c_step,device=device)  
        dec = policy.discretize_action(oracle_action)

        action_probs = dict() 
        for k,v in dist_dict.items(): 
            action_probs[k] = list() 
            for  dist in v._dists:
                acti_prob = torch.log(dist.probs)
                action_probs[k].append(acti_prob)


        ## this al for determing the next action 
        action_tokens = policy.forward_action_token(oracle_action)  # (1, B, E)
        action_tokens = action_tokens.squeeze(0)  # (B, E)
        inference_cache["action_tokens"].append(oracle_action)
        #action probs will contain the log probabilities of the actions taken by my model  
        #  
        l = 0
        for k in dist_dict.keys(): 
            for d in range(len(action_probs[k])):
                indiv_prob = action_probs[k][d].view((-1,))
                discrete_index = dec[k][d]
                l = l+  -1*indiv_prob[discrete_index]
        c_step =0 
        l.backward(retain_graph=False) 
        opti.step()
        opti.zero_grad()

def main(): 
    seed = 42
    #Path to the trajectories 
    trajectories = glob("/scratch/rlcorrea/vima_v6/rearrange_then_restore/*") 
    #filter out the metadata.pkl from our list of trajectory folders 
    trajectories = [e for e in trajectories if not e.endswith('.pkl')]
    meta_path = "/scratch/rlcorrea/vima_v6/rearrange_then_restore/metadata.pkl"
    device = 'cuda:0'
    #This are the  parameters for the  model used in the 2M configuration 
    vima_config = {'embed_dim': 256, 'xf_n_layers': 1, 'sattn_n_heads': 8, 'xattn_n_heads': 8}
    policy =  VIMAPolicy(**vima_config) 
    policy = policy.train()
    for n,e in policy.named_parameters(): 
        e.requires_grad = False 
    for n,e in policy.xattn_gpt.named_parameters(): 
        e.requires_grad = True 
    policy = policy.to(device)
    opti = Adam(policy.parameters(),lr=0.005)
    for n,e in  policy.named_parameters(): 
        if  e.requires_grad: 
            print(n)
    for traj in trajectories:
        elapsed_steps =0  
        traj_info =  load_trajectory_info(traj)
        model_train(policy=policy,traj_info=traj_info,device='cuda:0',opti=opti)
        


if __name__ == "__main__":
    main()
