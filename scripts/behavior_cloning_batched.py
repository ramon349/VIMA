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
from pathlib import Path 
from torch.utils.tensorboard import SummaryWriter
from trajectory_dataset import traj_object


def model_train(policy,traj_info,device='cuda:0',opti=None,writer=None,total_counter=0): 
    """ We train a model over the course of a single trajectory  
    policy  : Our Vima Policy model 
    traj_info:  Dictionary containing RGB Images, Trjaectory information etc 
    device:  GPU/CPU device used for training 
    opti: Optimizer used during training 
    writer: SummaryWriter Instane used for logging rewards and other metrics 
    total_counter:  this just counts which trajectory we are on. Used to  log entries in tensorboard
    """ 
    #how many steps to take in our trajectory 
    traj_steps= traj_info['traj_meta']['steps']
    #i make a dummy enviroment just to pull some extra metadata information. #TODO change hardcoded values  
    env = make('rearrange',modalities=['segm','rgb'],task_kwargs=PARTITION_TO_SPECS["test"]['placement_generalization']['rearrange'],seed=42,render_prompt=False,display_debug_window=False,hide_arm_rgb=False,record_gui=False)
    env.reset() 
    meta_info = env.meta_info 
    c_step = 0
    inference_cache = dict() # is used to maintained a list of  observations and actions 
    #TODO: i do not understand just yet what this does 
    inference_cache["obs_tokens"] = []
    inference_cache["obs_masks"] = []
    inference_cache["action_tokens"] = []
    #load the intial observation data 
    prompt,prompt_assets = traj_info['prompt'],traj_info['prompt_assets']
    #START THE FORWARD LOOP HERE  

    #pre-process the general prompt 
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
    total_l = 0 
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
        ### END complicated forward step 
        #the models actions are provided as a single embedding 
        predicted_action_tokens = policy.forward(
            obs_token=obs_tokens_to_forward,
            action_token=action_tokens_to_forward,
            prompt_token=prompt_tokens,
            prompt_token_mask=prompt_masks,
            obs_mask=obs_masks_to_forward,
        )  # (L, B, E)
        predicted_action_tokens[-1].view((1,1,256)) 
        pred_x,pred_y = predicted_action_tokens[-1].shape 
        predicted_action_tokens = predicted_action_tokens.view(1,pred_x,pred_y)
        # We discretize them into actual probabilities  of x,y,z placements 
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
        inference_cache["action_tokens"].append(oracle_action) # we append the oracles action not the models action to history 
        #why do we do this? because you want to condition on correct actions not incorrect ones 
        # Loss calculation TODO: Make this into a function that returns a singular value. Also make it use negative log likelihood
        # the ground truth is 
        #  arm0-position: [pos0,pos1]  <-- this is the  ideal ground truth positions  the values each range from 0-50
        #the model outputs probabilities such tht 
        #  action_probs[arm0-position_pred] : [pos0:(1x50),pos1:(1x50)] et 
        #our task is almost treated as predict the  next position given what the model has seen 
        #so we have posiiton labels and position probabilities you can apply the negative log likelihood to that. 
        l = 0
        for k in dist_dict.keys(): 
            for d in range(len(action_probs[k])):
                indiv_prob = action_probs[k][d].view((-1,))
                discrete_index = dec[k][d]
                l = l+  -1*indiv_prob[discrete_index]
        c_step =0 
        l.backward(retain_graph=False)
        total_l += l.detach()   
        opti.step()
        opti.zero_grad()
    writer.add_scalar("train_step",total_l/traj_steps,global_step=total_counter)

def init_summary_writter(summary_dir): 
    """ initialized the tensorboard logger
        - summary_dir is where the tensorboard logs should be store. will have format dir/version_{d} . only specify the dir part
        - will make a weights directory by swapping the word logs with weights. This is suboptimal but ok 

    """
    
    paths = [e for e in Path(summary_dir).rglob("version_*")] 
    if len(paths)>0: 
        max_version = max([ int(str(e).split("_")[-1]) for e in Path(summary_dir).rglob("version_*")])
    else: 
        max_version = -1
    new_version = max_version +1
    log_path = os.path.join(summary_dir,f"version_{new_version}")
    weight_path  =  log_path.replace("logs","weihgts") 
    os.makedirs(weight_path,exist_ok=True)
    weight_path= os.path.join(weight_path,'w.ckpt')
    writer = SummaryWriter(os.path.join(summary_dir,f"version_{new_version}"))
    return writer,weight_path



def main(): 
    seed = 42
    summary_dir ="/home/rlcorrea/CSE574_project_vima/model_logs"
    #Path to the trajectories 
    trajectories = glob("/scratch/rlcorrea/vima_v6/rearrange_then_restore/*") 
    #filter out the metadata.pkl from our list of trajectory folders 
    trajectories = [e for e in trajectories if not e.endswith('.pkl')]
    meta_path = "/scratch/rlcorrea/vima_v6/rearrange_then_restore/metadata.pkl"
    device = 'cuda:0'
    #This are the  parameters for the  model used in the 2M configuration 
    vima_config = {'embed_dim': 256, 'xf_n_layers': 1, 'sattn_n_heads': 8, 'xattn_n_heads': 8}
    policy =  VIMAPolicy(**vima_config) 
    weight_path = "/home/rlcorrea/CSE574_project_vima/model_weights/2M.ckpt"
    ckpt = torch.load(weight_path,map_location=device) 
    #load the pretrained model except for the policy agents weight. The action prediction is handeled by the cross attention_gpt 
    policy.load_state_dict({k.replace('policy.',""):v for k,v in ckpt['state_dict'].items() if 'xattn_gpt' not in k},strict=False)
    policy = policy.train()
    writer,weight_path = init_summary_writter(summary_dir)
    #make it so only the cross attention component is trainable 
    for n,e in policy.named_parameters(): 
        e.requires_grad = False 
    for n,e in policy.xattn_gpt.named_parameters(): 
        e.requires_grad = True 
    policy = policy.to(device)
    opti = Adam(policy.parameters(),lr=0.1)
    total_counter= 0 
    for i,traj in enumerate(trajectories):
        elapsed_steps =0  
        traj_info =  load_trajectory_info(traj)
        model_train(policy=policy,traj_info=traj_info,device='cuda:0',opti=opti,writer=writer,total_counter=total_counter)
        total_counter +=1 
        if (i%100) ==0:
            torch.save(policy.state_dict(),weight_path)

if __name__ == "__main__":
    main()
