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
from torch.optim import Adam 
from vima.policy.vima_policy  import VIMAPolicy 
from pathlib import Path 
from torch.utils.tensorboard import SummaryWriter
from vima.trajectory.trajectory_dataset import TrajectoryLoader
from collections import defaultdict

class LruCacheTrajectories(): 
    def __init__(self,capacity=10): 
        self.store_dict = defaultdict(init_empty_cache_dict)
        self.capacity = capacity 


def init_empty_cache_dict(): 
    new_d = dict() 
    new_d['obs_tokens']=list() 
    new_d['obs_masks'] = list() 
    new_d['action_tokens'] = list() 
    new_d['num_steps']= 0 
    return new_d

def clear_cache(inference_cache): 
    num_to_del = 0 
    keys_to_check = list(inference_cache.keys() )
    for k in keys_to_check: 
        history = inference_cache[k] 
        if len(history['action_tokens']) >= history['num_steps']: 
            num_to_del +=1
            del inference_cache[k]  
    return num_to_del

def model_train(policy,data_loader=None,device='cuda:0',opti=None,writer=None,total_counter=0): 
    """ We train a model over the course of a single trajectory  
    policy  : Our Vima Policy model 
    traj_info:  Dictionary containing RGB Images, Trjaectory information etc 
    device:  GPU/CPU device used for training 
    opti: Optimizer used during training 
    writer: SummaryWriter Instane used for logging rewards and other metrics 
    total_counter:  this just counts which trajectory we are on. Used to  log entries in tensorboard
    """ 
    #i make a dummy enviroment just to pull some extra metadata information. #TODO change hardcoded values  
    env = make('rearrange',modalities=['segm','rgb'],task_kwargs=PARTITION_TO_SPECS["test"]['placement_generalization']['rearrange'],seed=42,render_prompt=False,display_debug_window=False,hide_arm_rgb=False,record_gui=False)
    env.reset() 
    meta_info = env.meta_info 
    batched_inference_cahce = defaultdict(init_empty_cache_dict)
    step_counter = 0 
    for traj_ids,observations,actions ,prompt_infos ,trajectory_steps, in data_loader: 
       num_batches = len(prompt_infos)
       opti.zero_grad()
       total_loss =0 
       for batch in range(num_batches):
            traj_id = traj_ids[batch]
            prompt_token_type = prompt_infos[batch][0]
            word_batch = prompt_infos[batch][1]
            image_batch = prompt_infos[batch][2]
            obs = observations[batch]
            oracle_action = actions[batch] 
            #send everything to gpu 
            word_batch = word_batch.to(device)
            image_batch = image_batch.to_torch_tensor(device=device)
            prompt_tokens, prompt_masks = policy.forward_prompt_assembly(
                (prompt_token_type, word_batch, image_batch)
                )
            traj_inf_cache = batched_inference_cahce[traj_id]
            traj_inf_cache['num_steps'] =trajectory_steps[batch] 
            input_d = {
                'prompt_tokens':prompt_tokens,
                'prompt_masks':prompt_masks,
                'obs':obs,
                'oracle_action':oracle_action,
            }
            loss = simple_forward(policy=policy,inputs=input_d,inference_cache=traj_inf_cache,meta_info=meta_info,device=device)
            total_loss = total_loss + loss
       writer.add_scalar("batch_loss",total_loss,global_step=step_counter)
       step_counter +=1 
       print(f" On Step {step_counter} loss: {total_loss:0.2} cache has len: {len(batched_inference_cahce)}")
       #TODO add model saving here after some time
       total_loss.backward() 
       opti.step()
       if len(batched_inference_cahce) >= 20:
           print("Clearing out cache")
           num_to_del = clear_cache(batched_inference_cahce)
           print(f"I would delete {num_to_del} entries")
def action_to_device(action,device= None): 
    for e in action.keys(): 
        action[e] = action[e].to(device)
def simple_forward(policy,inputs,inference_cache,meta_info,device):
        #get the current observation  and preprocess it 
        obs= inputs['obs'] 
        oracle_action = inputs['oracle_action']
        action_to_device(oracle_action,device=device)
        c_step = len(inference_cache['obs_tokens'])
        prompt_tokens = inputs['prompt_tokens']
        prompt_masks = inputs['prompt_masks']
        obs = inputs['obs']
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
        predicted_action_tokens = predicted_action_tokens[-1].unsqueeze(0)
        # We discretize them into actual probabilities  of x,y,z placements 
        dist_dict = policy.forward_action_decoder(predicted_action_tokens)
        dec = policy.discretize_action(oracle_action)
        action_probs = dict() 
        for k,v in dist_dict.items(): 
            action_probs[k] = list() 
            for  dist in v._dists:
                acti_prob = dist.probs
                action_probs[k].append(acti_prob)
        ## this al for determing the next action 
        action_tokens = policy.forward_action_token(oracle_action)  # (1, B, E)
        action_tokens = action_tokens.squeeze(0)  # (B, E)
        inference_cache["action_tokens"].append(action_tokens) # we append the oracles action not the models action to history 
        #why do we do this? because you want to condition on correct actions not incorrect ones 
        l = 0
        #calcualte the negative log likelihood loss over the actions taken by the model 
        for k in dist_dict.keys(): 
            for d in range(len(action_probs[k])):
                indiv_prob = action_probs[k][d].view((-1,))
                discrete_index = dec[k][d]
                loss =  calc_nll(indiv_prob,discrete_index) 
                l = l+loss
        return l 

def calc_nll(model_probabilities,true_index):
    criterion = torch.nn.BCELoss() 
    label = torch.zeros(model_probabilities.shape,device='cuda')
    label[true_index]=1
    loss = criterion(model_probabilities,label)
    return loss 


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
    weight_path  =  log_path.replace("logs","weghts/behavior_clone") 
    os.makedirs(weight_path,exist_ok=True)
    weight_path= os.path.join(weight_path,'w.ckpt')
    writer = SummaryWriter(os.path.join(summary_dir,f"version_{new_version}"))
    return writer,weight_path



def main(): 
    seed = 42
    summary_dir ="/home/rlcorrea/CSE574_project_vima/model_logs_demo"
    #Path to the trajectories 
    traj_folder = "/scratch/rlcorrea/vima_v6/rearrange_then_restore/"
    dl = TrajectoryLoader(
        traj_folder=traj_folder,
        traj_name="rearrange_then_restore",
        n_workers=2,
        batch_size=2,
        n_epochs=1,
        max_queue_size=20,
    )
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
    model_train(policy=policy,data_loader = dl,device='cuda:0',opti=opti,writer=writer,total_counter=0)
    torch.save(policy.state_dict(),weight_path)

if __name__ == "__main__":
    main()
