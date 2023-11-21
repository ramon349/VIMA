from __future__ import annotations
import os
import pdb 
import numpy as np
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


import torch.nn.functional as F
def pad_tensors(tensors, padding_mode='constant', pad_value=0):
    max_dims = list() 
    new_tensors = list() 

    for i in range(len(tensors[0].shape)): 
        dim_dif = max([tens.shape[i] for tens in tensors])
        max_dims.append(dim_dif)
    padded_sensors = []
    for tensor in tensors:
        og_shape = tensor.shape  
        padding = [max_dims[i]-og_shape[i] for i in range(len(tensor.shape))] 
        zero_pads = [0 for i in range(len(tensor.shape))] 
        padding = zero_pads + padding
        if sum(padding)==0:
            padded_sensors.append(tensor)
        else:
            padded_sensor = F.pad(tensor, padding, mode=padding_mode, value=pad_value)
            padded_sensors.append(padded_sensor)
    return torch.stack(padded_sensors,dim=0)

def pad_action_tokens(tensors): 
    non_none_tensors = [e for e in tensors if e is not None] 
    num_elem = len(tensors)
    if len(non_none_tensors) ==0: 
       return  torch.zeros(num_elem,1,256) 
    else: 
        prox_shape = non_none_tensors[0].shape 
        return  pad_tensors([None if e is None else e  for e in tensors ])
def pad_prompt_tokens(tensors): 
    tens = [ e.unsqueeze(0) for e in tensors] 
    return pad_tensors(tens)
    
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
    for traj_ids,observations,actions ,prompt_infos ,trajectory_steps,meta_infos in data_loader: 
       num_batches = len(prompt_infos)
       opti.zero_grad()
       total_loss =0 
       prompt_tokens = list() 
       prompt_masks = list() 
       actions_forward_l = list() 
       obs_tokens_l = list() 
       obs_mask_to_forward_l = list() 
       for  i in range(num_batches): 
           b_sample =  prompt_infos[i]
           prompt_token_t, prompt_m  = process_prompt_tokens(policy,prompt_token_type=b_sample[0],word_batch=b_sample[1],image_batch=b_sample[2],device=device)
           prompt_tokens.append(prompt_token_t)
           prompt_masks.append(prompt_m)
           token_proc_input = {'obs':observations[i],'meta_info':meta_infos[i]}
           action_toks_to_forward,obs_tokens,obs_mask_to_forward = gen_individual_tokens(policy=policy,inputs=token_proc_input,inference_cache=batched_inference_cahce[traj_ids[i]],device=device)
           actions_forward_l.append(action_toks_to_forward)
           obs_tokens_l.append(obs_tokens)
           obs_mask_to_forward_l.append(obs_mask_to_forward)
       prompt_masks = pad_prompt_tokens(prompt_masks)
       prompt_tokens = pad_prompt_tokens(prompt_tokens)
       prompt_tokens = prompt_tokens.squeeze(1) 
       prompt_tokens = prompt_tokens.squeeze(-2)
       obs_tokens_l = pad_tensors(obs_tokens_l)
       actions_forward_l= pad_action_tokens(actions_forward_l)
       obs_mask_to_forward_l = pad_tensors(obs_mask_to_forward_l)
       predicted_action_tokens = policy.forward(
            obs_token=obs_tokens_l,
            action_token=actions_forward_l,
            prompt_token=prompt_tokens,
            prompt_token_mask=prompt_masks,
            obs_mask=obs_mask_to_forward_l,
        )  # (L, B, E)
       pdb.set_trace()
       loss = simple_forward(policy=policy,inputs=input_d,inference_cache=mini_traj_cache,meta_info=meta_info,device=device)
       total_loss = total_loss + loss
       writer.add_scalar("batch_loss",total_loss,global_step=step_counter)
       step_counter +=1 
       print(f" On Step {step_counter} loss: {total_loss:0.4} cache has len: {len(batched_inference_cahce)}",end='\r')
       #TODO add model saving here after some time
       total_loss.backward() 
       opti.step()
       if len(batched_inference_cahce) >= 20:
           print("Clearing out cache")
           clear_cache(batched_inference_cahce)

def process_prompt_tokens(policy,prompt_token_type,word_batch,image_batch,device): 
    #send everything to gpu 
    word_batch = word_batch.to(device)
    image_batch = image_batch.to_torch_tensor(device=device)
    prompt_tokens, prompt_masks = policy.forward_prompt_assembly(
        (prompt_token_type, word_batch, image_batch)
        )
    return prompt_tokens,prompt_masks

def action_to_device(action,device= None): 
    for e in action.keys(): 
        action[e] = action[e].to(device)
def gen_individual_tokens(policy,inputs,inference_cache,device): 
    c_step = len(inference_cache['obs_tokens'])
    meta_info = inputs['meta_info']
    obs = inputs['obs']
    obs = add_batch_dim(obs)
    obs = prepare_obs(obs=obs, rgb_dict=None, meta=meta_info).to_torch_tensor(
        device=device
        )
    #this forward is to just encode stuff 
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
    obs_tokens_to_forward = obs_tokens_to_forward.transpose(0, 1).squeeze(0)
    obs_masks_to_forward = obs_masks_to_forward.transpose(0, 1).squeeze(0)
    if c_step ==0: 
        action_tokens_to_forward = None
    else:
        action_tokens_to_forward = any_stack(
                [any_stack(inference_cache["action_tokens"], dim=0)],
                dim=0,
            )
        action_tokens_to_forward = action_tokens_to_forward.transpose(0, 1)

    return action_tokens_to_forward,obs_tokens_to_forward,obs_masks_to_forward


def simple_forward(policy,inputs,inference_cache,meta_info,device):
        #get the current observation  and preprocess it 
        ################# BEGIN COMPLICATED FORWARD STEP ###########
        ### END complicated forward step 
        #the models actions are provided as a single embedding 
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
