from __future__ import annotations
import os
import pdb 
from vima.utils import *
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from example import prepare_obs
import torch 
from vima.policy.vima_policy  import VIMAPolicy 
from pathlib import Path 
from torch.utils.tensorboard import SummaryWriter
from vima.trajectory.trajectory_dataset import TrajectoryLoader
from collections import defaultdict
import torch.nn.functional as F
from torch.optim import AdamW 
from torch.optim.lr_scheduler import CosineAnnealingLR 

def init_empty_cache_dict(): 
    """ Used to store  the history of a trajectory. We use a dictionary to store all relevant components
    NOTE: These entries should be removed frequently to avoid taking up too much ram
    i.e  history_dict = {
        trajectory_1: {'obs_tokens'....}
    }
    """
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
            #i do not like this put it will suffice 
            if  sum([e !=0 for e in padding]) ==1 and padding[-1] !=0: 
                padded_sensor = F.pad(tensor, [0,padding[-1]], mode=padding_mode, value=pad_value)
            else: 
                pdb.set_trace()
                padded_sensor = F.pad(tensor, padding , mode=padding_mode, value=pad_value)
            padded_sensors.append(padded_sensor)
    sanity_check = [padded_sensors[0].shape[i]==padded_sensors[1].shape[i] for i in range(len((padded_sensors[0].shape)))]
    if not all(sanity_check): 
        pdb.set_trace()
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
    tens = [ e.squeeze(1).unsqueeze(0) for e in tensors] 
    return pad_tensors(tens)
def pad_all_tokens(tensors):
    filler_token = None 
    tens = list() 
    filler_token = [e for e in tensors if e is not None][0]
    for e in tensors: 
        if e is None: 
            tens.append(filler_token)
        else: 
            tens.append(e)
    tens = [ e.squeeze(1).unsqueeze(0) for e in tens]  #swap out the batch dim 
    shape_len = [e for e in range(len(tens[0].shape))] #this is the numer of dimensions: 
    for i,dim in enumerate(shape_len):
        largest_dim = max([k.shape[i] for k in tens])
        tens = [ expand_dim(e,i,largest_dim) for e in tens ] 
    new_tens = torch.cat(tens)

    return new_tens
def expand_dim(ten_expand,dim_idx,dim_max): 
    dev = ten_expand.device 
    a_shape = torch.tensor(ten_expand.shape) 
    size_diff = dim_max - a_shape[dim_idx] 
    a_expand_shape = torch.tensor(ten_expand.shape) 
    a_expand_shape[dim_idx] += size_diff 
    new_b_t = torch.zeros(list(a_expand_shape),device=dev)
    slices = [ slice(0,e) for e in a_shape] 
    new_b_t[slices] = ten_expand
    return new_b_t

def max_pad(ten_expand,ten_large): 
    a_shape = torch.tensor(ten_expand.shape)
    b_shape = torch.tensor(ten_large.shape)
    dev = ten_expand.device
    size_diffs = b_shape - a_shape
    new_b = a_shape + size_diffs 
    new_b_t = torch.zeros(list(new_b),device=dev)
    slices = [ slice(0,e) for e in a_shape] 
    new_b_t[slices] = ten_expand
    return new_b_t
    
def model_train(policy,data_loader=None,device='cuda:0',lr_sch=None,opti=None,writer=None,total_counter=0,weight_path=None): 
    """ We train a model over the course of a single trajectory  
    policy  : Our Vima Policy model 
    traj_info:  Dictionary containing RGB Images, Trjaectory information etc 
    device:  GPU/CPU device used for training 
    opti: Optimizer used during training 
    writer: SummaryWriter Instane used for logging rewards and other metrics 
    total_counter:  this just counts which trajectory we are on. Used to  log entries in tensorboard
    """ 
    #i make a dummy enviroment just to pull some extra metadata information. #TODO change hardcoded values  
    batched_inference_cahce = defaultdict(init_empty_cache_dict)
    step_counter = 0 
    warmup = 7000
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
       prompt_masks = pad_all_tokens(prompt_masks)
       prompt_tokens = pad_all_tokens(prompt_tokens)
       obs_tokens_l = pad_all_tokens(obs_tokens_l)
       if not all([e is None for e in actions_forward_l]):
           actions_forward_l= pad_all_tokens(actions_forward_l)
       else:
           actions_forward_l = None 
       obs_mask_to_forward_l = pad_all_tokens(obs_mask_to_forward_l)
       predicted_action_tokens = policy.forward(
            obs_token=obs_tokens_l,
            action_token=actions_forward_l,
            prompt_token=prompt_tokens,
            prompt_token_mask=prompt_masks,
            obs_mask=obs_mask_to_forward_l,
        )  # (L, B, E)
       dist_dict = policy.forward_action_decoder(predicted_action_tokens)
       action_probs = dict() 
       for k,v in dist_dict.items(): 
           action_probs[k] = list() 
           for  dist in v._dists:
                acti_prob = dist.logits[0]
                action_probs[k].append(acti_prob)
       ## this al for determing the next action 
       batch_size = predicted_action_tokens.shape[1] 
       total_loss=0.0
       num_updates = 0 
       for e in range(batch_size): 
           b_probs = dict() 
           for k,v in dist_dict.items(): 
               b_probs[k] = list()
               for i in range(len(action_probs[k])): 
                   b_probs[k].append(action_probs[k][i][e])  #TODO: we get the activatiosn for a particulat batch 
           oracle_action = actions[e]
           #continious version 
           action_to_device(oracle_action,device=device)
           action_tokens = policy.action_encoder({ k: v.clone() for k,v in oracle_action.items() })  # (1, B, E)
           action_tokens = action_tokens.squeeze(0)  # (B, E)
           batched_inference_cahce[traj_ids[e]]["action_tokens"].append(action_tokens) # we append the oracles action not the models action to history 
           dec = policy.discretize_action({k:v.clone() for k,v in oracle_action.items() })
           loss = imitation_loss(b_probs,dec)
           total_loss += loss 
       total_loss = total_loss/batch_size
       writer.add_scalar("batch_loss",total_loss,global_step=step_counter)
       step_counter +=1 
       print(f" On Step {step_counter} loss: {total_loss:0.6}",end='\r')
       total_loss.backward() 
       torch.nn.utils.clip_grad_norm(policy.parameters(),1.0)
       opti.step()
       if step_counter >= warmup: 
           lr_sch.step() 
       if step_counter %100 ==0: 
           update_dict = {} 
           update_dict['state_dict'] = policy.state_dict() 
           update_dict['step_counter'] = step_counter 
           torch.save(update_dict,weight_path)
       for e in opti.param_groups:
           if step_counter < warmup: 
             e['lr'] =  0.0001* (step_counter/warmup)
           else: 
             e['lr'] = 0.0001
       if len(batched_inference_cahce) >= 300:
           clear_cache(batched_inference_cahce)
def imitation_loss(probs,oracle_dict):
    other_crit = torch.nn.CrossEntropyLoss()
    total_loss =0
    num_updates =0 
    for k in probs.keys(): 
        for d in range(len(probs[k])): 
            indiv_prob = probs[k][d]#.view((-1,))
            discrete_index = oracle_dict[k][d] 
            loss = other_crit(indiv_prob,discrete_index)
            total_loss += loss 
            num_updates +=1 
    return total_loss

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
    summary_dir ="/scratch/rlcorrea/model_logs_demo_200M_redo"
    #Path to the trajectories 
    #traj_folder = "/scratch/rlcorrea/vima_v6/rearrange_then_restore/"
    traj_folder = "/scratch/rlcorrea/vima_v6/"
    dl = TrajectoryLoader(
        traj_folder=traj_folder,
        traj_name="rotate",
        #traj_name="rearrange_then_restore",
        n_workers=32,
        batch_size=32,
        n_epochs=5,
        max_queue_size=300,
    )
    device = 'cuda:0'
    vima_config = {'embed_dim': 768, 'xf_n_layers': 11, 'sattn_n_heads': 24, 'xattn_n_heads': 24,'batch_infer':True}
    policy =  VIMAPolicy(**vima_config) 
    og_weight_path = "/home/rlcorrea/CSE574_project_vima/model_weights/200M.ckpt"
    ckpt = torch.load(og_weight_path,map_location=device) 
    policy.load_state_dict({k.replace('policy.',""):v for k,v in ckpt['state_dict'].items() if 'xattn_gpt' not in k},strict=False)
    policy = policy.train()
    writer,weight_path = init_summary_writter(summary_dir)
    print(f"I have weights at :{weight_path}")
    #make it so only the cross attention component is trainable 
    for n,e in policy.named_parameters(): 
        e.requires_grad = False  
    for n,e in policy.xattn_gpt.named_parameters(): 
        e.requires_grad = True 
    policy = policy.to(device)
    opti = AdamW(policy.parameters(),lr=0.0001) # Learning rate used in the paper 
    lr_schedule = CosineAnnealingLR(opti,T_max=17_000) #parameters following original manuscript 
    model_train(policy=policy,data_loader = dl,device=device,opti=opti,lr_sch=lr_schedule,writer=writer,total_counter=0,weight_path=weight_path)
    torch.save(policy.state_dict(),weight_path)
if __name__ == "__main__":
    main()
