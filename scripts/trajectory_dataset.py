from __future__ import annotations
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from glob import glob 
from vima.trajectory.trajectory_util import load_trajectory_info,obs_load,action_load,trajectory_load
from torch.utils.data import Dataset 
import torch 
import numpy as np 
import pdb 
import pickle as pkl 
from torch.utils.data import ConcatDataset,DataLoader 
class traj_object():
    def __init__(self,traj_path,device='cpu'): 
        self.traj_path = traj_path 
        self.traj_id = self.traj_path.split('/')[-1]
        self.traj_loading = load_trajectory_info
        self.actions_is_loaded = False 
        self.obs_is_loaded = False 
        self.device = device 
    def get_traj_meta(self): 
        path = self.traj_path 
        meta_path = os.path.join(path,'trajectory') 
        with open(meta_path,'rb') as f:
            meta = pkl.load(f)
        self.traj_meta = meta
    def get_prompt(self):
        return self._prompt
    def get_prompt_assets(self): 
        return self._prompt_assets
    def get_actions(self):
        if self.actions_is_loaded: 
            return self._actions
        else: 
            self._actions = action_load(self.traj_path) 
            self.actions_is_loaded= True 
        return self._actions
    def get_obs(self):
        if self.obs_is_loaded: 
            return self._obs 
        else:  
            self._obs = obs_load(self.traj_path)
            self.obs_is_loaded= True 
        return self._obs 
    def get_device(self):
        return self._device
    def __len__(self): 
        return self._traj_meta['steps'] 
    def __getitem__(self,idx): 
        c_action,old_actions =  self.index_action(idx,device=self.device)
        c_obs,old_obs = self.index_observation(idx) 
        return c_obs,c_action,old_obs,old_actions,idx
    def index_observation(self,index): 
        """ observations have the structure (number_of_steps,h,w,c) 
            we need to return new observations dictionary that is just  (1,h,w,c) 
            index: which time step we will pull from 
        """
        obs_d = self.get_obs()
        c_dict = dict() 
        old_dict  = dict()
        for modality in ['rgb','segm']: 
            c_dict[modality]= dict() 
            old_dict[modality]= dict() 
            for orientation in ['top','front']: 
                old_dict[modality][orientation] = obs_d[modality][orientation][0:(index)]
                c_dict[modality][orientation] = obs_d[modality][orientation][index]
        old_dict['ee'] = np.asarray(obs_d['ee'][0:(index)])
        c_dict['ee'] = np.asarray(obs_d['ee'][index])
        return c_dict,old_dict
    def index_action(self,index,device=None): 
        """ Same as obvervations but acting overt the actions dictionary 
        """
        action_d = self.get_actions()
        new_dict = dict()
        old_dict = dict()
        for e in action_d.keys(): 
            if  'position' in e: 
                new_dict[e] = torch.tensor(action_d[e][index,0:2],device=device)
                old_dict[e] = torch.tensor(action_d[e][0:(index),0:2],device=device)
            else: 
                new_dict[e] = torch.tensor(action_d[e][index],device=device)
                old_dict[e] = torch.tensor(action_d[e][0:(index)],device=device)
        return new_dict,old_dict
            

if __name__ =="__main__":
    trajectories = glob("/scratch/rlcorrea/vima_v6/rearrange_then_restore/*") 
    #filter out the metadata.pkl from our list of trajectory folders 
    trajectories = [e for e in trajectories if not e.endswith('.pkl')]
    traj = traj_object(trajectories[0])
    traj2 = traj_object(trajectories[2]) 
    merged = ConcatDataset([traj,traj2]) 
    dl = DataLoader(merged,batch_size=4,num_workers=2)
    for e in dl: 
        print('hi')
    

