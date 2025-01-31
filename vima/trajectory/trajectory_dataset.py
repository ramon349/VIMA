from __future__ import annotations
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
from glob import glob
from vima.trajectory.trajectory_util import (
    load_trajectory_info,
    obs_load,
    action_load,
    trajectory_load,
    prepare_prompt,
)
from torch.utils.data import Dataset
import torch
import numpy as np
import pdb
import pickle as pkl
from torch.utils.data import ConcatDataset, DataLoader
from pathlib import Path  


def index_observation(obs_d, index):
    """observations have the structure (number_of_steps,h,w,c)
    we need to return new observations dictionary that is just  (1,h,w,c)
    index: which time step we will pull from
    """
    new_dict = dict()
    for modality in ["rgb", "segm"]:
        new_dict[modality] = dict()
        for orientation in ["top", "front"]:
            new_dict[modality][orientation] = obs_d[modality][orientation][index]
    new_dict["ee"] = np.asarray(obs_d["ee"][index])
    return new_dict


def index_action(action_d, index, device=None):
    """Same as obvervations but acting overt the actions dictionary"""
    new_dict = dict()
    ##TODO:#Do a similar approach of indixing above where instead of having (num_steps,x,y) we have (x,y)
    for e in action_d.keys():
        if "position" in e:
            new_dict[e] = torch.tensor(action_d[e][index, 0:2], device=device)
        else:
            new_dict[e] = torch.tensor(action_d[e][index], device=device)
    return new_dict


import random
from multiprocessing import Process, Queue, Event


def data_proc_worker(task_queue, output_queue, quit_worker_event):
    """this is the function of our worker processess. It will load one sample for our batch.
    in our use case it will load  the contents of the trajectories and return them one step at a time
    task_queue: will have the task ids we need to obtain
    output_queue: is where we will write our data to
    quit_worker_event: some function to deal with things going sour
    """
    while True:
        task = task_queue.get()
        if task is None:  # needed in case queue becomes empty
            break
        traj_id, traj_path = task
        try: 
            traj_dict = load_trajectory_info(traj_path)
        except: 
           print(f"Skipping due to load failure")
           continue  
        traj_meta = traj_dict["traj_meta"]
        num_steps = traj_meta["steps"]
        prompt = traj_dict["prompt"]
        prompt_assets = traj_dict["prompt_assets"]
        actions = traj_dict["action"]
        obs = traj_dict["obs"]
        # lets do some helpful preprocessing here
        # prompt_token_type, word_batch,image_batch
        prommpt_info = prepare_prompt(
            prompt=prompt, prompt_assets=prompt_assets, views=["front", "top"]
        )
        for i in range(num_steps):
            if quit_worker_event.is_set():
                break
            c_obs = index_observation(obs, i)
            c_act = index_action(actions, i)
            output_queue.put((traj_id, c_obs, c_act, prommpt_info,num_steps,traj_meta))
        if quit_worker_event.is_set():
            break
    output_queue.put(None)  # this is the signal we are done


class TrajectoryLoader:
    def __init__(
        self,
        traj_folder,
        traj_name,
        n_workers=2,
        batch_size=8,
        n_epochs=1,
        max_queue_size=16,
    ):
        self.trajectory_folder = traj_folder
        self.traj_name = traj_name
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.max_queue_size = max_queue_size
        # do some setting
        self.set_trajectory_map()
        self.trajectory_ids = list(self.folder_map.keys())
        print(f"We have this many ids: {len(self.trajectory_ids)}")
        # we need to make the dataset repeatble so we will actually repeat the trajectoryids so we can sample from them
        self.trajectories_2_sample = list()

        for i in range(self.n_epochs):
            c_traj = self.trajectory_ids.copy()
            random.shuffle(c_traj)
            self.trajectories_2_sample += c_traj
        # we now specify the task queue to be consumed
        self.task_queue = Queue()
        self.n_steps_processed = 0  # TODO is this samples or epochs
        for i, traj_id in enumerate(self.trajectories_2_sample):
            queue_input = (traj_id, self.folder_map[traj_id])
            self.task_queue.put(queue_input)  # this makes it a single element tuple
        for _ in range(n_workers):
            self.task_queue.put(None)
        self.output_queues = [
            Queue(maxsize=self.max_queue_size) for _ in range(n_workers)
        ]
        self.quit_workers_event = Event()
        self.processes = list()
        for output_queue in self.output_queues:
            my_proc = Process(
                target=data_proc_worker,
                args=(self.task_queue, output_queue, self.quit_workers_event),
                daemon=True,
            )
            self.processes.append(my_proc)
        for procs in self.processes:
            procs.start()

    def __iter__(self):
        return self

    def __next__(self):
        """This samples the NEXT BATCH TO USE"""
        batch_observations = []
        batch_actions = []
        batch_episode_id = []
        batch_prompt_info = []
        num_steps = []
        meta_infos = []
        for i in range(self.batch_size):
            try: 
                workitem = self.output_queues[self.n_steps_processed % self.n_workers].get(
                    timeout=10
                )
            except: 
                workitem=None
            if workitem is None:
                raise StopIteration()
            traj_id, observation, action, prompt_info,num_step,meta_info  = workitem
            batch_observations.append(observation)
            batch_actions.append(action)
            batch_episode_id.append(traj_id)
            batch_prompt_info.append(prompt_info)
            num_steps.append(num_step)
            meta_infos.append(meta_info)
            self.n_steps_processed += 1
        return batch_episode_id, batch_observations, batch_actions, batch_prompt_info,num_steps,meta_infos

    def __del__(self):
        for proc in self.processes:
            proc.terminate()
            proc.join()

    def set_trajectory_map(self):
        folder = self.trajectory_folder
        traj_folders = sorted([str(e) for e in Path(folder).glob("*/*")])
        folder_map = dict()
        folder_map = {
            "_".join(e.split("/")[-2:]): e for e in traj_folders if not e.endswith(".pkl")
        } 
        self.folder_map = folder_map


def main():
    traj_folder = "/scratch/rlcorrea/vima_v6/rotate/*"
    good_names = list() 
    for e in sorted(glob(traj_folder))[0:500]: 
        try: 
            smaple = load_trajectory_info(e)  
            good_names.append(e) 
        except: 
            print('failed')
    with open('good_names.txt','w') as f: 
        for e in good_names:
            print(e,file=f)

    # filter out the metadata.pkl from our list of trajectory folders

if __name__ == "__main__":
    main()


# useful stuff for future
# meta_file  = os.path.join(folder,'metadata.pkl')
