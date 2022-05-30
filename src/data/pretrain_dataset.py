from torch.utils.data import Dataset
import os
import numpy as np
import scipy.sparse as sps
import torch
import dgl.backend as F
class MoleculeDataset(Dataset):
    def __init__(self, root_path):
        smiles_path = os.path.join(root_path, "smiles.smi")
        fp_path = os.path.join(root_path, "rdkfp1-7_512.npz")
        md_path = os.path.join(root_path, "molecular_descriptors.npz")
        with open(smiles_path, 'r') as f:
            lines = f.readlines()
            self.smiles_list = [line.strip('\n') for line in lines]
        self.fps = torch.from_numpy(sps.load_npz(fp_path).todense().astype(np.float32))
        mds = np.load(md_path)['md'].astype(np.float32)
        mds = np.where(np.isnan(mds), 0, mds)
        self.mds = torch.from_numpy(mds)
        self.d_fps = self.fps.shape[1]
        self.d_mds = self.mds.shape[1]        
        
        self._task_pos_weights = self.task_pos_weights()
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        return self.smiles_list[idx], self.fps[idx], self.mds[idx]

    def task_pos_weights(self):
        task_pos_weights = torch.ones(self.fps.shape[1])
        num_pos = torch.sum(torch.nan_to_num(self.fps,nan=0), axis=0)
        masks = F.zerocopy_from_numpy(
            (~np.isnan(self.fps.numpy())).astype(np.float32))
        num_indices = torch.sum(masks, axis=0)
        task_pos_weights[num_pos > 0] = ((num_indices - num_pos) / num_pos)[num_pos > 0]
        return task_pos_weights