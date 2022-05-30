import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse 

import sys
sys.path.append("..")
from src.utils import set_random_seed
from src.data.featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES
from src.data.finetune_dataset import MoleculeDataset
from src.data.collator import Collator_tune
from src.model.light import LiGhTPredictor as LiGhT
from src.model_config import config_dict

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    return args


def extract_features(args):
    config = config_dict[args.config]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    collator = Collator_tune(config['path_length'])
    mol_dataset = MoleculeDataset(root_path=args.data_path, dataset = args.dataset, dataset_type=None)
    loader = DataLoader(mol_dataset, batch_size=32, shuffle=False, num_workers=8, drop_last=False, collate_fn=collator)
    model = LiGhT(
        d_node_feats=config['d_node_feats'],
        d_edge_feats=config['d_edge_feats'],
        d_g_feats=config['d_g_feats'],
        d_hpath_ratio=config['d_hpath_ratio'],
        n_mol_layers=config['n_mol_layers'],
        path_length=config['path_length'],
        n_heads=config['n_heads'],
        n_ffn_dense_layers=config['n_ffn_dense_layers'],
        input_drop=0,
        attn_drop=0,
        feat_drop=0,
        n_node_types=vocab.vocab_size
        ).to(device)
    model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(args.model_path).items()})
    fps_list = []
    for batch_idx, batched_data in enumerate(loader):
        (_, g, ecfp, md, labels) = batched_data
        ecfp = ecfp.to(device)
        md = md.to(device)
        g = g.to(device)
        fps = model.generate_fps(g, ecfp, md)
        fps_list.extend(fps.detach().cpu().numpy().tolist())
    np.savez_compressed(f"{args.data_path}/{args.dataset}/kpgt_{args.config}.npz", fps=np.array(fps_list))
    print(f"The extracted features were saving at {args.data_path}/{args.dataset}/kpgt_{args.config}.npz")

if __name__ == '__main__':
    set_random_seed(22,1)
    args = parse_args()
    extract_features(args)