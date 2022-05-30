import sys
sys.path.append('..')

from src.utils import set_random_seed
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss, BCEWithLogitsLoss
import numpy as np
import random
from src.data.featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES
from src.data.finetune_dataset import MoleculeDataset
from src.data.collator import Collator_tune
from src.model.light import LiGhTPredictor as LiGhT
from src.trainer.scheduler import PolynomialDecayLR
from src.trainer.finetune_trainer import Trainer
from src.trainer.evaluator import Evaluator
from src.trainer.result_tracker import Result_Tracker
from src.model_config import config_dict

import warnings
warnings.filterwarnings("ignore")
def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training LiGhT")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--n_epochs", type=int, default=50)

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--dataset_type", type=str, required=True)
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)

    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--lr", type=float, required=True)

    parser.add_argument("--n_threads", type=int, default=8)
    args = parser.parse_args()
    return args

def get_predictor(d_input_feats, n_tasks, n_layers, predictor_drop, device, d_hidden_feats=None):
    if n_layers == 1:
        predictor = nn.Linear(d_input_feats, n_tasks)
    else:
        predictor = nn.ModuleList()
        predictor.append(nn.Linear(d_input_feats, d_hidden_feats))
        predictor.append(nn.Dropout(predictor_drop))
        predictor.append(nn.GELU())
        for _ in range(n_layers-2):
            predictor.append(nn.Linear(d_hidden_feats, d_hidden_feats))
            predictor.append(nn.Dropout(predictor_drop))
            predictor.append(nn.GELU())
        predictor.append(nn.Linear(d_hidden_feats, n_tasks))
        predictor = nn.Sequential(*predictor)
    predictor.apply(lambda module: init_params(module))
    return predictor.to(device)
def finetune(args):
    config = config_dict[args.config]
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    g = torch.Generator()
    g.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    collator = Collator_tune(config['path_length'])
    train_dataset = MoleculeDataset(root_path=args.data_path, dataset = args.dataset, dataset_type=args.dataset_type, split_name=f'{args.split}', split='train')
    val_dataset = MoleculeDataset(root_path=args.data_path, dataset = args.dataset, dataset_type=args.dataset_type, split_name=f'{args.split}', split='val')
    test_dataset = MoleculeDataset(root_path=args.data_path, dataset = args.dataset, dataset_type=args.dataset_type, split_name=f'{args.split}', split='test')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=args.n_threads, worker_init_fn=seed_worker, generator=g, drop_last=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=args.n_threads, worker_init_fn=seed_worker, generator=g, drop_last=False, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=args.n_threads, worker_init_fn=seed_worker, generator=g, drop_last=False, collate_fn=collator)
    # Model Initialization
    model = LiGhT(
        d_node_feats=config['d_node_feats'],
        d_edge_feats=config['d_edge_feats'],
        d_g_feats=config['d_g_feats'],
        d_fp_feats=train_dataset.d_fps,
        d_md_feats=train_dataset.d_mds,
        d_hpath_ratio=config['d_hpath_ratio'],
        n_mol_layers=config['n_mol_layers'],
        path_length=config['path_length'],
        n_heads=config['n_heads'],
        n_ffn_dense_layers=config['n_ffn_dense_layers'],
        input_drop=0,
        attn_drop=args.dropout,
        feat_drop=args.dropout,
        n_node_types=vocab.vocab_size
    ).to(device)
    # Finetuning Setting
    model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(f'{args.model_path}').items()})
    model.predictor = get_predictor(d_input_feats=config['d_g_feats']*3, n_tasks=train_dataset.n_tasks, n_layers=2, predictor_drop=args.dropout, device=device, d_hidden_feats=256)
    del model.md_predictor
    del model.fp_predictor
    del model.node_predictor
    print("model have {}M paramerters in total".format(sum(x.numel() for x in model.parameters())/1e6))
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(optimizer, warmup_updates=args.n_epochs*len(train_dataset)//32//10, tot_updates=args.n_epochs*len(train_dataset)//32,lr=args.lr, end_lr=1e-9,power=1)
    
    if args.dataset_type == 'classification':
        loss_fn = BCEWithLogitsLoss(reduction='none')
    else:
        loss_fn = MSELoss(reduction='none')
    if args.dataset_type == 'classification':
        evaluator = Evaluator(args.dataset, args.metric, train_dataset.n_tasks)
    else:
        evaluator = Evaluator(args.dataset, args.metric, train_dataset.n_tasks, mean=train_dataset.mean.numpy(), std=train_dataset.std.numpy())
    result_tracker = Result_Tracker(args.metric)
    summary_writer = None
    trainer = Trainer(args, optimizer, lr_scheduler, loss_fn, evaluator, result_tracker, summary_writer, device=device,model_name='LiGhT', label_mean=train_dataset.mean.to(device) if train_dataset.mean is not None else None, label_std=train_dataset.std.to(device) if train_dataset.std is not None else None)
    best_train, best_val, best_test = trainer.fit(model, train_loader, val_loader, test_loader)
    print(f"train: {best_train:.3f}, val: {best_val:.3f}, test: {best_test:.3f}")
if __name__ == '__main__':
    args = parse_args()
    set_random_seed(args.seed)
    finetune(args)
    
    

    
    
    
    
    


