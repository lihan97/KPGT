import sys
sys.path.append('..')

from src.utils import set_random_seed
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
import transformers
from torch.nn import MSELoss, BCEWithLogitsLoss, SmoothL1Loss
import numpy as np
import random
from src.data.featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES
from src.data.finetune_dataset import MoleculeDataset
from src.data.collator import Collator_tune
from src.model.light import LiGhTPredictor as LiGhT
from src.trainer.finetune_trainer import Trainer, FLAG_Trainer, L2SP_Trainer
from src.trainer.evaluator import Evaluator
from src.trainer.result_tracker import Result_Tracker
from src.model_config import config_dict
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Arguments for training LiGhT")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--n_threads", type=int, default=8)
    
    # Experiment settings
    parser.add_argument("--config", type=str, default='base')
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset_type", type=str, required=True, choices=["classification", 'regression'])
    parser.add_argument("--metric", type=str, required=True, choices=['rocauc', 'ap', 'acc', 'rmse', 'mae', 'r2', 'spearman', 'pearson'])
    parser.add_argument("--split", type=str, required=True)
    
    # Training hyperparameters
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no_norm_label", action='store_true')

    # Model hyperparameters
    parser.add_argument("--n_predictor_layers", type=int, default=2)
    parser.add_argument("--d_predictor_hidden", type=int, default=256)
    
    # Fine-tuning strategies
    # FLAG
    parser.add_argument("--use_flag", action='store_true')
    parser.add_argument("--flag_m", type=int, default=3, choices=[1,2,3,4])
    parser.add_argument("--flag_step_size", type=float, default=0.001, choices=[0.001, 0.003, 0.005, 0.01])
    # LLRD
    parser.add_argument("--use_llrd", action='store_true')
    parser.add_argument("--llrd_decay_coef", type=float, default=0.8, choices=[0.95, 0.9, 0.85, 0.8])
    # L2SP
    parser.add_argument("--use_l2sp", action='store_true')
    parser.add_argument("--l2sp_weight", type=float, default=0.0001, choices=[0.0001, 0.001, 0.01, 0.1])
    # ReInit
    parser.add_argument("--use_reinit", action='store_true')
    parser.add_argument("--reinit_n_layers", type=int, default=3, choices=[1,2,3,4])

    args = parser.parse_args()
    return args

def init_params(module):
    """
    Initialize model parameters
    """
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

def seed_worker(worker_id):
    """
    Set random seed
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_llrd_lr(model, init_lr, decay_coef):
    """
    Get LLRD learning rate
    """
    param_coef = {
        'predictor': init_lr * 1,
        'model.mol_T_layers.11.': init_lr * decay_coef,
        'model.mol_T_layers.10.': init_lr * (decay_coef ** 2),
        'model.mol_T_layers.9.': init_lr * (decay_coef ** 3),
        'model.mol_T_layers.8.': init_lr * (decay_coef ** 4),
        'model.mol_T_layers.7.': init_lr * (decay_coef ** 5),
        'model.mol_T_layers.6.': init_lr * (decay_coef ** 6),
        'model.mol_T_layers.5.': init_lr * (decay_coef ** 7),
        'model.mol_T_layers.4.': init_lr * (decay_coef ** 8),
        'model.mol_T_layers.3.': init_lr * (decay_coef ** 9),
        'model.mol_T_layers.2.': init_lr * (decay_coef ** 10),
        'model.mol_T_layers.1.': init_lr * (decay_coef ** 11),
        'model.mol_T_layers.0.': init_lr * (decay_coef ** 12),
    }
    layer_names = []
    for idx, (name, param) in enumerate(model.named_parameters()):
        layer_names.append(name)
    other = init_lr * (decay_coef ** 13)
    parameters = []
    for idx, (name, param) in enumerate(model.named_parameters()):
        lr = 0
        for key in param_coef.keys():
            if key in name:
                lr = param_coef[key]
                print(name, lr)
        if lr == 0:
            lr = other
        parameters += [{'params': [param], 'lr': lr}]
    return parameters

def get_predictor(d_input_feats, n_tasks, n_layers, predictor_drop, device, d_hidden_feats=None):
    """
    Get predictor
    """
    if n_layers == 1:
        predictor = nn.Linear(d_input_feats, n_tasks)
    else:
        predictor = nn.ModuleList()
        predictor.append(nn.Linear(d_input_feats, d_hidden_feats))
        predictor.append(nn.Dropout(predictor_drop))
        predictor.append(nn.GELU())
        for _ in range(n_layers - 2):
            predictor.append(nn.Linear(d_hidden_feats, d_hidden_feats))
            predictor.append(nn.Dropout(predictor_drop))
            predictor.append(nn.GELU())
        predictor.append(nn.Linear(d_hidden_feats, n_tasks))
        predictor = nn.Sequential(*predictor)
    predictor.apply(lambda module: init_params(module))
    return predictor.to(device)

def finetune(args):
    """
    Fine-tune model
    """
    set_random_seed(args.seed)
    config = config_dict[args.config]
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    g = torch.Generator()
    g.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    collator = Collator_tune(config['path_length'])

    # Dataset loading
    train_dataset = MoleculeDataset(root_path=args.data_path, dataset=args.dataset, dataset_type=args.dataset_type, split_name=f'{args.split}', split='train')
    val_dataset = MoleculeDataset(root_path=args.data_path, dataset=args.dataset, dataset_type=args.dataset_type, split_name=f'{args.split}', split='val')
    test_dataset = MoleculeDataset(root_path=args.data_path, dataset=args.dataset, dataset_type=args.dataset_type, split_name=f'{args.split}', split='test')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_threads, worker_init_fn=seed_worker, generator=g, drop_last=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_threads, worker_init_fn=seed_worker, generator=g, drop_last=False, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_threads, worker_init_fn=seed_worker, generator=g, drop_last=False, collate_fn=collator)

    # Model loading
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
    
    # Fine-tuning settings
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(f'{args.model_path}').items()})
    model.predictor = get_predictor(d_input_feats=config['d_g_feats'] * 3, n_tasks=train_dataset.n_tasks, n_layers=args.n_predictor_layers, predictor_drop=args.dropout, device=device, d_hidden_feats=args.d_predictor_hidden)
    del model.md_predictor
    del model.fp_predictor
    del model.node_predictor
    print("Total model parameters: {:.2f}M".format(sum(x.numel() for x in model.parameters()) / 1e6))

    optimizer = transformers.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = transformers.get_polynomial_decay_schedule_with_warmup(optimizer, args.n_epochs * len(train_dataset) // args.batch_size // 10, args.n_epochs * len(train_dataset) // args.batch_size, 1e-9)

    if args.metric in ['spearman', 'pearson']:
        loss_fn = SmoothL1Loss(reduction='none')
    elif args.metric in ['mae', 'rmse']:
        loss_fn = MSELoss(reduction='none')
    else:
        loss_fn = BCEWithLogitsLoss(reduction='none')

    if args.dataset_type == 'regression' and (not args.no_norm_label):
        mean, std = train_dataset.mean.numpy(), train_dataset.std.numpy()
    else:
        mean, std = None, None

    if args.dataset_type == 'classification':
        evaluator = Evaluator(args.dataset, args.metric, train_dataset.n_tasks)
    else:
        evaluator = Evaluator(args.dataset, args.metric, train_dataset.n_tasks, mean=mean, std=std)

    result_tracker = Result_Tracker(args.metric)
    summary_writer = None

    # Fine-tuning
    if args.use_flag:
        trainer = FLAG_Trainer(args, config['d_g_feats'], optimizer, lr_scheduler, loss_fn, evaluator, result_tracker, summary_writer, device=device, label_mean=torch.from_numpy(mean).to(device) if mean is not None else None, label_std=torch.from_numpy(std).to(device) if std is not None else None)
    elif args.use_llrd:
        parameters = get_llrd_lr(model, args.lr, args.llrd_decay_coef)
        optimizer = transformers.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = transformers.get_polynomial_decay_schedule_with_warmup(optimizer, args.n_epochs * len(train_dataset) // args.batch_size // 10, args.n_epochs * len(train_dataset) // args.batch_size, 1e-9)
        trainer = Trainer(args, optimizer, lr_scheduler, loss_fn, evaluator, result_tracker, summary_writer, device=device, label_mean=torch.from_numpy(mean).to(device) if mean is not None else None, label_std=torch.from_numpy(std).to(device) if std is not None else None)
    elif args.use_l2sp:
        trainer = L2SP_Trainer(args, optimizer, lr_scheduler, loss_fn, evaluator, result_tracker, summary_writer, device=device, label_mean=torch.from_numpy(mean).to(device) if mean is not None else None, label_std=torch.from_numpy(std).to(device) if std is not None else None)
    elif args.use_reinit:
        for layer in range(12 - args.reinit_n_layers, 11):
            model.model.mol_T_layers[layer + 1].initialize()
        trainer = Trainer(args, optimizer, lr_scheduler, loss_fn, evaluator, result_tracker, summary_writer, device=device, label_mean=torch.from_numpy(mean).to(device) if mean is not None else None, label_std=torch.from_numpy(std).to(device) if std is not None else None)
    else:
        trainer = Trainer(args, optimizer, lr_scheduler, loss_fn, evaluator, result_tracker, summary_writer, device=device, label_mean=torch.from_numpy(mean).to(device) if mean is not None else None, label_std=torch.from_numpy(std).to(device) if std is not None else None)

    best_train, best_val, best_test = trainer.fit(model, train_loader, val_loader, test_loader)
    print(f"Training: {best_train:.3f}, Validation: {best_val:.3f}, Test: {best_test:.3f}")

if __name__ == '__main__':
    args = parse_args()
    finetune(args)
