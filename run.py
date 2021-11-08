import sys
import os
import wandb
os.environ["WANDB_MODE"] = "dryrun"

import sys
import torch

from data import create_dataloaders
from trainer import Trainer
from model import Interactoformer

from utils import submit_script


import argparse

if __name__ == '__main__':
    with torch.no_grad():
        torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description='Dock some proteins.')

    # ========================
    # GENERAL
    # ========================
    parser.add_argument('--debug', dest='debug', default=False, action='store_true')
    parser.add_argument('--submit', dest='submit', default=False, action='store_true')
    parser.add_argument('--name', type=str, default='anonymous-glyder')
    parser.add_argument('--note', type=str, default='nonote')
    parser.add_argument('--report_frequency', type=int, default=20)

    # ========================
    # DATA
    # ========================

    parser.add_argument('--dataset_source', type=str, default='../data')
    parser.add_argument('--downsample', type=float, default=1.0)

    parser.add_argument('--spatial_clamp', type=int, default=128) # GPU mem = O(s^2)
    parser.add_argument('--max_seq_len', type=int, default=1024) # RAM mem = O(s)
    parser.add_argument('--num_workers', type=int, default=10)

    parser.add_argument('--sequence_embed', type=int, default=1)

    # ========================
    # ARCHITECTURE
    # ========================
    parser.add_argument('--distance_number_of_bins', type=int, default=32)
    parser.add_argument('--distance_max_radius', type=int, default=25)
    parser.add_argument('--angle_number_of_bins', type=int, default=16)
    parser.add_argument('--gaussian_noise', type=float, default=0)

    parser.add_argument('--checkpoint',type=int, default=1)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--edim', type=int, default=32)

    parser.add_argument('--encoder_depth',type=int, default=3)
    parser.add_argument('--cross_encoder_depth', type=int, default=6)
    parser.add_argument('--docker_depth', type=int, default=0)

    parser.add_argument('--heads', type=int, default=8) # mem, speed = O(heads), O(depth)
    parser.add_argument('--scalar_key_dim',type=int, default=16)
    parser.add_argument('--scalar_value_dim',type=int, default=16)
    parser.add_argument('--point_key_dim', type=int, default=8)
    parser.add_argument('--point_value_dim', type=int, default=8)

    parser.add_argument('--graph_head_dim', type=int, default=16)
    parser.add_argument('--graph_heads', type=int, default=8)


    # ITERATION STEPS
    parser.add_argument('--unroll_steps', type=int, default=25) # O(1) mem
    parser.add_argument('--eval_steps', type=int, default=15)


    # ========================
    # OPTIMIZATION
    # ========================
    parser.add_argument('--structure_only', type=int, default=0)
    parser.add_argument('--distogram_only', type=int, default=0)


    # PREDICTIONS
    parser.add_argument('--contact_cut', type=float, default=10)
    parser.add_argument('--angle_pred_number_of_bins', type=int, default=16)
    parser.add_argument('--distance_pred_number_of_bins', type=int, default=24)
    parser.add_argument('--distance_pred_max_radius', type=float, default=32)

    # OPTIM
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--accumulate_every', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=42)

    # DISTOGRAM LOSS
    parser.add_argument('--topography_loss_coeff', type=float, default=1.0)

    # FAPE LOSS
    parser.add_argument('--arrangement_loss_coeff', type=float, default=1.0)

    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--validation_check_rate', type=int, default=10)
    parser.add_argument('--validation_start', type=int, default=20)

    # FAPE CLAMP
    parser.add_argument('--fape_max_val', type=int, default=20)


    # ========================
    # TEST
    # ========================

    parser.add_argument('--test_model', type=str, default='-')
    parser.add_argument('--retrain_model', type=str, default='-')

    torch.cuda.empty_cache()

    config = parser.parse_args()

    for key, value in dict(vars(config)).items():
        print(f'{key}  =  {value}')

    if config.submit:
        submit_script(os.path.realpath(__file__), os.getcwd(), config)
        exit()

    wandb.init(
        reinit=True,
        name=config.name,
        config=config,
        project='Interactoformer',
    )


    loaders = create_dataloaders(config)

    if config.test_model != '-':
        model = Interactoformer(config)
        model.load_state_dict(torch.load(config.test_model))
        trainer = Trainer(config, model, loaders)
        trainer.test()
    else:
        model = Interactoformer(config)
        if config.retrain_model != '-':
            print(f'Loading weights from {config.retrain_model}')
            model.load_state_dict(torch.load(config.retrain_model))
        trainer = Trainer(config, model, loaders)
        print('Starting Train')
        trainer.train()
        print('Starting Test')
        trainer.test()

    print('Done.')
