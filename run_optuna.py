import sys
import os
import wandb
import optuna
os.environ["WANDB_MODE"] = "dryrun"

import sys
import torch

from data import create_dataloaders
from trainer import Trainer
from model import Interactoformer

from utils import random_name

import optuna
import argparse

from math import floor, ceil

from run import config_parser
from data import VALIDATION_DATASETS, TRAIN_DATASETS, TEST_DATASETS

with torch.no_grad():
    torch.cuda.empty_cache()


def get_config(trial, dataset_source="/home/gridsan/kalyanpa/DNAInteract_shared/allan"):
    config, _ = config_parser().parse_known_args()
    
#     # ========================
#     # GENERAL
#     # ========================
    config.debug = False
    config.submit = False
    config.note = "nonote"
    config.report_frequency = 20
    config.seed = 42

    
#     # ========================
#     # DATA
#     # ========================
    config.dataset_source = dataset_source
    config.spatial_clamp = 96

#     # ========================
#     # ARCHITECTURE
#     # ========================
    config.dim = trial.suggest_categorical("dim", [64, 128])
    config.edim = trial.suggest_categorical("edim", [32, 64])

    config.docker_depth = trial.suggest_int("docker_depth", 2, 5)

    total_depth = 11
    remaining_depth = total_depth - config.docker_depth

    cross_encoder_fraction = trial.suggest_float("cross_encoder_fraction", 0.01, 0.7)

    config.cross_encoder_depth = ceil(remaining_depth * cross_encoder_fraction)
    config.encoder_depth = remaining_depth - config.cross_encoder_depth

    if config.encoder_depth == 0:
        config.encoder_depth += 1
        config.cross_encoder_depth -= 1

    config.kernel_size = trial.suggest_int("kernel_size", 3, 8)
    config.num_conv_per_layer = trial.suggest_int("num_conv_per_layer", 1, 2)

    trial.set_user_attr(f"cross_encoder_depth", config.cross_encoder_depth)
    trial.set_user_attr(f"encoder_depth", config.encoder_depth)
    
    config.heads = trial.suggest_categorical("heads", [4, 8])
    config.scalar_key_dim = trial.suggest_categorical("scalar_key_dim", [16, 32])
    config.scalar_value_dim = trial.suggest_categorical("scalar_value_dim", [16, 32])

    # config.point_key_dim = trial.suggest_categorical("point_key_dim", [16, 32])
    # config.point_value_dim = trial.suggest_categorical("point_value_dim", [16, 32])

#     # ITERATION STEPS
    config.unroll_steps = trial.suggest_int("unroll_steps", 15, 20)
    config.eval_steps = 15
    
#     # ========================
#     # OPTIMIZATION
#     # ========================
    
    config.structure_only = 0
    config.distogram_only = 0

#     # OPTIM
    config.lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    config.accumulate_every = 1
    config.batch_size = trial.suggest_categorical("batch_size", [48]) 
    config.max_epochs = 40

#     # ========================
#     # TEST
#     # ========================
    
    config.test_model = "-"
    config.retrain_model = "-"
    config.num_test_visual_samples = 30

    return config 

def objective_function(trial, dataset_source="/home/gridsan/kalyanpa/DNAInteract_shared/allan"):
    config = get_config(trial, dataset_source=dataset_source)
    print(config)
    wandb.init(
        reinit=True,
        name=random_name(),
        config=config,
        project='Interactoformer',
    )
    loaders = create_dataloaders(config)


    model = Interactoformer(config)
    trainer = Trainer(config, model, loaders)
    print('Starting Train')
    trainer.train(optuna_trial=trial)
    metrics = trainer.evaluate(VALIDATION_DATASETS[0])

    print('Starting Test')
    trainer.test()

    print('Done.')
    return metrics['val loss']

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Dock some proteins')
    parser.add_argument("--dataset_source", default="/home/gridsan/kalyanpa/DNAInteract_shared/allan")
    parser.add_argument("--trials", default=50, type=int)

    args = parser.parse_args()

    study = optuna.create_study(direction="maximize")
    objective = lambda trial: objective_function(trial, dataset_source=args.dataset_source)
    study.optimize(objective, n_trials=args.trials)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))