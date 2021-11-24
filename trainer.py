import os

from functools import partial
from collections import defaultdict


from torch.nn.functional import cross_entropy
import numpy as np
import torch
import wandb
import pickle
from tqdm import tqdm
from visualization import plot_aligned_timeseries, plot_aligned_structures, plot_predictions

from data import VALIDATION_DATASETS, TRAIN_DATASETS, TEST_DATASETS
from utils import discretize, point_in_circum_to_angle, logit_expectation, get_alignment_metrics, fape_loss, unbatch, kabsch_torch
from mp_nerf.protein_utils import get_protein_metrics
from copy import deepcopy
from einops import repeat, rearrange
from torch_geometric.utils.metric import precision, recall


eps = 1e-7
IGNORE_IDX = -100

class Trainer():

    def __init__(self, config, model, loaders):
        super().__init__()
        self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.config = config

        self.model = model
        self.model.to(self.device)
        wandb.watch(self.model)

        self.loaders = loaders

        self.model_path = os.path.join(wandb.run.dir, 'checkpoint.pt')
        self.config_path = os.path.join(wandb.run.dir, 'config.pkl')
        with open(self.config_path, 'wb') as file:
            pickle.dump(config, file)

        torch.save(self.model.state_dict(), self.model_path)
        print(f'model path: {self.model_path}')
        print(f'config path: {self.config_path}')

        self.best_val_loss   = float('inf')

        self.binners = dict()
        self.binners['distance'] = partial(discretize, start=3, end=config.distance_pred_max_radius,
                                        number_of_bins=config.distance_pred_number_of_bins)
        self.predict_angles = config.predict_angles
        if self.predict_angles:
            self.binners['angles'] = partial(discretize, start=-1, end=1,
                                        number_of_bins=config.angle_pred_number_of_bins)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.accumulation = config.accumulate_every

    def train(self):
        for epoch in range(self.config.max_epochs):
            epoch_metrics = self.evaluate(TRAIN_DATASETS[0], epoch)

            if epoch >= self.config.validation_start:
                if epoch % self.config.validation_check_rate == 0:
                    for split in VALIDATION_DATASETS:
                        epoch_metrics.update(self.evaluate(split, epoch))
                    print('saving model')
                    torch.save(self.model.state_dict(), self.model_path)

            if wandb.run: wandb.log(epoch_metrics)

        print(f'model path: {self.model_path}')
        print(f'config path: {self.config_path}')
        torch.save(self.model.state_dict(), self.model_path)

    def test(self):
        for test_set in TEST_DATASETS:
            test_metrics = self.evaluate(test_set)
            if wandb.run: wandb.log(test_metrics)


    def evaluate(self, split_name, epoch=0):
        is_training = split_name in TRAIN_DATASETS
        is_testing = split_name in TEST_DATASETS

        epoch_metrics = defaultdict(list)
        loader = self.loaders[split_name]

        with torch.set_grad_enabled(is_training):
            with tqdm(enumerate(loader), total=len(loader)) as bar:
                bar.set_description(f'[{epoch}] {split_name}')
                accumulation, accumulated_loss = 0, 0
                for batch_idx, batch in bar:
                    torch.cuda.empty_cache()
                    batch = [b.to(self.device) for b in batch]

                    batch_predictions = self.model(batch, is_training=is_training)

                    (metrics, prediction_alignments,
                        prediction_images) = self.evaluate_predictions(batch, batch_predictions, split_name, batch_idx)

                    loss = metrics['loss'] = (
                        self.config.distogram_coefficient * metrics['fale']
                    )

                    accumulation += 1
                    accumulated_loss += loss / self.accumulation

                    if is_training and accumulation >= self.accumulation:
                        self.optim.zero_grad()
                        accumulated_loss.backward()
                        self.optim.step()
                        accumulation = 0
                        accumulated_loss = 0

                    if is_testing and batch_idx < self.config.num_test_visual_samples:
                        if prediction_images: self.plot_images(prediction_images)

                    for k, v in metrics.items(): epoch_metrics[k].append(v.item() if type(v) is not float else v)

                    if batch_idx % self.config.report_frequency == 0:
                        report = ', '.join([f'{k}={np.mean(epoch_metrics[k][-50:]):.3e}' for k, v in epoch_metrics.items()])
                        print(report)

                    bar.set_postfix(loss = f'{np.mean(epoch_metrics["loss"][-50:]):.3e}')

        epoch_metrics = { f'{split_name} {k}': np.mean(v)
                         for k, v in epoch_metrics.items() }

        return epoch_metrics


    def evaluate_predictions(self, batch, predictions, split, batch_idx):
        metrics, alignments, images = defaultdict(float), dict(), dict()

        should_fetch_results = split in TEST_DATASETS and batch_idx < self.config.num_test_visual_samples

        # ===== EVALUATE DISTOGRAM PREDICTIONS ======
        if 'logit_traj' in predictions:
            distogram_metrics, images, permutations = self.evaluate_distograms(
                batch,
                predictions['logit_traj'],
                should_fetch_results
            )
            metrics.update(distogram_metrics)
        else:
            permutations = torch.zeros(batch.seqs.size(0), device=batch.seqs.device)

        return metrics, alignments, images


    def evaluate_distograms(self, batch, logits_trajectory, fetch_images):
        metrics = defaultdict(int)
        batch_images = defaultdict(list)

        pair, cross_edges_ground = batch[:2], batch[2]
        n, m = [p.encs.size(1) for p in pair]
        device = cross_edges_ground.device

        trajectory = logits_trajectory['vector']
        trajectory_len = trajectory.size(1)
        batch_size = trajectory.size(0)

        predictions = repeat(trajectory, 'b t n m c -> b n m s t c', s=2)
        targets = repeat(cross_edges_ground, 'b n m c s -> b n m s t c', t=trajectory_len)
        mask = repeat(pair[0].node_pad_mask, 'b n -> b n m s', m=m, s=2) & repeat(pair[1].node_pad_mask, 'b m -> b n m s', n=n, s=2)
        mask = mask[..., None] & (torch.norm(targets, dim=-1, p=2) < 20)

        vector_differences = torch.norm(predictions - targets, dim=-1, p=2)
        vector_differences = vector_differences * mask.float()

        vector_differences = vector_differences.sum((1, 2, 4)) / (mask.sum((1, 2, 4)) + 1e-7)
        vector_differences, permuted = vector_differences.min(dim=-1)

        metrics['fale'] = vector_differences.mean()

        permutation_mask = torch.stack((permuted.bool(), ~(permuted.bool())), dim=-1)
        ground_images = rearrange(cross_edges_ground, 'b n m c s -> b s c n m')[permutation_mask]
        predicted_images = rearrange(trajectory, 'b t n m c -> t b c n m')[-1]

        if not fetch_images:
            return metrics, batch_images, permuted

        candidate_images = []
        for dimension_idx in range(3):
            pred_img = predicted_images[:, dimension_idx]
            grnd_img = ground_images[:, dimension_idx]
            candidate_images.extend([grnd_img, pred_img])

        for batch_idx, (chains1, chains2, id) in enumerate(zip(pair[0].chns, pair[1].chns, pair[0].ids)):
            images = [img[batch_idx].detach() for img in candidate_images]
            n, m = (chains1 == chains1[0]).sum(), (chains2 == chains2[0]).sum()
            images = [img[:n, :m].cpu() for img in images]
            batch_images[id].append(images)

        return metrics, batch_images, permuted


    def plot_images(self, images):
        for id, img_trajectory in images.items():
            for imgs in img_trajectory:
                prediction_figure = plot_predictions(imgs)
                if wandb.run:
                    wandb.log({
                        f'{id} at topography prediction': prediction_figure,
                    })
