import os

from functools import partial
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

import wandb
from tqdm import tqdm
from visualization import plot_aligned_timeseries, plot_aligned_structures, plot_predictions

from data import VALIDATION_DATASETS, TRAIN_DATASETS, TEST_DATASETS
from utils import discretize, point_in_circum_to_angle, logit_expectation, get_alignment_metrics, fape_loss, unbatch, kabsch_torch
from mp_nerf.protein_utils import get_protein_metrics
from copy import deepcopy
from einops import repeat, rearrange
eps = 1e-7
IGNORE_IDX = -100

class Trainer():

    def __init__(self, config, model, loaders):
        super().__init__()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.config = config
        self.model = model
        self.model.to(self.device)
        self.loaders = loaders

        self.model_path = os.path.join(wandb.run.dir, 'checkpoint.pt')
        print(f'model path: {self.model_path}')
        wandb.watch(self.model)

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

            if wandb.run: wandb.log(epoch_metrics)
            print('saving model')
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
                    batch = batch.to(self.device)

                    batch_predictions = self.model(batch, is_training=is_training)

                    (metrics, prediction_alignments,
                        prediction_images) = self.evaluate_predictions(batch, batch_predictions, split_name, batch_idx)

                    loss = metrics['loss'] = (
                        self.config.distogram_coefficient * metrics['distance xentropy'] +
                        self.config.anglegram_coefficient * metrics['angular xentropy'] +
                        self.config.fape_coefficient * metrics['fape']
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
                        if prediction_alignments: self.plot_alignments(batch, prediction_alignments)
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

        if 'logit_traj' in predictions:
            distogram_metrics, images, permutations = self.evaluate_distograms(
                batch,
                predictions['logit_traj'],
                split in TEST_DATASETS and batch_idx < self.config.num_test_visual_samples
            )
            metrics.update(distogram_metrics)
        else:
            permutations = torch.zeros_like(batch.seqs.size(0), device=batch.seqs.device)

        if ('translations' in predictions) and ('rotations' in predictions):
            permutation_mask = torch.stack((permutations.bool(), ~(permutations.bool())), dim=-1)
            translations = rearrange(batch.tgt_crds, 'b ... c s -> b s ... c')[permutation_mask]
            rotations = rearrange(batch.tgt_rots, 'b ... i j s -> b s ... i j')[permutation_mask]

            metrics['fape'] = fape_loss(translations, rotations, predictions['translations'], predictions['rotations'],
                                        mask=batch.node_pad_mask, edge_index=batch.edge_index, max_val=self.config.fape_max_val)

            if not (split in TRAIN_DATASETS):
                structural_metrics, alignments = self.evaluate_arrangement_predictions(
                    batch, translations, predictions['translations']
                )
                metrics.update(structural_metrics)

        return metrics, alignments, images


    def evaluate_distograms(self, batch, logits_trajectory, fetch_images):
        metrics = defaultdict(int)

        # house keeping
        external_edges = (rearrange(batch.chns, 'b s -> b () s') != rearrange(batch.chns, 'b s -> b s ()'))
        external_edges &= batch.edge_pad_mask

        distance_ground = batch.edge_distance
        distance_labels_all = self.binners['distance'](distance_ground)

        batch_images = defaultdict(list)
        trajectory_len = logits_trajectory['distance'].size(1)

        edge_mask = batch.edge_record_mask & external_edges

        # main loss is mean cross_entropy across layers:
        trajectory_mask = repeat(edge_mask, 'b ... -> b t ...', t=trajectory_len)
        trajectory_labels = repeat(distance_labels_all, 'b ... -> b t ...', t=trajectory_len)
        trajectory_labels[~trajectory_mask] = IGNORE_IDX

        # expand logits to consider homomeric symmetry
        distance_logits = repeat(logits_trajectory['distance'], '... l -> ... s l', s=2)
        distance_logits = rearrange(distance_logits, 'b ... l -> b l ...')

        # use solution that provides the best cross entropy for distogram loss
        xentropy = F.cross_entropy(distance_logits, trajectory_labels, reduction='none', ignore_index=IGNORE_IDX)
        xentropy = xentropy.sum((1, 2, 3)) / trajectory_mask.sum((1, 2, 3))[..., None]
        xentropy, permuted = xentropy.min(dim=-1)
        xentropy = xentropy.mean()

        permutation_mask = torch.stack((permuted.bool(), ~(permuted.bool())), dim=-1)
        trajectory_labels = rearrange(trajectory_labels, 'b ... s -> b s ...')[permutation_mask]

        dist_acc = (logits_trajectory['distance'][trajectory_mask].argmax(-1) == trajectory_labels[trajectory_mask]).float().mean()

        metrics[f'distance xentropy'] = xentropy
        metrics[f'distance accuracy'] = dist_acc

        # deal with angles
        if self.predict_angles:
            angles_ground = rearrange(batch.edge_angles, 'b ... s a -> b s ... a')
            angles_ground = angles_ground[permutation_mask]
            angles_labels = repeat(self.binners['angles'](angles_ground), 'b ... -> b t ...', t=trajectory_len)
            angles_logits = rearrange(logits_trajectory['angles'], '... (a l) -> ... l a', a=3)
            angles_labels[~trajectory_mask] = IGNORE_IDX
            angles_labels[trajectory_labels == self.config.distance_pred_number_of_bins - 1] = IGNORE_IDX
            angular_xentropy = F.cross_entropy(rearrange(angles_logits, 'b ... l a -> b l ... a'), angles_labels, ignore_index=IGNORE_IDX)
            angles_labels[trajectory_labels == self.config.distance_pred_number_of_bins - 1] = self.config.angle_pred_number_of_bins / 2
            metrics[f'angular xentropy'] = angular_xentropy

        # Now we evaluate final layer's predictions
        distance_logits = logits_trajectory['distance'][:, -1, ...]
        distance_labels = trajectory_labels[:, -1, ...]

        terminal_xentropy = F.cross_entropy(rearrange(distance_logits, 'b ... l -> b l ...'), distance_labels, ignore_index=IGNORE_IDX)
        metrics['terminal distance xentropy'] = terminal_xentropy

        terminal_acc = (distance_logits[edge_mask].argmax(-1) == distance_labels[edge_mask]).float().mean()
        metrics['terminal distance accuracy'] = terminal_acc

        expectation = logit_expectation(distance_logits)
        pred_contacts = expectation < self.config.contact_cut
        ground_contacts = distance_labels < self.config.contact_cut

        true_positives  = (pred_contacts & ground_contacts & edge_mask).sum()
        false_positives = (pred_contacts & ~ground_contacts & edge_mask).sum()
        false_negatives = (~pred_contacts & ground_contacts & edge_mask).sum()

        contact_precision = (true_positives / (true_positives + false_positives + eps))
        contact_recall = (true_positives / (true_positives + false_negatives + eps))

        metrics[f'contact precision'] =  contact_precision
        metrics[f'contact recall']    =  contact_recall

        if not fetch_images: return metrics, batch_images, permuted

        candidate_images = [distance_labels, expectation, -ground_contacts.float(), -pred_contacts.float()]

        if self.predict_angles:
            for angle_idx in range(3):
                angle_ground = angles_labels[:, 0, ..., angle_idx]
                angle_prediction = logit_expectation(angles_logits[:, 0, ..., angle_idx])
                angle_prediction[distance_logits.argmax(-1) >= self.config.distance_pred_number_of_bins - 2] = self.config.angle_pred_number_of_bins / 2
                candidate_images.extend([angle_ground, angle_prediction])

        for batch_idx, (chains, id, mask) in enumerate(zip(batch.chns, batch.ids, edge_mask)):
            images = [img[batch_idx].detach() for img in candidate_images]
            images = [img[mask] for img in images]
            images = [img[:int(len(img)/2)] for img in images]
            n, m = (chains == chains[0]).sum(), len(chains) - (chains == chains[0]).sum()
            images = [rearrange(img, '(n m) -> n m', n=n, m=m).cpu() for img in images]
            batch_images[id].append(images)

        return metrics, batch_images, permuted


    def evaluate_arrangement_predictions(self, batch, tgt_crds, traj):
        node_mask = batch.node_pad_mask & batch.node_record_mask
        traj = rearrange(traj, 'b n t e -> b t n e')
        batch_size, trajectory_len, _, _ = traj.size()

        alignments = defaultdict(lambda: defaultdict(list))
        metrics = defaultdict(int)

        for id, angles, gnd_wrap, pred_traj, chains, mask in zip(batch.ids, batch.angs,
                                            tgt_crds, traj, batch.chns, node_mask):
            gnd_wrap, angles, chains = gnd_wrap[mask], angles[mask], chains[mask]

            for step, pred_wrap in enumerate(pred_traj):
                pred_wrap = pred_wrap[mask]

                rotation, translation = kabsch_torch(pred_wrap, gnd_wrap)
                align_gnd_coors = torch.einsum('i p , p q -> i q', gnd_wrap, rotation.t()) + translation[None, :]

                alignment_metrics = get_alignment_metrics(align_gnd_coors, pred_wrap)

                for k, metric in alignment_metrics.items():
                    metrics[k] += metric.mean() / len(pred_traj) / batch_size

                alignments[id][f'trajectory'].append((align_gnd_coors.detach().cpu(),
                                                      pred_wrap.detach().cpu(),
                                                      angles.cpu()))

                for k, metric in alignment_metrics.items():
                    metrics[k] += metric.mean() / len(pred_traj) / batch_size

            if len(pred_traj) > 1 and step == len(pred_traj) -1:
                for k, metric in alignment_metrics.items():
                    metrics[f'final_{k}'] += metric.mean() / batch_size

        return metrics, alignments


    def plot_images(self, images):
        for id, img_trajectory in images.items():
            for imgs in img_trajectory:
                prediction_figure = plot_predictions(imgs)
                if wandb.run:
                    wandb.log({
                        f'{id} at topography prediction': prediction_figure,
                    })

    def plot_alignments(self, batch, alignments):
        for id, sequence, chain, mask in zip(batch.ids, batch.str_seqs, batch.chns, batch.node_pad_mask):
            timeseries = alignments[id]
            chain = chain[mask]

            timeseries_fig = plot_aligned_timeseries(
                sequence,
                timeseries['trajectory'],
                (chain == chain[0]).sum().item()
            )

            if wandb.run: wandb.log({ f'{id} timeseries': wandb.Html(timeseries_fig._make_html()) })
