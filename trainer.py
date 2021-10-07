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
from utils import discretize, point_in_circum_to_angle, logit_expectation, get_alignment_metrics, fape_loss, unbatch
from mp_nerf.protein_utils import get_protein_metrics
from copy import deepcopy
from einops import repeat, rearrange
eps = 1e-7


class Trainer():
    def __init__(self, hparams, model, loaders):
        super().__init__()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.config = hparams
        self.model = model
        self.model.to(self.device)
        self.loaders = loaders

        if wandb.run:
            self.model_path = os.path.join(wandb.run.dir, 'checkpoint.pt')
            print(f'model path: {self.model_path}')
            wandb.watch(self.model)

        self.best_val_loss   = float('inf')

        self.angle_binner    = partial(discretize, start=-1, end=1,
                                    number_of_bins=hparams.angle_pred_number_of_bins)
        self.distance_binner = partial(discretize, start=3, end=hparams.distance_pred_max_radius,
                                    number_of_bins=hparams.distance_pred_number_of_bins)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)


    def train(self):
        for epoch in range(self.config.max_epochs):
            epoch_metrics = self.evaluate(TRAIN_DATASETS[0], epoch)

            if epoch > self.config.validation_start:
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
                for batch_idx, batch in bar:
                    torch.cuda.empty_cache()
                    batch = batch.to(self.device)

                    batch_predictions = self.model(batch, is_training=is_training)

                    (metrics, prediction_alignments,
                        prediction_images) = self.evaluate_predictions(batch, batch_predictions, is_training)


                    loss = metrics['loss'] = (
                        self.config.topography_loss_coeff * metrics['dist_xentropy'] +
                        self.config.arrangement_loss_coeff * metrics['fape']
                    )

                    if is_training:
                        self.optim.zero_grad()
                        loss.backward()
                        self.optim.step()

                    if is_testing:
                        if prediction_alignments: self.plot_alignments(batch, prediction_alignments)
                        if prediction_images: self.plot_images(prediction_images)

                    for k, v in metrics.items(): epoch_metrics[k].append(v.item() if type(v) is not float else v)

                    if batch_idx % self.config.report_frequency == 0:
                        report = ', '.join([f'{k}={np.mean(epoch_metrics[k]):.3e}' for k, v in epoch_metrics.items()])
                        print(report)

                    bar.set_postfix(loss = f'{np.mean(epoch_metrics["loss"][-100:]):.3e}')

        epoch_metrics = { f'{split_name} {k}': np.mean(v)
                         for k, v in epoch_metrics.items() }

        return epoch_metrics


    def evaluate_predictions(self, batch, predictions, is_training=False):
        metrics, alignments, images = defaultdict(float), dict(), dict()

        if 'distance_logits' in predictions:
            topography_metrics, images = self.evaluate_topography_predictions(
                batch,
                predictions['distance_logits'],
            )
            metrics.update(topography_metrics)

        if ('translations' in predictions) and ('rotations' in predictions):
            metrics['fape'] = fape_loss(batch, predictions['translations'], predictions['rotations'])

            if not is_training:
                structural_metrics, alignments = self.evaluate_arrangement_predictions(
                    batch, predictions['translations'],
                )
                metrics.update(structural_metrics)

        return metrics, alignments, images


    def evaluate_topography_predictions(self, batch, distance_logits):
        metrics = defaultdict(int)

        external_edges = rearrange(batch.chns, 'b s -> b () s') != rearrange(batch.chns, 'b s -> b s ()')
        external_mask = batch.edge_pad_mask[external_edges]

        distance_logits = distance_logits[external_mask].squeeze()
        distance_ground = batch.edge_distance[external_edges][external_mask]
        distance_labels = self.distance_binner(distance_ground)

        metrics[f'dist_xentropy'] = F.cross_entropy(
            distance_logits[batch.edge_record_mask[external_edges][external_mask]],
            distance_labels[batch.edge_record_mask[external_edges][external_mask]]
        )
        metrics[f'dist_acc'] = (distance_logits.argmax(-1) == distance_labels).float().mean()

        probabilities = distance_logits.softmax(-1)
        values = torch.linspace(0, self.config.distance_pred_max_radius,
                    self.config.distance_pred_number_of_bins, device=distance_logits.device)
        expectation = (probabilities * values.unsqueeze(0)).sum(-1)

        pred_contacts = expectation < self.config.contact_cut
        ground_contacts = distance_ground < self.config.contact_cut

        true_positives = pred_contacts[ground_contacts].sum()
        false_positives = pred_contacts[~ground_contacts].sum()
        false_negatives = ~pred_contacts[~ground_contacts].sum()

        metrics[f'contact_precision'] = true_positives / (true_positives + false_positives + eps)
        metrics[f'contact_recall'] = true_positives / (true_positives + false_negatives + eps)

        batch_images = dict()
        batch_size, seq_len, _ = batch.edge_record_mask.size()
        edge_batches = repeat(torch.arange(0, batch_size), 'b -> b s z', s=seq_len, z=seq_len)
        edge_batches = edge_batches[external_edges][external_mask]

        for batch_idx, id in enumerate(batch.ids):
            batch_filter = edge_batches == batch_idx
            images = [img[batch_filter].detach() for img in
                (distance_labels, expectation, ground_contacts.float(), pred_contacts.float())]
            images = [img[:int(len(img)/2)] for img in images]

            chains = batch.chns[batch.node_pad_mask][batch.batch == batch_idx]
            n, m = chains.sum(), len(chains) - chains.sum()
            images = [rearrange(img, '(m n) -> n m', n=n, m=m).cpu() for img in images]
            batch_images[id] = images

        return metrics, batch_images


    def evaluate_arrangement_predictions(self, batch, traj):
        node_mask = batch.node_pad_mask & batch.node_record_mask
        traj = rearrange(traj, 'b n t e -> b t n e')
        batch_size, trajectory_len, _, _ = traj.size()

        alignments = defaultdict(lambda: defaultdict(list))
        metrics = defaultdict(int)

        for id, angles, gnd_wrap, pred_traj, chains, mask in zip(batch.ids, batch.angs,
                                batch.tgt_crds[:, :, 1, :], traj, batch.chns, node_mask):
            gnd_wrap, angles, chains = gnd_wrap[mask], angles[mask], chains[mask]

            for step, pred_wrap in enumerate(pred_traj):
                pred_wrap = pred_wrap[mask]

                # for computing internal aligment
                # for chain in (0, 1):
                #     chain_alignment_metrics, _ = get_alignment_metrics(
                #         deepcopy(gnd_wrap)[chains == chain],
                #         pred_wrap[chains == chain],
                #     )
                #
                #     for k, metric in chain_alignment_metrics.items():
                #         metrics[f'internal_{k}'] += metric.mean() / trajectory_len / batch_size / 2

                alignment_metrics, (align_gnd_coors, align_pred_coors) = get_alignment_metrics(
                    deepcopy(gnd_wrap),
                    pred_wrap,
                )

                alignments[id][f'trajectory'].append((align_gnd_coors.detach().cpu(),
                                                      align_pred_coors.detach().cpu(),
                                                      angles.cpu()))
                alignments[id][f'alignment_metrics'].append({k: v.mean().cpu().item() for k, v in alignment_metrics.items()})

                for k, metric in alignment_metrics.items():
                    metrics[k] += metric.mean() / trajectory_len / batch_size

        return metrics, alignments


    def plot_images(self, images):
        for id, imgs in images.items():
            prediction_figure = plot_predictions(*imgs)

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
                len(chain) - chain.sum().item()
            )

            if wandb.run: wandb.log({ f'{id} timeseries': wandb.Html(timeseries_fig._make_html()) })
