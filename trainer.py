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
from einops import repeat, rearrange, reduce
from torch_geometric.utils.metric import precision, recall

from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d

import torch.nn.functional as F

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

            if wandb.run: wandb.log(epoch_metrics)

            print(f'saving model: {self.model_path}')
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

                    batch_pair, tgt_translation_diff, tgt_rotation_diff = batch
                    batch_pair = [b.to(self.device) for b in (batch_pair)]

                    tgt_translation_diff, tgt_rotation_diff = tgt_translation_diff[..., 0].to(self.device), tgt_rotation_diff[..., 0, :, :].to(self.device)
                    tgt_rotation_diff6d = matrix_to_rotation_6d(tgt_rotation_diff)

                    batch_predictions = self.model(batch_pair, is_training=is_training)

                    translation_diff, rotation_diff6d, scores = batch_predictions['translation'], batch_predictions['rotation'], batch_predictions['score']
                    distance_diff = batch_predictions['distance']
                    rotation_diff = rotation_6d_to_matrix(rotation_diff6d)

                    tgt_rotation_diff6d = rearrange(tgt_rotation_diff6d, '... (s c) -> ... s c', s=2, c=3)
                    rotation_diff6d = rearrange(rotation_diff6d, '... (s c) -> ... s c', s=2, c=3)

                    receptor, ligand = batch_pair

                    b, t, n, m, _ = translation_diff.size()
                    pad_mask = repeat(receptor.node_pad_mask, 'b n -> b t n m', m=m, t=t) & repeat(ligand.node_pad_mask, 'b m -> b t n m', n=n, t=t)

                    translation_error = torch.norm(translation_diff - repeat(tgt_translation_diff, 'b ... -> b t ...', t=t), dim=-1, p=2)
                    translation_error = translation_error * pad_mask.float()

                    rotation_unity_error = (torch.abs(torch.norm(rotation_diff6d, dim=-1, p=2) - 1) * pad_mask[..., None].float()).sum((-2, -3)) / pad_mask.sum((-1, -2))[..., None]
                    rotation_unity_loss = rotation_unity_error.mean()

                    rotation_error = rotation_diff6d - repeat(tgt_rotation_diff6d, 'b ... -> b t ...', t=t)
                    rotation_error = torch.norm(rotation_error, dim=-1, p=2).mean(-1)

                    error_mask = torch.norm(repeat(tgt_translation_diff, 'b ... -> b t ...', t=t), dim=-1, p=2) < 15

                    norm_error = torch.abs(distance_diff - repeat(torch.norm(tgt_translation_diff, dim=-1, p=2), 'b ... -> b t ...', t=t)[..., None])
                    norm_error = norm_error[error_mask & pad_mask].mean()

                    translation_error = translation_error[error_mask & pad_mask].mean()
                    rotation_error = rotation_error[error_mask & pad_mask].mean()

                    # b n m 1
                    # ligand_rotations = torch.einsum('b t n m l r, b n w l -> b t n m w r', repeat(tgt_rotation_diff, 'b ... -> b t ...', t=t), receptor.tgt_rots[..., 0, ])
                    # ligand_translations = torch.einsum('b t n m l, b n w l -> b t n m w', repeat(tgt_translation_diff, 'b ... -> b t ...', t=t), receptor.tgt_rots[..., 0, ])

                    ligand_rotations = torch.einsum('b t n m l r, b n w l -> b t n m w r', rotation_diff, receptor.tgt_rots[..., 0, ])
                    ligand_translations = torch.einsum('b t n m l, b n w l -> b t n m w', translation_diff, receptor.tgt_rots[..., 0, ])

                    receptor_coords = repeat(receptor.tgt_crds[..., 0], 'b n w -> b t n m w', t=t, m=m)
                    ligand_coords = ligand_translations + receptor_coords

                    induced_rigid_alignments = torch.einsum('b m k l, b t n m w l -> b t n m k w', ligand.edge_vectors, ligand_rotations)
                    induced_rigid_alignments = induced_rigid_alignments + repeat(ligand_coords, '... w -> ... m w', m=m)

                    target_alignment = repeat(ligand.tgt_crds[..., 0], 'b m w -> b t n m k w', t=t, n=n, k=m) * pad_mask[..., None, None]

                    mask = torch.norm(target_alignment, dim=-1, p=2) > 0
                    norms = ((torch.norm(induced_rigid_alignments - target_alignment.transpose(-2, -3), dim=-1, p=2) ** 2 )  * mask.float())
                    mean_norm = (norms.sum((-1)) / (1e-7 + mask.sum((-1)))).relu().sqrt()

                    logits_killer = torch.full_like(scores, fill_value=-float('inf'), device=scores.device)
                    scores = torch.where(pad_mask[..., None], scores, logits_killer)
                    normalized_scores = F.softmax(rearrange(scores, '... n m s -> ... (n m s)'), dim=-1)
                    normalized_scores = rearrange(normalized_scores, '... (n m) -> ... n m', n=n, m=m)

                    mean_rmsd = (mean_norm.sum((-1,-2)) / pad_mask.sum((-1, -2))).mean()

                    # loss = (mean_norm[:, -1] * normalized_scores[:, -1]).sum((-1, -2)).mean() + 10 * rotation_unity_loss + translation_error / 10 + rotation_error * 10
                    # mean_norm[error_mask].mean()  +

                    loss = norm_error # rotation_unity_loss + translation_error + rotation_error

                    metrics = defaultdict(float)

                    metrics['mean_rmsd'] = mean_rmsd
                    metrics['loss'] = loss
                    metrics['translation error'] = translation_error.mean()
                    metrics['rotation error'] = rotation_error.mean()
                    metrics['distance error'] = norm_error.mean()
                    metrics['rotation_unity_loss'] = rotation_unity_loss
                    metrics['norm error'] = norm_error

                    batch_argmax = reduce(normalized_scores[:, -1], 'b n m ... -> b ...', 'max')
                    argmax_location = repeat(batch_argmax, '... -> ... n m', n=n, m=m) == normalized_scores[:, -1]
                    argmax_b, argmax_n, argmax_m  = torch.nonzero(argmax_location).T

                    metrics['argmax_rmsd'] = mean_norm[argmax_b, -1, argmax_n, argmax_m].mean()

                    loss = metrics['loss'] = (
                        self.config.distogram_coefficient * metrics['loss']
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
                        prediction_images, alignments = defaultdict(list), defaultdict(list)

                        img_pairs = [
                            distance_diff[:, 0, ..., 0], torch.norm(tgt_translation_diff, dim=-1, p=2),
                            -scores[:, -1, ...],  mean_norm[:, -1]
                        ]

                        ligand_argmax = induced_rigid_alignments[argmax_b, -1, argmax_n, argmax_m] + 1
                        ligand_ground = ligand.tgt_crds[..., 0]

                        receptor_argmax = receptor.bck_crds + 1
                        receptor_ground = receptor.tgt_crds[..., 0]

                        for batch_idx in range(b):
                            sequence = receptor.str_seqs[batch_idx] + ligand.str_seqs[batch_idx]
                            chains1, chains2 = receptor.chns[batch_idx], ligand.chns[batch_idx]
                            id = receptor.ids[batch_idx]

                            images = [img[batch_idx].detach() for img in img_pairs]
                            n, m = (chains1 == chains1[0]).sum(), (chains2 == chains2[0]).sum()

                            images = [img[:n, :m].cpu() for img in images]
                            prediction_images[id].append(images)

                            argmaxes = torch.cat((receptor_argmax[batch_idx, :n], ligand_argmax[batch_idx, :m]), dim=0).cpu()
                            grounds = torch.cat((receptor_ground[batch_idx, :n], ligand_ground[batch_idx, :m]), dim=0).cpu()
                            angles = torch.cat((receptor.angs[batch_idx, :n], ligand.angs[batch_idx, :m]), dim=0).cpu()

                            # breakpoint()
                            ts = plot_aligned_timeseries(sequence, [(grounds, argmaxes, angles)], n)
                            if wandb.run: wandb.log({ f'{id} timeseries': wandb.Html(ts._make_html()) })
                            with open('test.html', 'w') as file: file.write(ts._make_html())

                        self.plot_images(prediction_images)



                    for k, v in metrics.items(): epoch_metrics[k].append(v.item() if type(v) is not float else v)

                    if batch_idx % self.config.report_frequency == 0:
                        report = ', '.join([f'{k}={np.mean(epoch_metrics[k][-100:]):.3e}' for k, v in epoch_metrics.items()])
                        print(report)

                    bar.set_postfix(loss = f'{np.mean(epoch_metrics["loss"][-100:]):.3e}', rmsd=f'{np.mean(epoch_metrics["argmax_rmsd"][-100:]):.3e}')

        epoch_metrics = { f'{split_name} {k}': np.mean(v)
                         for k, v in epoch_metrics.items() }

        return epoch_metrics


    def evaluate_predictions(self, batch, predictions, split, batch_idx):
        metrics, alignments, images = defaultdict(float), dict(), dict()

        should_fetch_results = split in TEST_DATASETS and batch_idx < self.config.num_test_visual_samples

        # ===== EVALUATE DISTOGRAM PREDICTIONS ======
        # if 'logit_traj' in predictions:
            # distogram_metrics, images, permutations = self.evaluate_vectorgrams(
                # batch,
                # predictions['logit_traj'],
                # should_fetch_results
            # )
            # metrics.update(distogram_metrics)
        # else:
            # permutations = torch.zeros(batch.seqs.size(0), device=batch.seqs.device)


        breakpoint()

        return metrics, alignments, images

    #
    # def evaluate_vectorgrams(self, batch, logits_trajectory, fetch_images):
    #     metrics = defaultdict(int)
    #     batch_images = defaultdict(list)
    #
    #     pair, cross_edges_ground = batch[:2], batch[2]
    #     n, m = [p.encs.size(1) for p in pair]
    #     device = cross_edges_ground.device
    #
    #     if self.config.real_value:
    #         trajectory = logits_trajectory['vector']
    #     else:
    #         trajectory = logits_trajectory['distance']
    #
    #     trajectory_len = trajectory.size(1)
    #     batch_size = trajectory.size(0)
    #
    #     predictions = repeat(trajectory, 'b t n m c -> b n m s t c', s=2)
    #     targets = repeat(cross_edges_ground, 'b n m c s -> b n m s t c', t=trajectory_len)
    #     mask = (repeat(pair[0].node_pad_mask, 'b n -> b n m s t', m=m, s=2, t=trajectory_len)
    #           & repeat(pair[1].node_pad_mask, 'b m -> b n m s t', n=n, s=2, t=trajectory_len))
    #
    #     if self.config.real_value:
    #         vector_differences = torch.norm(predictions - targets / 25, dim=-1, p=2)
    #         # vector_differences = torch.norm(predictions - torch.clamp(torch.norm(targets, dim=-1)[..., None] / 25, min=0, max=1), dim=-1, p=1)
    #         vector_differences = vector_differences * mask.float() * (torch.norm(targets, dim=-1) < 25).float()
    #
    #         vector_differences = vector_differences.sum((1, 2, 4)) / (mask.sum((1, 2, 4)))
    #         vector_differences, permuted = vector_differences.min(dim=-1)
    #
    #         metrics['fale'] = vector_differences.mean()
    #         # breakpoint()
    #         permutation_mask = torch.stack((permuted.bool(), ~(permuted.bool())), dim=-1)
    #         ground_images = rearrange(rearrange(torch.clamp(targets / 25, min=-1, max=1)[..., -1, :], 'b ... s c -> b s ... c')[permutation_mask], 'b ... c -> c b ...') + 1
    #         # ground_images = rearrange(rearrange(torch.clamp(torch.norm(targets, dim=-1)[..., None] / 25, min=0, max=1)[..., -1, :], 'b ... s c -> b s ... c')[permutation_mask], 'b ... c -> c b ...') + 1
    #         predicted_images = rearrange(rearrange(predictions[..., -1, :], 'b ... s c -> b s ... c')[permutation_mask], 'b ... c -> c b ...') + 1
    #
    #     else:
    #         distance_labels = self.binners['distance'](torch.norm(targets, dim=-1, p=2))
    #         angle_labels = self.binners['angles'](F.normalize(targets, dim=-1))
    #
    #         distance_labels[~mask] = IGNORE_IDX
    #         distance_logits = rearrange(predictions, 'b ... l -> b l ...')
    #
    #         # use ground solution that provides the best cross entropy for distogram loss
    #         xentropy = cross_entropy(distance_logits, distance_labels, reduction='none', ignore_index=IGNORE_IDX)
    #         xentropy = xentropy.sum((1, 2, 4)) / (mask.sum((1, 2, 4)))
    #         xentropy, permuted = xentropy.min(dim=-1)
    #         xentropy = xentropy.mean()
    #         metrics[f'distance xentropy'] = xentropy
    #
    #         permutation_mask = torch.stack((permuted.bool(), ~(permuted.bool())), dim=-1)
    #         mask = rearrange(mask, 'b n m s t -> b s t n m ')[permutation_mask]
    #
    #         if self.predict_angles:
    #             angles_ground = rearrange(angle_labels, 'b ... s t a -> b s t ... a')
    #             angles_ground = angles_ground[permutation_mask]
    #             angles_logits = rearrange(logits_trajectory['angles'], 'b ... (a l) -> b l ... a', a=3)
    #             angles_ground[~mask] = IGNORE_IDX
    #             angles_ground[rearrange(distance_labels, 'b ... s t -> b s t ...')[permutation_mask] == self.config.distance_pred_number_of_bins - 1] = IGNORE_IDX
    #             angular_xentropy = cross_entropy(angles_logits, angles_ground, ignore_index=IGNORE_IDX)
    #             metrics[f'angular xentropy'] = angular_xentropy
    #
    #         terminal_logits = rearrange(distance_logits[..., -1], 'b l ... s -> b s ... l')[permutation_mask]
    #         final_dist_pred = logit_expectation(terminal_logits)[None, ...]
    #         final_angle_pred = logit_expectation(rearrange(angles_logits[:, :, -1], 'b l ... c -> c b ... l'))
    #         predicted_images = torch.cat((final_dist_pred, final_angle_pred), dim=0)
    #         angles_ground = rearrange(angles_ground[:, -1], 'b ... s -> s b ...')
    #         distance_labels = rearrange(distance_labels[..., -1], 'b ... s -> b s ...')[permutation_mask][None, ...]
    #         ground_images = torch.cat((distance_labels, angles_ground), dim=0)
    #
    #     if not fetch_images:
    #         return metrics, batch_images, permuted
    #
    #     candidate_images = []
    #     for dimension_idx in range(predicted_images.size(0)):
    #         pred_img = predicted_images[dimension_idx]
    #         grnd_img = ground_images[dimension_idx]
    #         candidate_images.extend([grnd_img, pred_img])
    #
    #     for batch_idx, (chains1, chains2, id) in enumerate(zip(pair[0].chns, pair[1].chns, pair[0].ids)):
    #         images = [img[batch_idx].detach() for img in candidate_images]
    #         n, m = (chains1 == chains1[0]).sum(), (chains2 == chains2[0]).sum()
    #         images = [img[:n, :m].cpu() for img in images]
    #         batch_images[id].append(images)
    #
    #     return metrics, batch_images, permuted


    def plot_images(self, images):
        for id, img_trajectory in images.items():
            for imgs in img_trajectory:
                prediction_figure = plot_predictions(imgs)
                if wandb.run:
                    wandb.log({
                        f'{id} at topography prediction': prediction_figure,
                    })


    def plot_alignments(self, timeseries):
        pass
        # for id, sequence, chain, mask in zip(batch.ids, batch.str_seqs, batch.chns, batch.node_pad_mask):
        #     timeseries = alignments[id]
        #     chain = chain[mask]
        #
        #     timeseries_fig = plot_aligned_timeseries(
        #         sequence,
        #         timeseries['trajectory'],
        #         (chain == chain[0]).sum().item()
        #     )
        #
        #     if wandb.run: wandb.log({ f'{id} timeseries': wandb.Html(timeseries_fig._make_html()) })
