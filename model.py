import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from torch_geometric.nn import global_mean_pool
from einops import rearrange, repeat
from torch import einsum
from functools import partial
from invariant_point_attention import InvariantPointAttention, IPABlock

from utils import soft_one_hot_linspace, angle_to_point_in_circum

from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles, rotation_6d_to_matrix


class Interactoformer(nn.Module):
    def __init__(
            self,
            config
        ):
        super().__init__()
        self.config = config

        self.node_crds_emb = nn.Linear(3, config.dim)
        self.node_token_emb = nn.Embedding(21, config.dim, padding_idx=0)

        self.circum_emb = nn.Linear(2, 12)

        self.backbone_dihedrals_emb = Dense([6 * 12, 64, 32])
        self.sidechain_dihedrals_emb = Dense([6 * 12, 64, 32])
        self.dihedral_emb = Dense([64, 64, config.dim])

        self.encoder = Encoder(config)
        self.cross_encoder = CrossEncoder(config)

        self.to_distance = Dense([config.edim, config.edim, config.distance_pred_number_of_bins])

        self.gaussian_noise = config.gaussian_noise
        self.unroll_steps = config.unroll_steps

        self.edge_norm_enc = partial(soft_one_hot_linspace, start=2,
                    end=config.distance_max_radius, number=config.distance_number_of_bins, basis='gaussian', cutoff=True)
        self.edge_angle_enc = partial(soft_one_hot_linspace, start=-1,
                    end=1, number=config.angle_number_of_bins, basis='gaussian', cutoff=True)

        edge_enc_size = config.distance_number_of_bins + 3 * config.angle_number_of_bins
        self.to_internal_edge = Dense([edge_enc_size, 2 * edge_enc_size, edge_enc_size, config.edim])

        self.external_leak = config.external_leak
        if self.external_leak:
            self.to_external_edge = Dense([edge_enc_size, 2 * edge_enc_size, edge_enc_size, config.edim])
        else:
            self.to_external_edge = Dense([2 * config.dim, 2 * config.dim, config.dim, config.edim])

        self.to_angle = Dense([config.edim, 2 * config.edim, 3 * config.angle_pred_number_of_bins])
        self.to_position = Dense([config.dim, config.dim, 3])


        self.unroll_steps = config.unroll_steps
        self.eval_steps = config.eval_steps



    def forward(self, batch, is_training):
        output = {}

        # utils
        node_pad_mask = batch.node_pad_mask
        edge_pad_mask = batch.edge_pad_mask
        batch_size, seq_len = batch.seqs.size()
        device = batch.seqs.device

        # embed individual node information
        nodes = self.node_token_emb(batch.seqs)

        angles = self.circum_emb(angle_to_point_in_circum(batch.angs))

        bb_dihedrals = rearrange(angles[..., :6, :], 'b s a h -> b s (a h)')
        bb_dihedrals = self.backbone_dihedrals_emb(bb_dihedrals)
        sc_dihedrals = rearrange(angles[..., 6:, :], 'b s a h -> b s (a h)')
        sc_dihedrals = self.sidechain_dihedrals_emb(sc_dihedrals)

        dihedrals = self.dihedral_emb(torch.cat((bb_dihedrals, sc_dihedrals), dim=-1))

        nodes = nodes + dihedrals

        # produce distance and angular signal for pairwise internal representations
        edges = torch.zeros(batch_size, seq_len, seq_len, self.config.edim, device=device)
        internal_edge_mask = rearrange(batch.chns, 'b s -> b () s') == rearrange(batch.chns, 'b s -> b s ()')
        internal_edge_mask =  internal_edge_mask & edge_pad_mask
        external_edge_mask = ~internal_edge_mask & edge_pad_mask

        max_radius = self.config.distance_max_radius
        dist_signal = self.edge_norm_enc(batch.edge_distance)
        angle_signal = self.edge_angle_enc(batch.edge_angles)
        angle_signal = rearrange(angle_signal, 'b s z a c -> b s z (a c)')
        angle_signal[batch.edge_distance < max_radius] = 0

        edge_structural_signal = torch.cat((dist_signal, angle_signal), dim=-1)

        edges[internal_edge_mask] = self.to_internal_edge(edge_structural_signal[internal_edge_mask])

        # start from superposition
        coors = batch.bck_crds.clone().detach()
        rots = batch.rots

        # encode chains independently
        encodings_buf = torch.zeros_like(nodes, device=device)
        for chain in (1, 2):
            chain_mask = (node_pad_mask) & (batch.chns == chain)
            chain_encoding = self.encoder(nodes, edges, coors, rots, mask=chain_mask)
            encodings_buf[batch.chns == chain] = chain_encoding[batch.chns == chain]
        nodes = nodes + encodings_buf

        # produce pairwise external representations and compute distance
        if not self.external_leak:
            endpoints = repeat(nodes, 'b s c -> b z s c', z=seq_len), repeat(nodes, 'b s c -> b s z c', z=seq_len)
            cross_cat = torch.cat(endpoints, dim=-1)
            external_edges = self.to_external_edge(cross_cat)
            output['distance_logits'] = self.to_distance(external_edges[external_edge_mask])
        else:
            external_edges = self.to_external_edge(edge_structural_signal)
        edges = edges + external_edges * external_edge_mask[..., None]


        # add gaussian noise if you're feeling like it
        if self.gaussian_noise:
            coors = coors + torch.normal(0, self.gaussian_noise, coors.shape, device=device)
        rots, coors = rots.clone(), coors.clone()

        # unroll network so we learn to refine ourselves
        if is_training and self.unroll_steps:
            batch_steps = torch.randint(0, self.unroll_steps, [batch_size])
            with torch.no_grad():
                max_step = torch.max(batch_steps).item()
                for step in range(max_step):
                    chosen = batch_steps >= step
                    _, translations_timeseries, rotations_timeseries = self.cross_encoder(
                        nodes[chosen], edges[chosen], coors[chosen], rots[chosen],
                        mask=batch.node_pad_mask[chosen]
                    )
                    rots[chosen] = rotations_timeseries[-1]
                    coors[chosen] = translations_timeseries[-1]

        # produce timeseries
        translations, rotations  = [ coors ], [ rots ]
        for _ in range(1 if is_training else self.eval_steps):
            nodes_, translations_timeseries, rotations_timeseries = self.cross_encoder(nodes, edges, coors, rots, mask=batch.node_pad_mask)
            translations.extend(translations_timeseries)
            rotations.extend(rotations_timeseries)
            rots = rotations_timeseries[-1].detach()
            coors = translations_timeseries[-1].detach()

        output['translations'] = rearrange(torch.stack(translations, dim=0), 't b n e -> b n t e')
        output['rotations'] = rearrange(torch.stack(rotations, dim=0), 't b n p q -> b n t p q')

        return output


class Encoder(nn.Module):
    def __init__(
            self,
            config
        ):
        super().__init__()
        self.layers = nn.ModuleList([
            IPABlock(
                dim = config.dim,
                pairwise_repr_dim = config.edim,
                heads = config.heads,
                scalar_key_dim = config.scalar_key_dim,
                scalar_value_dim = config.scalar_value_dim,
                point_key_dim = config.point_key_dim,
                point_value_dim = config.point_value_dim,
            )
            for _ in range(config.encoder_depth)
        ])

        self.checkpoint = config.checkpoint

    def block_checkpoint(self, layer):
        def checkpoint_forward(nodes, edges, coors, rotations, mask):
            return layer(
                nodes,
                pairwise_repr = edges,
                rotations = rotations,
                translations = coors,
                mask = mask
            )
        return checkpoint_forward

    def forward(self, nodes, edges, coors, rotations, mask):
        for layer in self.layers:
            if self.checkpoint:
                nodes = checkpoint.checkpoint(
                    self.block_checkpoint(layer),
                    nodes, edges, coors, rotations, mask
                )
            else:
                nodes = layer(
                    nodes,
                    pairwise_repr = edges,
                    rotations = rotations,
                    translations = coors,
                    mask = mask
                )
        return nodes


class CrossEncoder(nn.Module):
    def __init__(self,
            config
        ):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.ModuleList([IPABlock(
                dim = config.dim,
                pairwise_repr_dim = config.edim,
                heads = config.heads,
                scalar_key_dim = config.scalar_key_dim,
                scalar_value_dim = config.scalar_value_dim,
                point_key_dim = config.point_key_dim,
                point_value_dim = config.point_value_dim,
            ),
            nn.Linear(config.dim, 9)])
            for _ in range(config.cross_encoder_depth)
        ])

        self.checkpoint = config.checkpoint

    def block_checkpoint(self, layer, to_step):
        def checkpoint_forward(nodes, edges, coors, rotations, mask):
            nodes = layer(
                nodes,
                pairwise_repr = edges,
                rotations = rotations,
                translations = coors,
                mask = mask
            )
            batch_steps = to_step(nodes)
            return nodes, batch_steps
        return checkpoint_forward


    def forward(self, nodes, edges, coors, rotations, mask):
        trajectory_crds, trajectory_rots  = [], []
        batch_index = repeat(torch.arange(0, len(nodes), device=nodes.device),
                                            'd -> d p', p=nodes.size(1))[mask]

        for (layer, to_step) in self.layers:
            if self.checkpoint:
                nodes, batch_steps = checkpoint.checkpoint(
                    self.block_checkpoint(layer, to_step),
                    nodes, edges, coors, rotations, mask
                )
            else:
                nodes = layer(
                    nodes,
                    pairwise_repr = edges,
                    rotations = rotations,
                    translations = coors,
                    mask = mask
                )

                batch_steps = to_step(nodes)

            rot_update, crd_update = batch_steps[..., :6], batch_steps[..., 6:]
            rot_update = rotation_6d_to_matrix(rot_update)
            crd_update = einsum('b n c, b n c r -> b n r', crd_update, rotations)

            coors = coors + crd_update
            rotations = rotations @ rot_update

            trajectory_crds.append(coors)
            trajectory_rots.append(rotations)
            rotations = rotations.detach()

        return nodes, trajectory_crds, trajectory_rots


class Dense(nn.Module):
    def __init__(self, layer_structure, checkpoint=False):
        super().__init__()
        layers = []
        for idx, (back, front) in enumerate(zip(layer_structure[:-1],
                                            layer_structure[1:])):
            layers.append(nn.Linear(back, front))
            if idx < len(layer_structure) - 2: layers.append(nn.LeakyReLU())
        self.layers = nn.Sequential(*layers)
        self.checkpoint = checkpoint

    def block_checkpoint(self, layers):
        def checkpoint_forward(x):
            return layers(x)
        return checkpoint_forward

    def forward(self, x):
        return checkpoint.checkpoint(self.block_checkpoint(self.layers), x) if self.checkpoint else self.layers(x)
