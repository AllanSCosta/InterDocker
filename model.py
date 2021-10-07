import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from einops import rearrange, repeat
from torch import einsum
from functools import partial
from invariant_point_attention import InvariantPointAttention, IPABlock

from utils import soft_one_hot_linspace, angle_to_point_in_circum, rotation_6d_to_matrix


class Encoder(nn.Module):
    def __init__(
            self,
            dim,
            edim,
            heads,
            scalar_key_dim = 16,
            scalar_value_dim = 16,
            point_key_dim = 4,
            point_value_dim = 4,
            depth=30
        ):
        super().__init__()

        self.layers = nn.ModuleList([
            IPABlock(
                dim = 64,
                pairwise_repr_dim=32,
                heads = 4,
                scalar_key_dim = 16,
                scalar_value_dim = 16,
                point_key_dim = 4,
                point_value_dim = 4
            )
            for _ in range(depth)
        ])

    def block_checkpoint(self, layers):
        def checkpoint_forward(x):
            return layers(x)
        return checkpoint_forward

    def forward(self, nodes, edges, coors, rotations, mask):
        for layer in self.layers:
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
            dim,
            edim,
            heads,
            scalar_key_dim = 16,
            scalar_value_dim = 16,
            point_key_dim = 4,
            point_value_dim = 4,
            depth=30
        ):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.ModuleList([IPABlock(
                dim = dim,
                pairwise_repr_dim=edim,
                heads = heads,
                scalar_key_dim = scalar_key_dim,
                scalar_value_dim = scalar_value_dim,
                point_key_dim = point_key_dim,
                point_value_dim = point_value_dim
            ),
            nn.Linear(dim, 9)])
            for _ in range(depth)
        ])

    def block_checkpoint(self, layers):
        def checkpoint_forward(x):
            return layers(x)
        return checkpoint_forward

    def forward(self, nodes, edges, coors, rotations, mask):
        trajectory_crds, trajectory_rots  = [], []
        for (layer, to_step) in self.layers:
            nodes = layer(
                nodes,
                pairwise_repr = edges,
                rotations = rotations,
                translations = coors,
                mask = mask
            )

            update = to_step(nodes)
            rotation_update, translation_update = update[..., :6], update[..., 6:]
            rotation_update = rotation_6d_to_matrix(rotation_update)

            coors = coors + einsum('b n c, b n c r -> b n r', translation_update, rotations)
            rotations = rotation_update @ rotations

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

        self.encoder = Encoder(
            config.dim,
            config.edim,
            config.heads,
            scalar_key_dim = config.scalar_key_dim,
            scalar_value_dim = config.scalar_value_dim,
            point_key_dim = config.point_key_dim,
            point_value_dim = config.point_value_dim,
            depth=config.encoder_depth
        )

        self.cross_encoder = CrossEncoder(
            config.dim,
            config.edim,
            config.heads,
            scalar_key_dim = config.scalar_key_dim,
            scalar_value_dim = config.scalar_value_dim,
            point_key_dim = config.point_key_dim,
            point_value_dim = config.point_value_dim,
            depth=config.cross_encoder_depth
        )

        self.to_distance = Dense([config.edim, config.edim, config.distance_pred_number_of_bins])
        self.to_angle = Dense([config.edim, 2 * config.edim, 3 * config.angle_pred_number_of_bins])
        self.to_position = Dense([config.dim, config.dim, 3])

        self.gaussian_noise = config.gaussian_noise
        self.unroll_steps = config.unroll_steps

        self.edge_norm_enc = partial(soft_one_hot_linspace, start=0,
                    end=config.distance_max_radius, number=config.distance_number_of_bins, basis='gaussian', cutoff=True)
        self.edge_angle_enc = partial(soft_one_hot_linspace, start=-1,
                    end=1, number=config.angle_number_of_bins, basis='gaussian', cutoff=True)

        edge_enc_size = config.distance_number_of_bins + 3 * config.angle_number_of_bins
        self.to_internal_edge = Dense([edge_enc_size, 2 * edge_enc_size, edge_enc_size, config.edim])
        self.to_external_edge = Dense([2 * config.dim, 2 * config.dim, config.dim, config.edim])

        self.unroll_steps = config.unroll_steps
        self.eval_steps = config.eval_steps


    def forward(self, batch, is_training):
        output = {}

        node_pad_mask = batch.node_pad_mask
        edge_pad_mask = batch.edge_pad_mask
        batch_size, seq_len = batch.seqs.size()

        internal_edge_mask = rearrange(batch.chns, 'b s -> b () s') == rearrange(batch.chns, 'b s -> b s ()')
        external_edge_mask = ~internal_edge_mask

        # produce distance and angular signal for pairwise representations
        dist_signal = soft_one_hot_linspace(batch.edge_distance, start=2, end=30,
                    number=self.config.distance_number_of_bins, basis='gaussian', cutoff=True)

        angle_signal = soft_one_hot_linspace(batch.edge_angles, start=-1, end=1,
                    number=self.config.angle_number_of_bins, basis='gaussian', cutoff=True)
        angle_signal = rearrange(angle_signal, 'b s z a c -> b s z (a c)')

        edges = self.to_internal_edge(torch.cat((dist_signal, angle_signal), dim=-1))


        # experimental ---------------> folding start

        # (a) noised ground truth; useful for testing the module
        # coors = batch.tgt_crds[:, :, 1, :].clone() + torch.normal(0, 3, batch.tgt_crds[:, :, 1, :].shape, device=batch.tgt_crds[:, :, 1, :].device)

        # (b) hold one protein in place, start the other from the origin
        # coors = batch.bck_crds.clone()
        # rots = batch.rots
        #
        # coors[batch.chns == 1] = 0
        # rots[batch.chns == 1] = repeat(torch.eye(3), '... -> b ...', b = rots[batch.chns == 1].size(0)).to(batch.rots.device)

        # (c) hold the two proteins in place
        # coors = batch.bck_crds.clone()
        # rots = batch.rots

        # (d) start both from origin ('black-hole start')
        coors = torch.zeros_like(batch.bck_crds, device=batch.bck_crds.device)
        rots = repeat(torch.eye(3), '... -> b n ...', b = batch_size, n = seq_len).to(batch.rots.device)

        # ----------------------------------

        # embed node information
        nodes = self.node_token_emb(batch.seqs)
        angles = self.circum_emb(angle_to_point_in_circum(batch.angs))

        bb_dihedrals = rearrange(angles[..., :6, :], 'b s a h -> b s (a h)')
        sc_dihedrals = rearrange(angles[..., 6:, :], 'b s a h -> b s (a h)')

        bb_dihedrals = self.backbone_dihedrals_emb(bb_dihedrals)
        sc_dihedrals = self.sidechain_dihedrals_emb(sc_dihedrals)

        dihedrals = self.dihedral_emb(torch.cat((bb_dihedrals, sc_dihedrals), dim=-1))

        nodes = nodes + dihedrals


        # add noise to prevent dx=0 being a solution when we start from local solutions
        coors = coors.clone().detach() + torch.normal(0, 1, coors.shape, device=coors.device)

        # unroll network
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


# This code will be re-integrated into forward() once our task is done

# dist_signal = soft_one_hot_linspace(batch.edge_distance[internal_edge_mask], start=2, end=30,
#             number=self.config.distance_number_of_bins, basis='gaussian', cutoff=True).detach()
#
# angle_signal = soft_one_hot_linspace(batch.edge_angles[internal_edge_mask], start=-1, end=1,
#             number=self.config.angle_number_of_bins, basis='gaussian', cutoff=True)
# angle_signal = rearrange(angle_signal, 'e a c -> e (a c)').detach()
# edges = torch.zeros(batch_size, seq_len, seq_len, self.config.edim, device=dist_signal.device)

# internal_edges = self.to_internal_edge(torch.cat((dist_signal, angle_signal), dim=-1))
# edges[internal_edge_mask] = internal_edges

# encodings = torch.zeros_like(nodes, device=nodes.device)
# for chain in (0, 1):
#     chain_mask = (node_pad_mask) & (batch.chns == chain)
#     chain_encoding, _ = self.encoder(nodes=nodes, edges=edges, mask=chain_mask)
#     encodings[batch.chns == chain] = chain_encoding[batch.chns == chain]
#
# endpoints = repeat(encodings, 'b s c -> b z s c', z=seq_len), repeat(encodings, 'b s c -> b s z c', z=seq_len)
# cross_cat = torch.cat(endpoints, dim=-1)[external_edge_mask]

# external_edges = self.to_external_edge(cross_cat)
# self.node_crds_emb(coors)

# coors[batch.chns == 1] = 0

# output['distance_logits'] = self.to_distance(external_edges)
# dist_signal = soft_one_hot_linspace(batch.edge_distance[external_edge_mask], start=2, end=30,
#             number=self.config.distance_number_of_bins, basis='gaussian', cutoff=True).detach()
#
# angle_signal = soft_one_hot_linspace(batch.edge_angles[external_edge_mask], start=-1, end=1,
#             number=self.config.angle_number_of_bins, basis='gaussian', cutoff=True)
# angle_signal = rearrange(angle_signal, 'e a c -> e (a c)').detach()
#
# edges = torch.zeros(batch_size, seq_len, seq_len, self.config.edim, device=dist_signal.device)
# external_edges = self.to_internal_edge(torch.cat((dist_signal, angle_signal), dim=-1))
#
# edges = torch.zeros(batch_size, seq_len, seq_len, self.config.edim, device=dist_signal.device)
# edges[internal_edge_mask] = internal_edges
# edges[external_edge_mask] = external_edges

# output['distance_logits'] = self.to_distance(edges[external_edge_mask])
