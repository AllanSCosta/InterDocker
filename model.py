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


def FeedForward(dim, mult = 1., num_layers = 2, act = nn.ReLU):
    layers = []
    dim_hidden = dim * mult

    for ind in range(num_layers):
        is_first = ind == 0
        is_last  = ind == (num_layers - 1)
        dim_in   = dim if is_first else dim_hidden
        dim_out  = dim if is_last else dim_hidden

        layers.append(nn.Linear(dim_in, dim_out))

        if is_last:
            continue

        layers.append(act())

    return nn.Sequential(*layers)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        pos_emb = None,
        dim_head = 64,
        heads = 8,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim)
        self.to_kv = nn.Linear(dim, inner_dim * 2)

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, nodes, edge_mask = None):
        h = self.heads

        q = self.to_q(nodes)
        k, v = self.to_kv(nodes).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b ... (h d) -> (b h) ... d', h = h), (q, k, v))
        k, v = map(lambda t: rearrange(t, 'b j d -> b () j d '), (k, v))

        sim = einsum('b i d, b i j d -> b i j', q, k) * self.scale

        if edge_mask is not None:
            max_neg_value = -torch.finfo(sim.dtype).max
            edge_mask = repeat(edge_mask, 'b i j -> (b h) i j', h = h)
            sim.masked_fill_(~edge_mask, max_neg_value)

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b i j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)

        return self.to_out(out)


class Encoder(nn.Module):
    def __init__(self,
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
            ) for _ in range(config.encoder_depth)
        ])

        self.checkpoint = config.checkpoint

    def block_checkpoint(self, layer):
        def checkpoint_forward(nodes, edges, translations, rotations, internal_edge_mask):
            nodes = layer(
                nodes,
                pairwise_repr = edges,
                rotations = rotations,
                translations = translations,
                edge_mask = internal_edge_mask
            )
            return nodes
        return checkpoint_forward

    def forward(self, nodes, edges, translations, rotations, internal_edge_mask):
        for layer in self.layers:
            if self.checkpoint:
                nodes = checkpoint.checkpoint(
                    self.block_checkpoint(layer),
                    nodes, edges, translations, rotations, internal_edge_mask
                )
            else:
                nodes = layer(
                    nodes, edges, translations, rotations, internal_edge_mask
                )
        return nodes


class CrossEncoderBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        edim,
        heads,
        graph_heads,
        scalar_key_dim,
        scalar_value_dim,
        point_key_dim,
        point_value_dim,
        graph_head_dim,
        ff_mult = 1,
        ff_num_layers = 3,     # in the paper, they used 3 layer transition (feedforward) block
        post_norm = True,      # in the paper, they used post-layernorm - offering pre-norm as well
    ):
        super().__init__()
        self.post_norm = post_norm

        self.attn_norm = nn.LayerNorm(dim)

        self.ipa_attn = InvariantPointAttention(
            dim, heads=heads, scalar_key_dim=scalar_key_dim,
            scalar_value_dim=scalar_value_dim, point_key_dim=point_key_dim,
            point_value_dim=point_value_dim, pairwise_repr_dim=edim
        )

        self.attn = Attention(dim, dim_head=graph_head_dim, heads=heads)

        self.ff_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult = ff_mult, num_layers = ff_num_layers)

    def forward(self, nodes, edges, translations, rotations, internal_edge_mask, external_edge_mask, **kwargs):
        post_norm = self.post_norm

        attn_input = nodes if post_norm else self.attn_norm(nodes)

        ipa_attn = self.ipa_attn(
            attn_input,
            pairwise_repr = edges,
            rotations = rotations,
            translations = translations,
            edge_mask = internal_edge_mask
        )

        attn = self.attn(
            attn_input,
            edge_mask=external_edge_mask
        )

        nodes = nodes + ipa_attn + attn
        nodes = self.attn_norm(nodes) if post_norm else nodes

        ff_input = nodes if post_norm else self.ff_norm(nodes)
        nodes = self.ff(ff_input) + nodes
        nodes = self.ff_norm(nodes) if post_norm else nodes

        return nodes


class CrossEncoder(nn.Module):
    def __init__(
            self,
            config
        ):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                CrossEncoderBlock(
                    dim = config.dim,
                    edim = config.edim,
                    heads = config.heads,
                    graph_heads = config.graph_heads,
                    scalar_key_dim = config.scalar_key_dim,
                    scalar_value_dim = config.scalar_value_dim,
                    point_key_dim = config.point_key_dim,
                    point_value_dim = config.point_value_dim,
                    graph_head_dim=config.graph_head_dim,
                ),
                Dense(
                  [2 * config.dim, 2 * config.dim, 2 * config.distance_pred_number_of_bins, config.distance_pred_number_of_bins]
                )
            ]) for _ in range(config.cross_encoder_depth)
        ])


        self.checkpoint = config.checkpoint

    def block_checkpoint(self, layer, to_distance):
        def checkpoint_forward(nodes, edges, translations, rotations, internal_edge_mask, external_edge_mask):
            nodes = layer(nodes, edges, translations, rotations, internal_edge_mask, external_edge_mask)
            endpoints = (repeat(nodes, 'b s c -> b z s c', z=nodes.size(1)),
                         repeat(nodes, 'b s c -> b s z c', z=nodes.size(1)))
            external_edges = torch.cat(endpoints, dim=-1)
            logits = to_distance(external_edges)
            return nodes, logits
        return checkpoint_forward

    def forward(self, nodes, edges, translations, rotations, internal_edge_mask, external_edge_mask):
        distance_logits_trajectory = []
        for (layer, to_distance) in self.layers:
            if self.checkpoint:
                nodes, logits = checkpoint.checkpoint(
                    self.block_checkpoint(layer, to_distance),
                    nodes, edges, translations,  rotations, internal_edge_mask, external_edge_mask
                )
            else:
                nodes, logits = layer(
                    nodes, edges, translations, rotations, internal_edge_mask, external_edge_mask
                )
            distance_logits_trajectory.append(logits)
        return nodes, distance_logits_trajectory


class Docker(nn.Module):
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
            for _ in range(config.docker_depth)
        ])

        self.checkpoint = config.checkpoint

    def block_checkpoint(self, layer, to_step):
        def checkpoint_forward(nodes, edges, translations, rotations, mask):
            nodes = layer(
                nodes,
                pairwise_repr = edges,
                rotations = rotations,
                translations = translations,
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


class Interactoformer(nn.Module):
    def __init__(
            self,
            config
        ):
        super().__init__()
        self.config = config

        self.node_crds_emb = nn.Linear(3, config.dim)

        self.sequence_embed = config.sequence_embed
        if config.sequence_embed:
            self.node_seq_emb = Dense([128, config.dim])
        else:
            self.node_token_emb = nn.Embedding(21, config.dim, padding_idx=0)

        self.circum_emb = nn.Linear(2, 12)
        # self.chain_emb = nn.Embedding(2, config.dim, padding_idx=0)

        self.backbone_dihedrals_emb = Dense([6 * 12, 64, 32])
        self.sidechain_dihedrals_emb = Dense([6 * 12, 64, 32])
        self.dihedral_emb = Dense([64, 64, config.dim])

        self.encoder = Encoder(config)
        self.cross_encoder = CrossEncoder(config)
        self.docker = Docker(config)

        self.distance_pred_number_of_bins = config.distance_pred_number_of_bins
        self.gaussian_noise = config.gaussian_noise
        self.unroll_steps = config.unroll_steps

        self.edge_norm_enc = partial(soft_one_hot_linspace, start=2,
                    end=config.distance_max_radius, number=config.distance_number_of_bins, basis='gaussian', cutoff=True)
        self.edge_angle_enc = partial(soft_one_hot_linspace, start=-1,
                    end=1, number=config.angle_number_of_bins, basis='gaussian', cutoff=True)

        edge_enc_size = config.distance_number_of_bins + 3 * config.angle_number_of_bins
        self.to_internal_edge = Dense([edge_enc_size, 2 * edge_enc_size, edge_enc_size, config.edim])

        self.structure_only = config.structure_only
        if self.structure_only:
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

        # embed node information
        if self.sequence_embed:
            nodes = self.node_seq_emb(batch.encs)
        else:
            nodes = self.node_token_emb(batch.seqs)

        angles = self.circum_emb(angle_to_point_in_circum(batch.angs))

        bb_dihedrals = rearrange(angles[..., :6, :], 'b s a h -> b s (a h)')
        bb_dihedrals = self.backbone_dihedrals_emb(bb_dihedrals)
        sc_dihedrals = rearrange(angles[..., 6:, :], 'b s a h -> b s (a h)')
        sc_dihedrals = self.sidechain_dihedrals_emb(sc_dihedrals)

        dihedrals = self.dihedral_emb(torch.cat((bb_dihedrals, sc_dihedrals), dim=-1))
        # chains = self.chain_emb(batch.chns)

        nodes = nodes + dihedrals # + chains

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
        coors = torch.zeros_like(batch.bck_crds, device=batch.bck_crds.device)
        rots = repeat(torch.eye(3), '... -> b n ...', b = batch_size, n = seq_len).to(batch.rots.device)

        # encode chains independently
        nodes = nodes + self.encoder(nodes, edges, coors, rots, internal_edge_mask)

        cross_encodings, logits_trajectory = self.cross_encoder(
            nodes, edges, coors, rots,
            internal_edge_mask,
            external_edge_mask
        )

        nodes = nodes + cross_encodings

        endpoints = (repeat(cross_encodings, 'b s c -> b z s c', z=seq_len),
                     repeat(cross_encodings, 'b s c -> b s z c', z=seq_len))
        cross_cat = torch.cat(endpoints, dim=-1)

        external_edges = self.to_external_edge(cross_cat)

        if self.structure_only:
            # if we are only dealing with structure docking, we give the distogram solution
            external_edges = self.to_external_edge(edge_structural_signal)
        else:
            # otherwise we build external pairwise representations from encodings
            logits_trajectory = rearrange(logits_trajectory, 't b s z l -> b s z t l')
            logits_trajectory = rearrange(logits_trajectory[external_edge_mask], 'e t l -> t e l')
            output['logit_traj'] = logits_trajectory

        edges = edges + external_edges * external_edge_mask[..., None]

        if self.config.distogram_only:
            return output

        # add gaussian noise if you're feeling like it
        if self.gaussian_noise:
            coors = coors + torch.normal(0, self.gaussian_noise, coors.shape, device=device)
        rots, coors = rots.clone().detach(), coors.clone().detach()

        # unroll network and learn to fix own mistakes
        with torch.no_grad():
            if is_training and self.unroll_steps:
                batch_steps = torch.randint(0, self.unroll_steps, [batch_size])
                max_step = torch.max(batch_steps).item()
                for step in range(max_step):
                    chosen = batch_steps >= step
                    _, translations_timeseries, rotations_timeseries = self.docker(
                        nodes[chosen], edges[chosen], coors[chosen], rots[chosen],
                        mask=batch.node_pad_mask[chosen]
                    )
                    rots[chosen] = rotations_timeseries[-1]
                    coors[chosen] = translations_timeseries[-1]

        # produce timeseries
        translations, rotations  = [ coors ], [ rots ]
        for _ in range(1 if is_training else self.eval_steps):
            nodes_, translations_timeseries, rotations_timeseries = self.docker(nodes, edges, coors, rots, mask=batch.node_pad_mask)
            translations.extend(translations_timeseries)
            rotations.extend(rotations_timeseries)
            rots = rotations_timeseries[-1].detach()
            coors = translations_timeseries[-1].detach()

        output['translations'] = rearrange(torch.stack(translations, dim=0), 't b n e -> b n t e')
        output['rotations'] = rearrange(torch.stack(rotations, dim=0), 't b n p q -> b n t p q')

        return output
