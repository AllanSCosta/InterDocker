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
from collections import defaultdict


class Interactoformer(nn.Module):
    def __init__(
            self,
            config
        ):

        super().__init__()

        self.config = config
        self.encoder = Encoder(config)
        self.cross_encoder = CrossEncoder(config)

        self.to_interface = Dense([2 * config.dim, 2 * config.dim, config.dim, config.edim], checkpoint=config.checkpoint_denses)
        self.eval_steps = config.eval_steps

        self.cross_unroll = config.cross_unroll
        self.dock_unroll = config.dock_unroll
        self.stochastic_embed = Dense([config.dim, config.dim, config.dim])


    def forward(self, batch, is_training):
        output = {}

        pair, edges = batch[:2], batch[2]
        n, m = [batch.encs.size(1) for batch in pair]
        b = batch[0].encs.size(0)

        # ====== INDEPENDENT ENCODING ENRICHES NODES ======
        pair = [self.encoder(batch) for batch in pair]

        # ====== ADD STOCHASTICITY TO CROSS =======
        p1, p2 = pair
        s = [self.stochastic_embed(torch.randn(nodes.size(), device=nodes.device)) for nodes in (p1.nodes, p2.nodes)]

        # ====== CROSS CAT FOR EDGE PRIOR BELIEF ========
        cross_cat = repeat(p1.nodes + s[0], 'b n h -> b n m h', m=m), repeat(p2.nodes + s[1], 'b m h -> b n m h', n=n)
        cross_cat = torch.cat(cross_cat, dim=-1)

        # ====== MUTUAL ENCODING ITERATIVELY ENRICHES EXTERNAL EDGES ======
        cross_cat_size = cross_cat.size()[:-1]
        hypothesis = (
            torch.zeros((*cross_cat_size, self.config.distance_pred_number_of_bins + 3 * self.config.angle_pred_number_of_bins), device=cross_cat.device)
        )

        with torch.no_grad():
            if is_training and self.cross_unroll:
                batch_steps = torch.randint(0, self.cross_unroll, [b])
                max_step = torch.max(batch_steps).item()
                for step in range(max_step):
                    chosen = batch_steps >= step

                    internal_trajectory = self.cross_encoder(self.to_interface(cross_cat[chosen]), hypothesis[chosen])

                    distances = internal_trajectory['distance'][-1]
                    angles = F.softmax(rearrange(internal_trajectory['angles'][-1], '... (a l) -> ... a l', a=3), dim=-1)
                    solution = (distances, rearrange(angles, '... a l -> ... (a l)'))


                    hypothesis[chosen] = hypothesis[chosen] + torch.cat(solution, dim=-1)



        trajectories = defaultdict(list)
        for _ in range(1 if is_training else self.eval_steps):
            internal_trajectory = self.cross_encoder(self.to_interface(cross_cat), hypothesis)

            for key, value in internal_trajectory.items():
                trajectories[key].append(value)

            distances = F.softmax(internal_trajectory['distance'][-1], dim=-1)
            angles = F.softmax(rearrange(internal_trajectory['angles'][-1], '... (a l) -> ... a l', a=3), dim=-1)
            solution = (distances, rearrange(angles, '... a l -> ... (a l)'))

            hypothesis = hypothesis + torch.cat(solution, dim=-1).detach()

        for key, value in trajectories.items():
            trajectories[key] = rearrange(torch.cat(value, dim=0), 't b ... -> b t ...')

        output['logit_traj'] = trajectories

        return output


class Encoder(nn.Module):
    def __init__(self,
            config
        ):
        super().__init__()
        self.config = config

        self.sequence_embed = config.sequence_embed
        if config.sequence_embed:
            self.node_seq_emb = Dense([128, config.dim])
        else:
            self.node_token_emb = nn.Embedding(21, config.dim, padding_idx=0)

        self.circum_emb = nn.Linear(2, 12)
        self.chain_emb = nn.Embedding(3, config.dim, padding_idx=0)

        self.backbone_dihedrals_emb = Dense([6 * 12, 6 * 12, 64], checkpoint=config.checkpoint_denses)
        self.sidechain_dihedrals_emb = Dense([6 * 12, 6 * 12, 64], checkpoint=config.checkpoint_denses)
        self.dihedral_emb = Dense([128, 128, config.dim], checkpoint=config.checkpoint_denses)

        self.distance_pred_number_of_bins = config.distance_pred_number_of_bins
        self.gaussian_noise = config.gaussian_noise

        self.edge_norm_enc = partial(soft_one_hot_linspace, start=2,
                    end=config.distance_max_radius, number=config.distance_number_of_bins, basis='gaussian', cutoff=True)
        self.edge_angle_enc = partial(soft_one_hot_linspace, start=-1,
                    end=1, number=config.angle_number_of_bins, basis='gaussian', cutoff=True)

        edge_enc_size = config.distance_number_of_bins + 3 * config.angle_number_of_bins
        self.to_internal_edge = Dense([edge_enc_size, 2 * edge_enc_size, edge_enc_size, config.edim], checkpoint=config.checkpoint_denses)
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

        self.checkpoint = config.checkpoint_encoder

    def forward_constructor(self, layer):
        def forward_function(nodes, edges, translations, rotations, edge_mask):
            nodes = layer(
                nodes,
                pairwise_repr = edges,
                rotations = rotations,
                translations = translations,
                edge_mask=edge_mask,
            )
            return nodes
        return forward_function

    def forward(self, batch):
        node_pad_mask = batch.node_pad_mask
        edge_pad_mask = batch.edge_pad_mask
        batch_size, seq_len = batch.seqs.size()
        device = batch.seqs.device

        # ====== EMBED SEQUENCE ======
        if self.sequence_embed: sequence = self.node_seq_emb(batch.encs)
        else: sequence = self.node_token_emb(batch.seqs)

        # ====== EMBED ANGLES ======
        angles = self.circum_emb(angle_to_point_in_circum(batch.angs))

        bb_dihedrals = rearrange(angles[..., :6, :], 'b s a h -> b s (a h)')
        bb_dihedrals = self.backbone_dihedrals_emb(bb_dihedrals)
        sc_dihedrals = rearrange(angles[..., 6:, :], 'b s a h -> b s (a h)')
        sc_dihedrals = self.sidechain_dihedrals_emb(sc_dihedrals)

        dihedrals = self.dihedral_emb(torch.cat((bb_dihedrals, sc_dihedrals), dim=-1))

        # ====== BREAK HOMOMERIC SYMMETRY ======
        chains = self.chain_emb(batch.chns.type(torch.long))

        # BASE NODE REPRESENTATION
        # =========================
        nodes = sequence + dihedrals + chains
        # =========================

        # ====== DEFINE EDGES  ======
        edges = torch.zeros(batch_size, seq_len, seq_len, self.config.edim, device=device)

        # ====== PRODUCE STRUCTURAL SIGNALS  ======
        max_radius = self.config.distance_max_radius
        edge_distance = batch.edge_distance
        edge_angles = batch.edge_angles

        if len(edge_distance.size()) == 4:
            edge_distance = edge_distance[..., 0]
            edge_angles = edge_angles[..., 0, :]

        dist_signal = self.edge_norm_enc(edge_distance)
        angle_signal = self.edge_angle_enc(edge_angles)
        angle_signal = rearrange(angle_signal, 'b s z a c -> b s z (a c)')
        angle_signal[edge_distance < max_radius] = 0

        # ====== BASE INTERNAL EDGES REPRESENTATION  ======
        edge_structural_signal = torch.cat((dist_signal, angle_signal), dim=-1)
        edges = self.to_internal_edge(edge_structural_signal)

        for layer in self.layers:
            forward = self.forward_constructor(layer)
            if self.checkpoint: forward = partial(checkpoint.checkpoint, forward)
            nodes = forward(nodes, edges, batch.bck_crds, batch.rots, edge_pad_mask)

        batch.nodes = nodes
        return batch


class CrossEncoder(nn.Module):
    def __init__(
            self,
            config
        ):
        super().__init__()

        self.layers = nn.ModuleList([
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
                kernel_size=config.kernel_size,
                num_conv_per_layer=config.num_conv_per_layer,
            )
            for _ in range(config.cross_encoder_depth)
        ])

        self.predict_angles = config.predict_angles

        if config.real_value:
            projectors = []
            for _ in range(config.cross_encoder_depth):
                to_projection = dict(vector=Dense([config.edim, 1]), error=Dense([config.edim, 1]))
                projectors.append(nn.ModuleDict(to_projection))
            self.projectors = nn.ModuleList(projectors)
        else:
            binners = []
            for _ in range(config.cross_encoder_depth):
                to_bins = dict()
                to_bins['distance'] = Dense([config.edim, config.distance_pred_number_of_bins])
                if self.predict_angles:
                    to_bins[f'angles'] = Dense([config.edim, 3 * config.angle_pred_number_of_bins])
                binners.append(nn.ModuleDict(to_bins))
            self.projectors = nn.ModuleList(binners)

        self.checkpoint = config.checkpoint_cross_encoder
        self.predict_angles = config.predict_angles

        self.from_dcoords = Dense([config.distance_pred_number_of_bins + 3 * config.angle_pred_number_of_bins, 2 * config.edim, config.edim])
        self.edge_update = Dense([config.edim, config.edim, config.edim])

    def forward_constructor(self, layer):
        def forward_function(*args):
            output = layer(*args)
            return output
        return forward_function

    def forward(self, cross_edges, dcoords):

        trajectory = defaultdict(list)

        cross_edges = self.from_dcoords(dcoords) + cross_edges
        cross_edges = self.edge_update(cross_edges)

        for (layer, projector) in zip(self.layers, self.projectors):
            forward = self.forward_constructor(layer)
            if self.checkpoint:
                forward = partial(checkpoint.checkpoint, forward)

            cross_edges = forward(cross_edges)

            bins = dict()
            for (name, projection_function) in projector.items():
                bins[name] = projection_function(cross_edges)

            for key, value in bins.items():
                trajectory[key].append(value)

        for key, value in trajectory.items():
            trajectory[key] = torch.stack(value, dim=0)

        return trajectory

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
        kernel_size,
        num_conv_per_layer,
    ):
        super().__init__()

        self.attn_norm = nn.LayerNorm(dim)
        self.ipa_attn = InvariantPointAttention(
            dim, heads=heads, scalar_key_dim=scalar_key_dim,
            scalar_value_dim=scalar_value_dim, point_key_dim=point_key_dim,
            point_value_dim=point_value_dim, pairwise_repr_dim=edim
        )

        self.attn = Attention(dim, dim_head=graph_head_dim, heads=heads)

        self.ff_norm = nn.LayerNorm(dim)
        self.ff = Dense([dim, dim])
        self.edges_norm = nn.LayerNorm(edim)

        self.to_edge = Dense([dim, dim, edim])
        self.to_edge_update = Dense2D([edim] + [edim] * num_conv_per_layer, kernel_size)
        self.to_edge_update_1d = Dense([edim, edim, edim], kernel_size)

    def forward(self, cross_edges, **kwargs):

        # for batch in pair:
        #     attn_input = self.attn_norm(batch.nodes)
        #     ipa_attn = self.ipa_attn(
        #         attn_input,
        #         pairwise_repr = edges,
        #         rotations = rotations,
        #         translations = translations,
        #         edge_mask = internal_edge_mask
        #     )

        # attn = self.attn(
        #     attn_input,
        #     edge_mask = external_edge_mask
        # )
        # nodes = nodes + ipa_attn + attn
        # nodes = self.ff(nodes) + nodes
        #
        # endpoints = (repeat(nodes, 'b s c -> b z s c', z=nodes.size(1)),
        #              repeat(nodes, 'b s c -> b s z c', z=nodes.size(1)))

        # update_mask = external_edge_mask[..., None].float()
        # external_edges = self.to_edge(endpoints[0] - endpoints[1])
        # edges = edges + update_mask * external_edges
        # cross_edges = self.to_edge_update_1d(cross_edges)
        res = cross_edges
        cross_edges = rearrange(cross_edges, 'b s z c -> b c s z')
        cross_edges = rearrange(self.to_edge_update(cross_edges), 'b c s z -> b s z c') + res
        # edge_update = rearrange(edge_update, )
        # edges = edges + update_mask * external_edges

        return cross_edges


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


class Dense2D(nn.Module):
    def __init__(self, layer_structure, kernel_size=9):
        super().__init__()
        layers = []
        for idx, (back, front) in enumerate(zip(layer_structure[:-1],
                                            layer_structure[1:])):
            layers.append(nn.Conv2d(back, front, kernel_size=kernel_size, padding='same'))
            layers.append(nn.BatchNorm2d(front))
            if idx < len(layer_structure) - 2: layers.append(nn.LeakyReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


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

    def forward_constructor(self, layers):
        def forward_function(x):
            return layers(x)
        return forward_function

    def forward(self, x):
        forward = self.forward_constructor(self.layers)
        if self.checkpoint: forward = partial(checkpoint.checkpoint, forward)
        return forward(x)
