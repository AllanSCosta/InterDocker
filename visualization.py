import matplotlib.pyplot as plt
import sidechainnet as scn
import wandb
import py3Dmol
import torch
from mp_nerf.protein_utils import build_scaffolds_from_scn_angles
from mp_nerf.proteins import sidechain_fold, ca_bb_fold

import plotly
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, splprep, splev
import torch
from plotly.subplots import make_subplots

import gif
gif.options.matplotlib["dpi"] = 300

import plotly.graph_objects as go
from torch.linalg import norm


plt.rcParams.update({
    "figure.facecolor":  (0.0, 0.0, 0.0, 0.0),
    "axes.facecolor":    (0.0, 0.0, 0.0, 0.0),
    "savefig.facecolor": (0.0, 0.0, 0.0, 0.0),
})


def plot_predictions(dist, pred_dist, conct, pred_conct):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for idx, img in enumerate([dist, pred_dist]):
        axes[0][idx].imshow(-img)
    for idx, img in enumerate([conct, pred_conct]):
        axes[1][idx].imshow(img)
    return fig


gnd_colors = ('#43A047', '#7CB342')
pred_colors = ('#673AB7', '#3F51B5')

def plot_aligned_structures(seq, gnd_crd, pred_crd, angs, boundary):
    pred_bb = ca_bb_fold(pred_crd.unsqueeze(0))[0]
    gnd_bb = ca_bb_fold(gnd_crd.unsqueeze(0))[0]

    gnd_scaffolds = build_scaffolds_from_scn_angles(seq, angles=angs, device="cpu")
    gnd_coords, _ = sidechain_fold(wrapper = gnd_bb.clone(), **gnd_scaffolds, c_beta = 'torsion')

    pred_scaffolds = build_scaffolds_from_scn_angles(seq, angles=angs, device="cpu")
    pred_coords, _ = sidechain_fold(wrapper = pred_bb.clone(), **pred_scaffolds, c_beta = 'torsion')

    gnd_struct = scn.StructureBuilder(seq, gnd_coords.reshape(-1, 3))
    pred_struct = scn.StructureBuilder(seq, pred_coords.reshape(-1, 3) + 0.1)

    gnd_pdb = gnd_struct.to_pdbstr()
    pred_pdb = pred_struct.to_pdbstr()

    view = py3Dmol.view(width=400, height=300)
    view.setBackgroundColor(0x000000,0)
    view.addModelsAsFrames(gnd_pdb)
    view.addModelsAsFrames(pred_pdb)

    view.setStyle({'model': 0, 'resi': [f'0-{boundary}']}, {'cartoon': {'color': gnd_colors[0], 'opacity': 0.7}})
    view.setStyle({'model': 0, 'resi': [f'{boundary}-{len(gnd_crd)}']}, {'cartoon': {'color': gnd_colors[1], 'opacity': 0.7}})

    view.setStyle({'model': 1, 'resi': [f'0-{boundary}']}, {'cartoon': {'color': pred_colors[0]}, 'stick': {'radius': .15, 'color': pred_colors[0]}}, )
    view.setStyle({'model': 1, 'resi': [f'{boundary}-{len(gnd_crd)}']}, {'cartoon': {'color': pred_colors[1]}, 'stick': {'radius': .15, 'color': pred_colors[1]}}, )

    view.zoomTo()
    view.rotate(1, 'y')

    return view

def plot_aligned_timeseries(seq, timeseries, boundary):
    models = ""
    view = py3Dmol.view(width=400, height=300)
    view.setBackgroundColor(0x000000,0)

    for i, (gnd_crd, pred_crd, angs) in enumerate(timeseries):
        pred_bb = ca_bb_fold(pred_crd.unsqueeze(0))[0]
        gnd_bb = ca_bb_fold(gnd_crd.unsqueeze(0))[0]

        scaffolds = build_scaffolds_from_scn_angles(seq + seq, angles=torch.cat((angs, angs)), device="cpu")
        coords, _ = sidechain_fold(wrapper = torch.cat((gnd_bb, pred_bb)).clone(), **scaffolds, c_beta = 'torsion')
        structures = scn.StructureBuilder(seq + seq, coords.reshape(-1, 3))

        pdb = structures.to_pdbstr()
        models += "MODEL " + str(i) + "\n" + pdb + "ENDMDL\n"

    view.addModelsAsFrames(models)

    view.setStyle({'model': -1, 'resi': [f'0-{boundary}']}, {'cartoon': {'color': gnd_colors[0], 'opacity': 0.7}, 'stick': {'radius': .15, 'color': gnd_colors[0], 'opacity': 0.7}})
    view.setStyle({'model': -1, 'resi': [f'{boundary}-{len(gnd_crd)-1}']}, {'cartoon': {'color': gnd_colors[1], 'opacity': 0.7}, 'stick': {'radius': .15, 'color': gnd_colors[1], 'opacity': 0.7}})
    view.setStyle({'model': -1, 'resi': [f'{len(gnd_crd)}-{len(gnd_crd) + boundary - 1}']}, {'cartoon': {'color': pred_colors[0]}, 'stick': {'radius': .15, 'color': pred_colors[0]}})
    view.setStyle({'model': -1, 'resi': [f'{len(gnd_crd) + boundary}-{ 2 * len(gnd_crd)}']}, {'cartoon': {'color': pred_colors[1]}, 'stick': {'radius': .15, 'color': pred_colors[1]}})


    view.zoomTo()
    view.animate({'loop': "forward"})
    return view
