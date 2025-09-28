'''
Dieses Programm kann dazu genutzt werden, die fertigen Heatmaps zu nochmal überarbeiten, ohne einen kompletten Grid Search staarten zu müssen.
'''
import pickle
import numpy as np
from pathlib import Path
from brian2 import *

# Pfad zur Pickle-Datei
base_path = Path('./experiments/decision_simulation/grid_search_tau_I/v_post_E_fixed')  # Passe das an deinen Pfad an
heatmap_dir_new = base_path / 'reworked_heatmaps'
heatmap_dir = base_path / 'heatmaps'
pickle_path = heatmap_dir / 'heatmaps_data.pkl'

TAU_I_VALUES  = np.arange(1, 11) * ms           # 1 .. 10 ms
GRID_VALUES   = np.arange(0.5, 7.51, 0.5)*mV  # 0.05 .. 0.25 

# Pickle-Datei laden
with open(pickle_path, 'rb') as f:
    heatmaps = pickle.load(f)

# Zugriff auf einzelne Matrizen
accuracy_map = heatmaps['accuracy']
clarity_map = heatmaps['clarity']
eff_inhib_map = heatmaps['eff_inhibition']
win_clarity_map = heatmaps['winner_clarity']
lose_clarity_map = heatmaps['loser_clarity']
sensitivity_map = heatmaps['sensitivity']
over_sens_map = heatmaps['over_sensitivity']


def save_heatmap(matrix: np.ndarray, title: str, cmap: str, fname: str,
                 percent=False, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, origin='lower', aspect='auto', cmap=cmap,
                   extent=(GRID_VALUES[0]/mV, GRID_VALUES[-1]/mV,
                           TAU_I_VALUES[0]/ms, TAU_I_VALUES[-1]/ms),
                   vmin=vmin, vmax=vmax)
    ax.set_xlabel('v_E_I (mV)', fontsize=16)
    ax.set_ylabel('tau_I (ms)', fontsize=16)
    ax.set_title(title + ' der Grid Search zwischen tau_I und v_E_I')
    ax.tick_params(labelsize=14)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('%' if percent else 'value', fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    fig.tight_layout()
    fig.savefig(heatmap_dir / fname)
    plt.close(fig)

# Save plots with scale limits
save_heatmap(accuracy_map,      'Genauigkeits - Heatmap',                    'RdYlGn',   'heatmap_accuracy.png',                     percent=True)
save_heatmap(clarity_map,       'Eindeutigkeits - Heatmap',                     'plasma',   'heatmap_activity_difference.png',          vmin=0, vmax=140000)
save_heatmap(eff_inhib_map,     'Effektive Inhibitions - Heatmap',        'seismic',  'heatmap_eff_inhib.png',                    vmin=-3000, vmax=3000)
save_heatmap(win_clarity_map,   'Winner Clarity Heatmap',              'plasma',   'heatmap_win_clarity.png',                  vmin=10000, vmax=140000)
save_heatmap(lose_clarity_map,  'Loser Clarity Heatmap',               'plasma',   'heatmap_lose_clarity.png',                 vmin=0, vmax=3000)
save_heatmap(sensitivity_map,   'Stabilitäts - Heatmap',                   'RdYlGn_r', 'heatmap_decision_making_competition.png',  percent=True)
save_heatmap(over_sens_map,     'Vorentscheidungs - Heatmap',        'RdYlGn_r', 'heatmap_pre_stimuli_decision.png',         percent=True)