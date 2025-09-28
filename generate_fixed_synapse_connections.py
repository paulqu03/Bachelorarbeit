'''
Dieses Programm generiert eine Verbindungslisten, um die Modellstruktur zu fixieren.
Dabei werden für jede Gruppe von Synapsen Verbindungslisten erstellt für die Wahrsscheinlichkeiten von 5% bis 50%.
 '''

import numpy as np
import pickle
from pathlib import Path
from typing import Tuple

def generate_connection_list_pair(
    n_source: int,
    n_target: int,
    probability: float,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Erzeuge Verbindungslisten i und j für Synapsen mit fixer Struktur.
    i enthält jeden Quellneuronenindex so oft, wie er Verbindungen erzeugt.
    j enthält zugehörige Zielneuronen (zufällig gezogen).
    """
    if seed is not None:
        np.random.seed(seed)

    n_connections_per_source = int(np.round(n_target * probability))
    i_list, j_list = [], []

    for i in range(n_source):
        j_targets = np.random.choice(n_target, size=n_connections_per_source, replace=True)
        i_list.extend([i] * n_connections_per_source)
        j_list.extend(j_targets)

    return np.array(i_list, dtype=int), np.array(j_list, dtype=int)


# Verbindungswahrscheinlichkeiten: 5% bis 50% (in 1%-Schritten)
probs = np.arange(0.05, 0.501, 0.01)

# Definiere alle Synapsengruppen (Name, n_pre, n_post)
synapse_groups = {
    'S_P_E1':   (100, 100), 
    'S_E1_I':   (100, 20),
    'S_E2_I':   (100, 20),
    'S_I_E1':   (20, 100),
    'S_I_E2':   (20, 100),
    'S_E1_E1':  (100, 100),
    'S_E2_E2':  (100, 100),
    'S_I_I':    (20, 20),
    'S_StE1':   (20, 100),
    'S_StE2':   (20, 100),
    'S_P_E2':   (100, 100),
}

# Speicherverzeichnis
base_path = Path("./connection_lists")
base_path.mkdir(parents=True, exist_ok=True)

# Erzeuge und speichere die Listen
for name, (n_pre, n_post) in synapse_groups.items():
    group_dict = {}
    for p in probs:
        i_list, j_list = generate_connection_list_pair(n_pre, n_post, p)
        group_dict[round(p, 2)] = (i_list, j_list)
    print(group_dict)
    with open(base_path / f"{name}.pkl", 'wb') as f:
        pickle.dump(group_dict, f)

