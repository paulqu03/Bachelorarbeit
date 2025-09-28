"""
Dieses Programm simuliert Entscheidungen unter der finalen Parametereinstellung. Dabei wird die Entscheidung
20-mal simuliert, damit die Entsscheidung über die Metriken ausgewertet werden kann. Für jede Simulation werden
dabei ein Spike-Raster-Plot, ein Spike-Activity-Plot und ein Stimulus-Plot erstellt. Der Metrik Report der Entscheidung
wird zum Schluss in einer Text Datei im Ordner gespeichert.

Vor der Benutzung des Programms muss überprüft werden, ob der base_path richtig eingestellt issst, sowie die jeweiligen Paths
der connection_list. Sollte der connection_list Ordner noch nicht existieren oder leer sein, so muss erst das Programm 
'generate_fixed_synapse_connections.py' ausgeführt werden. Zudem ssind die Metriken momentan darauf ausgelegt mit einem E1 Bias zu rechnen.
"""

import os
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

# Hilfsfunktionen

def create_directory(path: str | Path):
    """Create directory *path* (incl. parents) if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def pop_rate(spike_monitor: SpikeMonitor, t0, t1, n_neurons: int) -> float:
    """Return mean firing rate (Hz) in [t0, t1)."""
    count = np.sum((spike_monitor.t >= t0) & (spike_monitor.t < t1))
    return count / ((t1 - t0) / second) /100


#Setup

base_path = Path("./ergebnisse/final_model")
create_directory(base_path)

#Basisparametereinsstellung

#Neuronen Parameter
N_E1, N_E2, N_I = 100, 100, 20

tau_E  = 19*ms
tau_I = 5*ms

v_th_E = 15*mV
v_th_I  = 15*mV 

v_reset = 5*mV

t_ref = 0*ms

# Stimulus / runtime
stim_on, stim_off = 500*ms, 1500*ms
runtime = 2000*ms

mu0, mu1 = 100*Hz, 85*Hz        # E1‑Vorteil
sigma = 4*Hz
stim_intv = 50*ms

# Hintergrundrausschen
rate_bg = 55*Hz
v_bg = 1.2*mV

# Synaptische Gewichte
v_StE = 3.8*mV
v_E_I = 2.3*mV
v_I_E  = 2.3*mV
v_E_E = 0.5*mV
v_I_I = 0.01*mV

#Konnektivitätswahrscheinlichkeiten
p_P = p_P = 0.05
p_E_E = 0.35
p_I_I = 0.05
p_E_I = 0.15
p_I_E = 0.11
p_StE = 0.05

#Modellstruktur laden
with open('./connection_lists/S_E1_E1.pkl', 'rb') as f:
    conn_dict = pickle.load(f)

S_E1_E1_i, S_E1_E1_j = conn_dict[p_E_E]

with open('./connection_lists/S_E1_I.pkl', 'rb') as f:
    conn_dict_pe1 = pickle.load(f)

S_E1_I_i, S_E1_I_j = conn_dict_pe1[p_E_I]

with open('./connection_lists/S_E2_E2.pkl', 'rb') as f:
    conn_dict = pickle.load(f)

S_E2_E2_i, S_E2_E2_j = conn_dict[p_E_E]

with open('./connection_lists/S_E2_I.pkl', 'rb') as f:
    conn_dict_pe2 = pickle.load(f)

S_E2_I_i, S_E2_I_j = conn_dict_pe2[p_E_I]

with open('./connection_lists/S_I_E1.pkl', 'rb') as f:
    conn_dict_ie1 = pickle.load(f)

S_I_E1_i, S_I_E1_j = conn_dict_ie1[p_I_E]

with open('./connection_lists/S_I_E2.pkl', 'rb') as f:
    conn_dict_ie2 = pickle.load(f)

S_I_E2_i, S_I_E2_j = conn_dict_ie2[p_I_E]
    
with open('./connection_lists/S_I_I.pkl', 'rb') as f:
    conn_dict = pickle.load(f)

S_I_I_i, S_I_I_j = conn_dict[p_I_I]

with open('./connection_lists/S_P_E1.pkl', 'rb') as f:
    conn_dict = pickle.load(f)

S_P_E1_i, S_P_E1_j = conn_dict[p_P]

with open('./connection_lists/S_P_E2.pkl', 'rb') as f:
    conn_dict = pickle.load(f)

S_P_E2_i, S_P_E2_j = conn_dict[p_P]

with open('./connection_lists/S_StE1.pkl', 'rb') as f:
    conn_dict = pickle.load(f)

S_StE1_i, S_StE1_j = conn_dict[p_StE]
    
with open('./connection_lists/S_StE2.pkl', 'rb') as f:
    conn_dict = pickle.load(f)

S_StE2_i, S_StE2_j = conn_dict[p_StE]

#Netzwerkaufbau
E1 = NeuronGroup(N_E1, 'dv/dt = -v / tau_E : volt (unless refractory)',
                threshold='v >= v_th_E', reset='v = v_reset',
                method='euler', refractory=t_ref, name='E1')
E2 = NeuronGroup(N_E2, 'dv/dt = -v / tau_E : volt (unless refractory)',
                threshold='v >= v_th_E', reset='v = v_reset',
                method='euler', refractory=t_ref, name='E2')
I  = NeuronGroup(N_I,  'dv/dt = -v / tau_I : volt (unless refractory)',
                threshold='v >= v_th_I', reset='v = v_reset',
                method='euler', refractory=t_ref, name='I')

E1.v = 'v_th_E-5*mV'
E2.v = 'v_th_E-5*mV'
I.v = 'v_th_I-5*mV'

P_E1 = PoissonGroup(100, rates=rate_bg, name='P_E1')
P_E2 = PoissonGroup(100, rates=rate_bg, name='P_E2')
P_I = PoissonGroup(20, rates=rate_bg, name='P_I')

S_P_E1 = Synapses(P_E1, E1, on_pre='v += v_bg', name='S_P_E1')
S_P_E2 = Synapses(P_E2, E2, on_pre='v += v_bg', name='S_P_E2')
S_P_E1.connect(i=S_P_E1_i, j=S_P_E1_j)
S_P_E2.connect(i=S_P_E2_i, j=S_P_E2_j)

S_P_I = Synapses(P_I, I, on_pre='v += v_post_poisson', name='S_P_I')
S_P_I.connect(p=p_P)

StE1 = PoissonGroup(20, rates=0*Hz, name='StE1')
StE2 = PoissonGroup(20, rates=0*Hz, name='StE2')

StE1.run_regularly(
    "rates = int(t >= stim_on and t < stim_off) * "
    "(mu0 + sigma*randn())",
    dt=stim_intv
)
StE2.run_regularly(
    "rates = int(t >= stim_on and t < stim_off) * "
    "(mu1 + sigma*randn())",
    dt=stim_intv
)

S_StE1 = Synapses(StE1, E1, on_pre='v += v_StE', delay=0.5*ms, name='S_StE1')
S_StE2 = Synapses(StE2, E2, on_pre='v += v_StE', delay=0.5*ms, name='S_StE2')
S_StE1.connect(i=S_StE1_i, j=S_StE1_j)
S_StE2.connect(i=S_StE2_i, j=S_StE2_j)

S_E1_E1 = Synapses(E1, E1, on_pre='v += v_E_E', delay=0.5*ms, name='S_E1_E1')
S_E2_E2 = Synapses(E2, E2, on_pre='v += v_E_E', delay=0.5*ms, name='S_E2_E2')
S_I_I   = Synapses(I, I,   on_pre='v -= v_I_I', delay=0.5*ms, name='S_I_I')

S_E1_E1.connect(i=S_E1_E1_i, j=S_E1_E1_j)
S_E2_E2.connect(i=S_E2_E2_i, j=S_E2_E2_j)
S_I_I.connect(i=S_I_I_i, j=S_I_I_j)

S_E1_I = Synapses(E1, I, on_pre='v += v_E_I', delay=0.5*ms, name='S_E1_I')
S_E2_I = Synapses(E2, I, on_pre='v += v_E_I', delay=0.5*ms, name='S_E2_I')
S_E1_I.connect(i=S_E1_I_i, j=S_E1_I_j)
S_E2_I.connect(i=S_E2_I_i, j=S_E2_I_j)

S_I_E1 = Synapses(I, E1, on_pre='v -= v_I_E', delay=0.5*ms, name='S_I_E1')
S_I_E2 = Synapses(I, E2, on_pre='v -= v_I_E', delay=0.5*ms, name='S_I_E2')
S_I_E1.connect(i=S_I_E1_i, j=S_I_E1_j)
S_I_E2.connect(i=S_I_E2_i, j=S_I_E2_j)

#Monitore
PR_E1, PR_E2, PR_I = (PopulationRateMonitor(grp, name=f'PR_{grp.name}')
                        for grp in (E1, E2, I))
SM_E1, SM_E2, SM_I = (SpikeMonitor(grp, name=f'SM_{grp.name}')
                        for grp in (E1, E2, I))
StM_E1 = StateMonitor(StE1, 'rates', record=0, dt=1*ms)
StM_E2 = StateMonitor(StE2, 'rates', record=0, dt=1*ms)

#Netzwerk für Simulation speichern
net = Network()
net.add(E1, E2, I, P_E1, P_E2, 
        S_P_E1, S_P_E2, StE1, StE2, S_StE1, S_StE2, 
        S_E1_E1, S_E2_E2, S_E1_I, S_E2_I, S_I_I, S_I_E1, S_I_E2, S_P_I,
        PR_E1, PR_E2, PR_I, 
        SM_E1, SM_E2, SM_I, StM_E1, StM_E2)
net.store()

#Simulation vorbereiten
TAU_I_VALUES  = np.arange(1, 11) * ms           # 1 .. 10 ms
GRID_VALUES   = np.arange(0.5, 7.51, 0.5)*mV  # 0.05 .. 0.25 
N_RUNS        = 20
BIN_SIZE      = 15*ms                           # smoothing for plots

n_tau = len(TAU_I_VALUES)
n_val = len(GRID_VALUES)

accuracy_map   = np.zeros((n_tau, n_val))
clarity_map    = np.zeros((n_tau, n_val))
eff_inhib_map  = np.zeros((n_tau, n_val))
win_clarity_map = np.zeros((n_tau, n_val))
lose_clarity_map= np.zeros((n_tau, n_val))
sensitivity_map= np.zeros((n_tau, n_val))
over_sens_map  = np.zeros((n_tau, n_val))

true_pos, comp_cnt, predec_cnt = 0, 0, 0
delta_abs_list, i_eff_list = [], []
win_act_list, lose_act_list = [], []

#20 Simulation
for run_idx in range(N_RUNS):
    net.restore()
    net.run(runtime)

    A_E1 = pop_rate(SM_E1, stim_on, stim_off, N_E1)
    A_E2 = pop_rate(SM_E2, stim_on, stim_off, N_E2)
    B_E1 = (
        pop_rate(SM_E1, 0*ms, stim_on, N_E1) * (stim_on/runtime) +
        pop_rate(SM_E1, stim_off, runtime, N_E1) * ((runtime - stim_off)/runtime)
    ) / ((stim_on/runtime) + ((runtime - stim_off)/runtime))
    B_E2 = (
        pop_rate(SM_E2, 0*ms, stim_on, N_E2) * (stim_on/runtime) +
        pop_rate(SM_E2, stim_off, runtime, N_E2) * ((runtime - stim_off)/runtime)
    ) / ((stim_on/runtime) + ((runtime - stim_off)/runtime))

    delta = A_E1 - A_E2
    delta_abs_list.append(abs(delta))

    # Genauigkeit
    if delta >= 30:    # True Positive (E1 sollte gewinnen)
        true_pos += 1

    # Effektive inhibition
    if delta >= 0:   
        i_eff_list.append(A_E2 - B_E2)
        win_act_list.append(A_E1)
        lose_act_list.append(A_E2)
    else:           
        i_eff_list.append(A_E1 - B_E1)
        win_act_list.append(A_E2)
        lose_act_list.append(A_E1)

    # Stabilität
    winners = []
    for step in range(10):
        t0 = stim_on + step*100*ms
        t1 = t0 + 100*ms
        a1 = pop_rate(SM_E1, t0, t1, N_E1)
        a2 = pop_rate(SM_E2, t0, t1, N_E2)
        winners.append(0 if a1 >= a2 else 1)  
    if len(set(winners)) > 1:
        comp_cnt += 1

    # Vorentscheidung
    pre_a1 = pop_rate(SM_E1, 0*ms, stim_on, N_E1)
    pre_a2 = pop_rate(SM_E2, 0*ms, stim_on, N_E2)
    if abs(pre_a1 - pre_a2) > 30:
        predec_cnt += 1

    sim_folder = base_path / f"simulation_{run_idx+1}"
    create_directory(sim_folder)

    # Spike‑Rate Plot 
    fig1 = plt.figure(figsize=(8, 4))
    ax1 = fig1.add_subplot(111)
    pr_e1 = PR_E1.smooth_rate(window='flat', width=BIN_SIZE)/Hz
    pr_e2 = PR_E2.smooth_rate(window='flat', width=BIN_SIZE)/Hz
    ax1.plot(PR_E1.t/ms, pr_e1, label='E1')
    ax1.plot(PR_E2.t/ms, pr_e2, label='E2')
    ax1.set_xlabel('Zeit (ms)', fontsize=14)
    ax1.set_ylabel('Feuerrate (Hz)', fontsize=14)
    ax1.legend(fontsize=12); fig1.tight_layout()
    ax1.tick_params(labelsize=14)
    ax1.grid(True)
    fig1.savefig(sim_folder / 'spike_activity.png'); plt.close(fig1)

    # Spike Raster Plot
    fig2 = plt.figure(figsize=(8, 4))
    ax2 = fig2.add_subplot(111)
    ax2.scatter(SM_E1.t/ms, SM_E1.i, s=2, label='E1', color='steelblue')
    ax2.scatter(SM_E2.t/ms, SM_E2.i+N_E1, s=2, label='E2', color='orange')
    ax2.scatter(SM_I.t/ms, SM_I.i+N_E1+N_E2, s=2, label='I', color='green')
    ax2.set(xlabel='Zeit (ms)', ylabel='Neuron Index', xlim=(0, runtime/ms))
    ax2.legend(loc='upper right'); fig2.tight_layout()
    fig2.savefig(sim_folder / 'spike_raster_plot.png'); plt.close(fig2)

    # Stimulus plot
    fig3 = plt.figure(figsize=(8, 4))
    ax3 = fig3.add_subplot(111)
    ax3.plot(StM_E1.t/ms, StM_E1.rates[0]/Hz, color='steelblue', label='Input E1')
    ax3.plot(StM_E2.t/ms, StM_E2.rates[0]/Hz, color='orange', label='Input E2')
    ax3.set_xlabel('Zeit (ms)', fontsize=14)
    ax3.set_ylabel('Feuerrate (Hz)', fontsize=14)
    ax3.legend(fontsize=12); fig3.tight_layout()
    ax3.tick_params(labelsize=14)
    ax3.grid(True)
    fig3.savefig(sim_folder / 'stimulus.png'); plt.close(fig3)

    '''
    Hier wurden nur Diagramme für die Arbeit generiert

    fig, axs = plt.subplots(3, 1, figsize=(8, 7), sharex=True, layout='constrained', gridspec_kw={'height_ratios': [2, 2, 2]})
    axs[1].plot(StM_E1.t / ms, StM_E1.rates[0] / Hz, color='steelblue', label='Input E1')
    axs[1].plot(StM_E2.t / ms, StM_E2.rates[0] / Hz, color='orange', label='Input E2')
    axs[1].set_ylabel('Input (Hz)', fontsize=15)
    axs[1].set_title('B', loc='left', fontsize=15)
    axs[1].tick_params(labelsize=14)
    axs[1].legend(fontsize=12)
    axs[1].grid(True)

    axs[0].plot(PR_E1.t / ms, pr_e1, color='steelblue', label='E1')
    axs[0].plot(PR_E2.t / ms, pr_e2, color='orange', label='E2')
    axs[0].set_ylabel('Feuerrate (Hz)', fontsize=15)
    axs[0].set_title('A', loc='left', fontsize=15)
    axs[0].tick_params(labelsize=14)
    axs[0].legend(fontsize=12)
    axs[0].grid(True)

    axs[2].scatter(SM_E1.t/ms, SM_E1.i, s=2, label='E1', color='steelblue')
    axs[2].scatter(SM_E2.t/ms, SM_E2.i+N_E1, s=2, label='E2', color='orange')
    axs[2].scatter(SM_I.t/ms, SM_I.i+N_E1+N_E2, s=2, label='I', color='green')
    axs[2].set_ylabel('Neuron Index', fontsize=15)
    axs[2].set_title('C', loc='left', fontsize=15)
    axs[2].tick_params(labelsize=14)
    axs[2].legend(fontsize=12)
    axs[2].grid(True)

    axs[2].set_xlabel('Zeit (ms)', fontsize=14)

    fig.align_ylabels(axs)  

    #fig.savefig('final_result.png'); plt.close(fig)'''
metrik_path = base_path/ "metrik_report.txt"
with open(metrik_path, "w") as f:
    f.write('Genauigkeit: ' + str(true_pos / N_RUNS * 100) + '%\n')
    f.write('Eindeutigkeit: ' + str(round(np.mean(delta_abs_list),2)) + 'Hz\n')
    f.write('Effektive Inhibition: ' + str(round(np.mean(i_eff_list),2)) + 'Hz\n')
    f.write('Stabilität: ' + str(100-(comp_cnt / N_RUNS * 100)) + '%\n')
    f.write('Vorentscheidung: ' + str(predec_cnt / N_RUNS * 100) + '%\n')
    f.write('Durchschnittliche Aktivität der gewinnenden Population: ' + str(round(np.mean(win_act_list),2)) + 'Hz\n')
    f.write('Durchschnittliche Aktivität der verlierenden Population: ' + str(round(np.mean(lose_act_list),2)) + 'Hz\n')

