'''
Dieses Programm wurde dazu genutzt, um Abbildung 2.4 zu generieren.
'''
from brian2 import *
import matplotlib.pyplot as plt


duration = 1*second
defaultclock.dt = 0.1*ms

# Zeitabhängige Rate: 10 Hz bis 500 ms, dann 40 Hz
rate_timedep = TimedArray([20, 60]*Hz, dt=500*ms)

# Poisson-Gruppe mit zeitabhängiger Rate
poisson_group = PoissonGroup(1, rates='rate_timedep(t)')

# Monitore
spike_mon = SpikeMonitor(poisson_group)

dummy_group = NeuronGroup(1, model='rate_val = rate_timedep(t) : Hz', method='euler')
state_mon = StateMonitor(dummy_group, 'rate_val', record=True)


run(duration)

# Plotten
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 5 ), layout='constrained')

# Spike Train Plot 
for t in spike_mon.t/ms:
    ax1.axvline(t, ymin=0.3, ymax=0.7, color='k', linewidth=0.8)
ax1.set_xlim(0, duration/ms)
ax1.set_ylim(0, 1)
ax1.set_yticks([0.5])
ax1.set_yticklabels(['1'], fontsize=14)
ax1.set_ylabel('Neuron', fontsize=16)
ax1.grid(True)

# Eingaberate 
ax2.plot(state_mon.t/ms, state_mon.rate_val[0], color='b', label='Poisson Input')
ax2.set_xlabel('Zeit (ms)', fontsize=16)
ax2.set_ylabel('Rate (Hz)', fontsize=16)
ax2.set_ylim(bottom=0)
ax2.legend(fontsize=14)
ax2.grid(True)


ax1.tick_params(labelsize=14)
ax2.tick_params(labelsize=14)

fig.align_ylabels((ax1, ax2))
fig.savefig('poisson_example.png')
