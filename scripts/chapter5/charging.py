import numpy as np
import yaml

from aptapy.plotting import plt
from aptapy.models import Constant, StretchedExponential
from thesis import FIGURES_DIR


data = yaml.safe_load(open('/home/augusto/Thesis/master_thesis/scripts/chapter5/analysis_run.yaml'))
main_gain = data['folders']['trend']['tasks']['gain']['main']

# Estrazione dei dati
times = np.array(main_gain['times'])
gains = np.array([g['val'] for g in main_gain['gain_vals']])
gain_errors = np.array([g['err'] for g in main_gain['gain_vals']])

mask = times < 3.30
charging = gains[mask] / gains[mask][0]
charging_errors = gain_errors[mask] / gains[mask][0]

model = StretchedExponential() + Constant()
model.fit(times[mask], charging, sigma=charging_errors)

discharging = gains[~mask] / gains[mask][0]
discharging_errors = gain_errors[~mask] / gains[mask][0]

# Calcolo dei residui normalizzati per i dati di charging (su cui è stato fatto il fit)
fit_curve = model(times[mask])
norm_residuals = (charging - fit_curve) / charging_errors

# Creazione della figura a due pannelli
fig, (ax_main, ax_res) = plt.subplots(
    2, 1, 
    sharex=True, 
    gridspec_kw={'height_ratios': [3, 1]}, 
    figsize=(8, 6)
)

# --- Grafico Principale ---
plt.sca(ax_main)
ax_main.errorbar(times[mask], charging, yerr=charging_errors, fmt='.k', label='Charging')
ax_main.errorbar(times[~mask], discharging, yerr=discharging_errors, fmt='vk', label='Discharging')
scale = model._components[0].scale.ufloat()
label = f'Charging model\n' + rf"$\tau$ = {scale} h"
model.plot(label=label, plot_components=False)
ax_main.set_ylabel('Norm. Gain @ 5.9 keV')
ax_main.legend()
ax_main.set_xlim(-0.2, max(times) * 1.1)

# --- Grafico Residui Normalizzati ---
ax_res.errorbar(times[mask], norm_residuals, yerr=np.ones_like(norm_residuals), fmt='.k')
ax_res.axhline(0, color='gray', linestyle='--', linewidth=1)
ax_res.set_xlabel('Time [h]')
ax_res.set_ylabel("Norm. residuals")
ax_res.set_xlim(-0.2, max(times) * 1.1)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'chapter5' / 'charging_fit.png', bbox_inches='tight')
plt.show()