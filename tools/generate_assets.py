import numpy as np
import matplotlib.pyplot as plt

# Parametri
sigma = 1.0
mu_hn = sigma * np.sqrt(2 / np.pi)
sigma_hn = sigma * np.sqrt(1 - 2/np.pi)

# Dati
xx = np.linspace(-4, 4, 1000)
yy = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (xx/sigma)**2)

fig, ax = plt.subplots(figsize=(8, 5))

# Plot
ax.plot(xx, yy, "-k", lw=1.5, label="Gaussian")
ax.fill_between(xx[xx>=0], yy[xx>=0], color='gray', alpha=0.2, label="Positive half-Gaussian")

# --- FIX ASSI ---
ax.set_ylim(0, max(yy) * 1.1)  # Blocca l'asse Y partendo da zero
ax.set_xlim(-4, 4)

# Sposta assi
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Rimuovi tick Y
ax.set_yticks([])

# --- AGGIUNTA FRECCE (Stile LaTeX) ---
# Freccia asse X
ax.annotate('', xy=(4.2, 0), xytext=(-4.2, 0),
            arrowprops=dict(arrowstyle="->", color='black', lw=1), clip_on=False)
# Freccia asse Y
ax.annotate('', xy=(0, max(yy)*1.15), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color='black', lw=1), clip_on=False)

# --- ANNOTAZIONI ---
# Nota: usiamo clip_on=False per assicurarci che il testo sotto lo zero sia visibile
ax.axvline(mu_hn, ymin=0, ymax=0.75, color='black', linestyle='--', lw=1)
ax.text(mu_hn, -0.005, r'$\mu_{pos}$', color='black', ha='center', va='top', fontsize=12)

# Sigma_HN disegnata come intervallo
y_pos_sigma = 0.1
ax.hlines(y_pos_sigma, mu_hn - sigma_hn, mu_hn + sigma_hn, colors='black', lw=1)
ax.vlines([mu_hn - sigma_hn, mu_hn + sigma_hn], y_pos_sigma-0.01, y_pos_sigma+0.01, colors='black', lw=1)
ax.text(mu_hn - 0.2, y_pos_sigma + 0.02, r'$\sigma_{pos}$', color='black', ha='center', fontsize=12)

plt.legend(frameon=False, loc='upper right')
plt.tight_layout()
plt.savefig("/home/augusto/Thesis/thesis_doc/figures/noise/half_gaussian.pdf", format="pdf")