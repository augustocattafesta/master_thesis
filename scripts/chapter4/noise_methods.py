import numpy as np
import matplotlib.pyplot as plt
from thesis import FIGURES_DIR

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

# Spaziatura orizzontale ridotta (wspace=0.2) per avvicinare i grafici
fig, axs = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True, 
                        gridspec_kw={'wspace': 0.2})

x = np.linspace(0, 10, 1000)
sigma = 1.5
mu1 = 4.0
mu2 = 1.2
mu3 = 0.0
amplitude = 0.45

def conceptual_gaussian(x, mu, sigma, amp):
    return amp * np.exp(-0.5 * ((x - mu) / sigma)**2)

# --- Method 1 ---
y2 = conceptual_gaussian(x, mu2, sigma, amplitude)
axs[2].plot(x, y2, color='black', linewidth=1.2)
axs[2].plot([0, 0], [0, conceptual_gaussian(0, mu2, sigma, amplitude)], color='black', linewidth=1.2)
axs[2].annotate("Method 3", xy=(0.1, 0.9), xycoords='axes fraction', fontsize=11)

# --- Method 2 ---
y1 = conceptual_gaussian(x, mu1, sigma, amplitude)
axs[1].plot(x, y1, color='black', linewidth=1.2)
# axs[1].text(mu1 + 0.5, amplitude * 0.4, 'Pedestal', fontsize=9, color='black')
axs[1].annotate("Method 2", xy=(0.1, 0.9), xycoords='axes fraction', fontsize=11)

# --- Method 3 ---
y3 = conceptual_gaussian(x, mu3, sigma, amplitude)
axs[0].plot(x, y3, color='black', linewidth=1.2)
axs[0].annotate("Method 1", xy=(0.1, 0.9), xycoords='axes fraction', fontsize=11)

# --- Formattazione estetica comune ---
for ax in axs:
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 0.6)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "chapter4/calibration/noise_methods.pdf", dpi=300, bbox_inches="tight")
plt.show()
