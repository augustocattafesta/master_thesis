import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,  # Dimensione generale del testo
    "axes.labelsize": 12,  # Etichette assi (xlabel, ylabel)
    "xtick.labelsize": 12,  # Numeri asse X
    "ytick.labelsize": 12,  # Numeri asse Y
    "legend.fontsize": 12,  # Testo della legenda
})

def distribution(x, A, B):
    return A + B*np.cos(x - np.pi/2)**2

xx = np.linspace(-np.pi, np.pi, 100)
yy = distribution(xx, 300, 700)

pol = plt.figure(figsize=(6, 3))
plt.plot(np.rad2deg(xx), yy, "-k", label="Detector response to linearly polarized beam")
plt.xlabel(r"Angle $\varphi$ [deg]")
plt.ylabel(r"Entries / bin")
plt.xlim(-180, 180)
plt.ylim(0)
plt.legend(frameon=False)
plt.tight_layout()

output_path = "/home/augusto/Thesis/master_thesis/slides/polarization.png"
pol.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()