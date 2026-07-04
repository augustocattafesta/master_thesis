import numpy as np
from aptapy.plotting import plt
from thesis import FIGURES_DIR
import pandas as pd

l, t = np.loadtxt("/home/augusto/Thesis/master_thesis/data/em_spectrum/eso.txt", delimiter=" ", unpack=True)
l_x, t_x = np.loadtxt("/home/augusto/Thesis/master_thesis/data/em_spectrum/x.txt", delimiter=" ", unpack=True)
df = pd.read_csv('/home/augusto/Thesis/master_thesis/data/em_spectrum/radio.txt', sep=r'\s+', comment='#', header=None)

optical_e = 1240 / l[::200]
transmission = t[::200]
radio_e = df[0][::200] * 4.1357e-6
t_radio = df[1][::200]


plt.plot(l_x, t_x, linestyle="-", color="black", linewidth=1.5)
plt.plot(optical_e, transmission, linestyle="-", color="dimgray", linewidth=1.5)
plt.plot(radio_e, t_radio, linestyle="-", color="lightgray", linewidth=1.5)
plt.xlabel("Photon energy [eV]")
plt.ylabel("Transmission [%]")
plt.xscale("log")
plt.show()