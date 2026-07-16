import numpy as np
from aptapy.plotting import plt

from thesis import FIGURES_DIR

name, y_min, y_max, sources = np.loadtxt("/home/augusto/Thesis/master_thesis/data/missions.txt", unpack=True, dtype=str, delimiter=",")
y_min = y_min.astype(float)
y_max = y_max.astype(float)
sources = sources.astype(float)
y_c = (y_min + y_max) / 2


fig_sources = plt.figure(figsize=(8, 6))

FONTSIZE = 12

plt.errorbar(y_c, sources, xerr=(y_max - y_min) / 2, color="k", fmt='none', elinewidth=2)
plt.scatter([1962], [1], color="k", s=20, edgecolors='black', zorder=3)
plt.annotate('Sco X-1', (1962, 1), xytext=(0, 8), textcoords='offset points', 
             fontsize=FONTSIZE, ha='center')
for i, txt in enumerate(name):
    plt.annotate(txt, (y_c[i], sources[i]), xytext=(-2, 2), 
                 textcoords='offset points', fontsize=FONTSIZE, ha='center')
plt.xlim(1955, 2026)
plt.xlabel('Year')
plt.ylabel('Number of X-ray sources detected')
plt.yscale('log')
plt.grid(True, which="both", ls=":", alpha=0.5)
plt.tight_layout()


fig_sources.savefig(FIGURES_DIR / "chapter2/xsources.pdf", dpi=300, bbox_inches="tight")




plt.show()
