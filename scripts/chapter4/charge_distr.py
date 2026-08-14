import numpy as np
from aptapy.plotting import plt
from thesis import FIGURES_DIR




data = np.array([[19.6 , 74.0 , 17.5 , 7.3, 1.1]
,[19.4 , 71.2 , 19.9 , 7.9, 1.0]
,[19.2 , 71.0 , 20.1 , 7.9, 1.0]
,[18.7 , 71.5 , 19.9 , 7.8, 0.9]
,[18.1 , 72.5 , 19.4 , 7.3, 0.7]
,[17.4 , 73.8 , 18.8 , 6.8, 0.6]
,[16.6 , 74.9 , 18.2 , 6.3, 0.5]
,[15.9 , 75.8 , 17.8 , 5.9, 0.5]
,[15.3 , 76.7 , 17.2 , 5.6, 0.4]
,[14.9 , 77.3 , 16.9 , 5.4, 0.4]
,[14.5 , 77.9 , 16.5 , 5.2, 0.4]
,[14.3 , 78.3 , 16.3 , 5.1, 0.3]])

x = data[:, 0]
one = data[:, 1]
two = data[:, 2]
three = data[:, 3]
others = data[:, 4]

plt.plot(x, one, "-k", label=r"$1^{\text{st}}$ pixel")
plt.plot(x, two, "--k", label=r"$2^{\text{nd}}$ pixel")
plt.plot(x, three, "-.k", label=r"$3^{\text{rd}}$ pixel")
plt.plot(x, others, ":k", label=r"$4^{\text{th}}-7^{\text{th}}$ pixels")
plt.xlabel(r"$\langle \sigma \rangle / p$ [%]")
plt.ylabel("Fraction of events [%]")
plt.xlim(min(x), max(x))
plt.ylim(-5, 100)
plt.legend(frameon=False)
plt.tight_layout()

plt.savefig(FIGURES_DIR / "chapter4/design/charge_distr.png", bbox_inches="tight")

plt.show()