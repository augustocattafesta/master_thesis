import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import Delaunay

# --- 1. SETUP GRAFICO PER TESI ---
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "mathtext.fontset": "cm",
        "figure.dpi": 300,
    }
)

# --- 2. GEOMETRIA CLUSTER ESAGONALE ---
pitch = 1.0
r_in = pitch / 2.0  # Apotema
R_circ = r_in / np.cos(np.radians(30))  # Raggio circoscritto

# Centri dei 7 pixel
centers = {
    0: (0.0, 0.0),
    1: (0.5 * pitch, np.sqrt(3) / 2 * pitch),
    2: (1.0 * pitch, 0.0),
    3: (0.5 * pitch, -np.sqrt(3) / 2 * pitch),
    4: (-0.5 * pitch, -np.sqrt(3) / 2 * pitch),
    5: (-1.0 * pitch, 0.0),
    6: (-0.5 * pitch, np.sqrt(3) / 2 * pitch),
}


def get_hexagon_vertices(cx, cy, R):
    angles = np.deg2rad(np.arange(30, 390, 60))
    return np.column_stack((cx + R * np.cos(angles), cy + R * np.sin(angles)))


# --- 3. MODELLO DI DIFFUSIONE CARICA ---
x0, y0 = 0.3 * pitch, 0.2 * pitch
sigma = 0.15 * pitch

N_grid = 1000
x_span = np.linspace(-1.6 * pitch, 1.6 * pitch, N_grid)
y_span = np.linspace(-1.6 * pitch, 1.6 * pitch, N_grid)
X, Y = np.meshgrid(x_span, y_span)

# Gaussiana 2D
dx = x_span[1] - x_span[0]
dy = y_span[1] - y_span[0]
Z = (1.0 / (2.0 * np.pi * sigma**2)) * np.exp(
    -((X - x0) ** 2 + (Y - y0) ** 2) / (2.0 * sigma**2)
)

# Calcolo frazione di carica integrata per pixel
charge_fractions = {}
for idx, (cx, cy) in centers.items():
    poly = get_hexagon_vertices(cx, cy, R_circ)
    delaunay = Delaunay(poly)
    points = np.column_stack((X.ravel(), Y.ravel()))
    mask = (delaunay.find_simplex(points) >= 0).reshape(X.shape)
    charge_fractions[idx] = np.sum(Z[mask]) * dx * dy

# --- 4. PLOT ---
fig, ax = plt.subplots(figsize=(6, 6))

# Gradiente continuo con maschera morbida
dist_from_hit = np.hypot(X - x0, Y - y0)
Z_vis = np.exp(-0.5 * (dist_from_hit / sigma) ** 2)
Z_vis[dist_from_hit > 4.0 * sigma] = 0.0

# Colormap morbida (da trasparente/bianco a blu/viola intenso)
cmap = plt.cm.Blues

# Disegna la nuvola di carica come gradiente continuo
ax.imshow(
    Z_vis,
    extent=[x_span.min(), x_span.max(), y_span.min(), y_span.max()],
    origin="lower",
    cmap="YlGnBu",
    alpha=0.85,
    interpolation="bicubic",
    zorder=2,
)

# Disegna i pixel esagonali e le etichette di carica
for idx, (cx, cy) in centers.items():
    verts = get_hexagon_vertices(cx, cy, R_circ)

    # Bordo dell'esagono
    poly = patches.Polygon(
        verts,
        closed=True,
        facecolor="none",
        edgecolor="#222222",
        linewidth=1.5,
        zorder=4,
    )
    ax.add_patch(poly)

    # Testo con nome pixel e percentuale raccolta
    q_val = charge_fractions[idx] * 100.0
    text_str = (
        f"$q_{{{idx}}}$\n{q_val:.1f}%"
        if q_val >= 0.1
        else f"$q_{{{idx}}}$\n$<0.1\%$ "
    )

    ax.text(
        cx,
        cy,
        text_str,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="medium",
        color="#111111",
        zorder=6,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.7),
    )

# Punto di impatto reale (x0, y0)
ax.plot(
    x0,
    y0,
    marker="o",
    markersize=5,
    markerfacecolor="#d62728",
    markeredgecolor="white",
    markeredgewidth=1.0,
    zorder=7,
)
ax.text(
    x0 + 0.05 * pitch,
    y0 + 0.04 * pitch,
    r"$(x_0, y_0)$",
    fontsize=11,
    color="#b2182b",
    fontweight="bold",
    zorder=7,
)

# Pulizia assi
ax.set_xlim(-1.5 * pitch, 1.5 * pitch)
ax.set_ylim(-1.5 * pitch, 1.5 * pitch)
ax.set_aspect("equal")
ax.axis("off")  # Rimuove assi e bordi per un look pulito e schematico

plt.tight_layout()
plt.savefig("charge_diffusion_gradient.pdf", bbox_inches="tight")
plt.savefig("charge_diffusion_gradient.png", dpi=300, bbox_inches="tight")
plt.show()