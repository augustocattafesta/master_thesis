import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

SIGMA_TRANS = 0.12
SIGMA_RECON = 0.12
SIGMA_NOISE = 20
N = 1800
Q_TH = 60

# PDFs for 2 pixel events with zero suppression
def q_pdf(q, r):
    mu = N * (1 - norm.cdf((0.5 - r) / SIGMA_TRANS))
    return 1 / np.sqrt(2 * np.pi * SIGMA_NOISE**2) * np.exp( - (q - mu)**2 / (2 * SIGMA_NOISE**2) )


def eta_pdf(eta, r):
    mu = 1 - norm.cdf((0.5 - r) / SIGMA_TRANS)
    return N / np.sqrt(2 * np.pi * SIGMA_NOISE**2) * np.exp( - 1 / (2 * (SIGMA_NOISE / N)**2) * (eta - mu)**2 )


def recon_pdf(r_rc, r):
    eta = norm.cdf((r_rc - 0.5) / SIGMA_RECON)
    return 1 / np.sqrt(2 * np.pi * SIGMA_RECON**2) * np.exp( - (r_rc - 0.5)**2 / (2 * SIGMA_RECON**2) ) * eta_pdf(eta, r)


def res_pdf(res, r):
    # r_rc = r - res
    return recon_pdf(r + res, r) + recon_pdf(r - res, r)


def efficiency(r):
    mu_q = N * (1 - norm.cdf((0.5 - r) / SIGMA_TRANS))
    return 1 - norm.cdf(Q_TH, loc=mu_q, scale=SIGMA_NOISE)
    

def marginalized(res):
    steps = 200
    r_grid = np.linspace(0., 0.5, steps)
    dr = r_grid[1] - r_grid[0]
    eff = efficiency(r_grid)
    probabilities = res_pdf(res, r_grid) * r_grid * eff
    norm_factor = np.sum(eff * r_grid * dr)
    marginal_prob = np.sum(probabilities * dr) / norm_factor
    return marginal_prob


def marginalized_complete(res):
    steps = 200
    r_grid = np.linspace(1e-5, 0.5, steps)
    dr = r_grid[1] - r_grid[0]
    
    eff = efficiency(r_grid)
    
    # 1. Contributo eventi a 2 pixel (PDF continua)
    p_2pixel = res_pdf(res, r_grid) * eff
    
    # 2. Contributo eventi a 1 pixel (Picco nel punto res = r)
    # Usiamo una gaussiana molto stretta per simulare la Delta di Dirac
    p_1pixel = (1 - eff) * norm.pdf(res, loc=r_grid, scale=0.001)
    
    # Integrazione pesata per la geometria r
    probabilities = (p_2pixel + p_1pixel) * r_grid
    
    # Normalizzazione sull'intera geometria
    return np.sum(probabilities * dr) / np.sum(r_grid * dr)

def sample_from_pdf(res_axis, p_res, size=1000):
    cdf = np.cumsum(p_res)
    cdf /= cdf[-1]
    u = np.random.uniform(0, 1, size)
    return np.interp(u, cdf, res_axis)





e = np.linspace(-0.5, 0.5, 1000)
p_res_marginal = np.array([marginalized_complete(v) for v in e])
plt.figure("Resolution PDF")
plt.plot(e, p_res_marginal)

plt.figure("recon PDF at r=0.2")
r_test = 0.2
xx_recon = np.linspace(0, 0.5, 1000)
yy_recon = recon_pdf(xx_recon, r=r_test)
plt.plot(xx_recon, yy_recon)


res_samples = sample_from_pdf(e, p_res_marginal, size=30000)
plt.figure("Resolution Samples")
plt.hist(res_samples, bins=100)

def absolute_resolution_pdf(res_axis, p_res):
    pos_mask = res_axis >= 0
    y_axis = res_axis[pos_mask]    
    p_abs = []
    for val in y_axis:
        p_positive = p_res[np.argmin(np.abs(res_axis - val))]
        p_negative = p_res[np.argmin(np.abs(res_axis + val))]
        p_abs.append(p_positive + p_negative)
        
    return y_axis, np.array(p_abs)

abs_res_samples = np.abs(res_samples)

plt.figure("Absolute Error Distribution")
plt.hist(abs_res_samples, bins=50, density=True, alpha=0.7, label="Absolute Error")
plt.xlabel("|Position Error| [mm]")
plt.ylabel("Probability Density")

# Calcolo del valore atteso (errore medio)
mean_error = np.mean(abs_res_samples)
plt.axvline(mean_error, color='r', linestyle='--', label=f'Mean: {mean_error:.3f}')
plt.legend()
plt.show()










plt.show()


