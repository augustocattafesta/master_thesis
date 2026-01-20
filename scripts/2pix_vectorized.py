import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

PITCH = 1.0
N_ELECTRONS = 1800
TRANS_DIFF_SIGMA = 0.12 * PITCH
TRANS_DIFF_SIGMA_RECON = 0.12
NOISE_SIGMA = 20

def q2_prob(r):
    return 1 - norm.cdf(PITCH * 0.5, loc=r, scale=TRANS_DIFF_SIGMA)

def q2_pdf(q2, r, noise_threshold):
    mu = N_ELECTRONS * q2_prob(r)
    
    pdf_cont = norm.pdf(q2, loc=mu, scale=NOISE_SIGMA)
    
    res = np.where(q2 > noise_threshold, pdf_cont, 0.0)
    normalization = (1.0 - norm.cdf(noise_threshold, loc=mu, scale=NOISE_SIGMA))
    return res / normalization


def r_recon_pdf(r_recon, r, noise_threshold):
    z = (r_recon - 0.5 * PITCH) / (PITCH * TRANS_DIFF_SIGMA_RECON)
    q2 = N_ELECTRONS * norm.cdf(z)
    jacobian_r = N_ELECTRONS * norm.pdf(z) / (PITCH * TRANS_DIFF_SIGMA_RECON)
    pdf_q2_val = q2_pdf(q2, r, noise_threshold)
    
    res = np.where(q2 > noise_threshold, pdf_q2_val * jacobian_r, 0.0)
    return res


def dr_pdf(dr, r, noise_threshold):
    pdf_right = r_recon_pdf(r + dr, r, noise_threshold)
    pdf_left = r_recon_pdf(r - dr, r, noise_threshold)
    
    return np.where(dr >= 0, pdf_right + pdf_left, 0.0)


def dr_marginalized_pdf(dr_vals, noise_threshold):
    r_grid = np.linspace(0, 0.5 * PITCH, 200)
    
    dr_col = np.atleast_1d(dr_vals)[:, np.newaxis]
    r_row = r_grid[np.newaxis, :]

    pdf_matrix = dr_pdf(dr_col, r_row, noise_threshold)
    weights = r_grid
    integral = np.trapezoid(pdf_matrix * r_grid, r_grid, axis=1)
    
    return integral / (0.5 * PITCH)


R = 0.2

plt.figure("q2 PDF")
xx_q2 = np.linspace(0, N_ELECTRONS / 2, 1000)
yy_q2 = q2_pdf(xx_q2, r=R * PITCH, noise_threshold=60)
plt.plot(xx_q2, yy_q2)


plt.figure("r_recon PDF")
xx_r_recon = np.linspace(0, 0.5 * PITCH, 1000)
yy_r_recon = r_recon_pdf(xx_r_recon, r=R * PITCH, noise_threshold=0)
plt.plot(xx_r_recon, yy_r_recon)

plt.figure("dr PDF")
xx_dr = np.arange(0, 0.5 * PITCH, R * PITCH / 100)
yy_dr = dr_pdf(xx_dr, r=R * PITCH, noise_threshold=0)
plt.plot(xx_dr, yy_dr)

plt.figure("Marginalized dr PDF")
xx_dr_marginalized = np.linspace(0., 0.5 * PITCH, 200)
yy_dr_marginalized = dr_marginalized_pdf(xx_dr_marginalized, 60)
plt.plot(xx_dr_marginalized, yy_dr_marginalized)
area = np.trapezoid(yy_dr_marginalized, xx_dr_marginalized)
print(f"Area totale: {area}")


plt.show()