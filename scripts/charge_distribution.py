import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

PITCH = 1.0
N_ELECTRONS = 1800
TRANS_DIFF_SIGMA = 0.1 * PITCH
TRANS_DIFF_SIGMA_RECON = 0.1
NOISE_SIGMA = 20
NOISE_THRESHOLD = 30


def q2_prob(r):
    """Mean charge collected by the second pixel for a photon incident at position r.
    """
    return 1 - norm.cdf(PITCH * 0.5, loc=r, scale=TRANS_DIFF_SIGMA)


def q2_pdf(q2, r):
    """Probability density function of the second pixel charge.
    
    Arguments
    ---------
    q2 : float
        Charge collected by the second pixel.
    r : float
        Photon incident position relative to the first pixel center.
    """
    mu = N_ELECTRONS * q2_prob(r)
    p_below_th = norm.cdf(NOISE_THRESHOLD, loc=mu, scale=NOISE_SIGMA)
    pdf_over_th = norm.pdf(q2, loc=mu, scale=NOISE_SIGMA)
    pdf_values = np.where(q2 > NOISE_THRESHOLD, pdf_over_th, 0)
    return np.where(np.isclose(q2, 0), p_below_th, pdf_values)


def eta_pdf(eta, r):
    """Probability density function of the eta variable.
    
    Arguments
    ---------
    eta : float
        Charge ratio of the second pixel eta = q2 / Q_TOT.
    r : float
        Photon incident position relative to the first pixel center.
    """
    eta_threshold = NOISE_THRESHOLD / N_ELECTRONS
    q2 = eta * N_ELECTRONS
    jacobian = N_ELECTRONS

    original_pdf = q2_pdf(q2, r)
    mask = eta > eta_threshold
    new_pdf = np.zeros_like(original_pdf)
    new_pdf[mask] = original_pdf[mask] * jacobian
    mask_zero = np.isclose(eta, 0)
    new_pdf[mask_zero] = original_pdf[mask_zero]
    
    return new_pdf


def r_recon_pdf(r_recon, r):
    """Probability density function of the reconstructed position.
    
    Arguments
    ---------
    r_recon : float
        Reconstructed photon incident position relative to the first pixel center.
    r : float
        True photon incident position relative to the first pixel center.
    """
    z = (r_recon - 0.5 * PITCH) / (PITCH * TRANS_DIFF_SIGMA_RECON)
    eta = norm.cdf(z)
    jacobian = norm.pdf(z) / (PITCH * TRANS_DIFF_SIGMA_RECON)

    pdf_eta = eta_pdf(eta, r)
    eta_threshold = NOISE_THRESHOLD / N_ELECTRONS

    new_pdf = np.zeros_like(pdf_eta)
    mask = eta > eta_threshold
    new_pdf[mask] = jacobian[mask] * pdf_eta[mask]

    p_below_th = eta_pdf(0, r)
    mask_zero = np.isclose(r_recon, 0)
    new_pdf[mask_zero] = p_below_th

    return new_pdf


def dr_pdf(dr, r):
    """Probability density function of the distance between true and reconstructed position.
    
    Arguments
    ---------
    dr : float
        Distance between true and reconstructed position.
    r : float
        True photon incident position relative to the first pixel center.
    """
    pdf_right = r_recon_pdf(r + dr, r)
    pdf_left = r_recon_pdf(r - dr, r)
    
    return np.where(dr >= 0, pdf_right + pdf_left, 0.0)


def dr_marginalized_pdf(dr_vals):
    r_grid = np.linspace(0, 0.5 * PITCH, 200)
    
    dr_col = np.atleast_1d(dr_vals)[:, np.newaxis]
    r_row = r_grid[np.newaxis, :]

    pdf_matrix = dr_pdf(dr_col, r_row)

    integral = np.trapezoid(pdf_matrix, r_grid, axis=1)
    
    return integral / (0.5 * PITCH)

R = 0.1

plt.figure("q2 PDF")
xx_q2 = np.linspace(0, N_ELECTRONS / 2, 1000)
yy_q2 = q2_pdf(xx_q2, r=R * PITCH)
plt.plot(xx_q2, yy_q2)

plt.figure("eta PDF")
xx_eta = np.linspace(0, 0.5, 1000)
yy_eta = eta_pdf(xx_eta, r=R * PITCH)
plt.plot(xx_eta, yy_eta)

plt.figure("r_recon PDF")
xx_r_recon = np.linspace(0, 0.5 * PITCH, 1000)
yy_r_recon = r_recon_pdf(xx_r_recon, r=R * PITCH)
plt.plot(xx_r_recon, yy_r_recon)

plt.figure("dr PDF")
xx_dr = np.arange(0, 0.5 * PITCH, R * PITCH / 100)
yy_dr = dr_pdf(xx_dr, r=R * PITCH)
plt.plot(xx_dr, yy_dr)

plt.figure("Marginalized dr PDF")
xx_dr_marginalized = np.linspace(0, 0.5 * PITCH, 200)
yy_dr_marginalized = dr_marginalized_pdf(xx_dr_marginalized)
plt.plot(xx_dr_marginalized, yy_dr_marginalized)

plt.show()