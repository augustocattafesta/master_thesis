import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

PITCH = 1.0
N_ELECTRONS = 1800
TRANS_DIFF_SIGMA = 0.1384 * PITCH
TRANS_DIFF_SIGMA_RECON = 0.1295
NOISE_SIGMA = 20


def pdf_area(xx, yy):
    """Compute the area under the PDF curve using the trapezoidal rule."""
    return np.trapezoid(yy, xx)


def q2_ratio(r):
    """Fraction of charge collected by the second pixel for a photon incident at position r.
    """
    return 1 - norm.cdf(0.5 * PITCH, loc=r, scale=TRANS_DIFF_SIGMA)


def q2_pdf(q2, r, q_th):
    """Probability density function of the second pixel charge.

    Note that we are studying only events that are above threshold, thus we are not including
    the effect of 1-pixel events on this PDF and on the final resolution.
    """
    mu = N_ELECTRONS * q2_ratio(r)
    pdf = norm.pdf(q2, loc=mu, scale=NOISE_SIGMA)
    condition = np.logical_and(q2 > q_th, q2 < N_ELECTRONS / 2)
    final_pdf = np.where(condition, pdf, 0.0)

    normalization = norm.cdf(N_ELECTRONS / 2, loc=mu, scale=NOISE_SIGMA) - norm.cdf(q_th, loc=mu, scale=NOISE_SIGMA)
    return final_pdf / normalization


def r_recon_pdf(r_recon, r, q_th):
    """Probability density function of the reconstructed position r_recon.
    """
    z = (r_recon - 0.5 * PITCH) / (PITCH * TRANS_DIFF_SIGMA_RECON)
    q2 = N_ELECTRONS * norm.cdf(z)
    jacobian_r = N_ELECTRONS * norm.pdf(z) / (PITCH * TRANS_DIFF_SIGMA_RECON)
    pdf_q2 = q2_pdf(q2, r, q_th)
    return pdf_q2 * jacobian_r


def dr_pdf(dr, r, q_th):
    """Probability density function of the position residual dr = r_recon - r.
    """
    pdf_right = r_recon_pdf(r + dr, r, q_th)
    pdf_left = r_recon_pdf(r - dr, r, q_th)
    return np.where(dr >= 0, pdf_right + pdf_left, 0.0)


def dr_marginalized_pdf(dr, q_th):
    """Marginalized PDF of the position residual dr over the incident position r.
    """
    r_grid = np.linspace(0, 0.5 * PITCH, 500)
    pdf_matrix = dr_pdf(dr[:, np.newaxis], r_grid[np.newaxis, :], q_th)
    integral = np.trapezoid(pdf_matrix * r_grid, r_grid, axis=1)
    return integral / (0.5 * 0.25 * PITCH**2)


# PLOTS
R = 0.0 * PITCH
Q_TH = 0

# Plot of q2 PDF
plt.figure("q2 PDF")
xx_q2 = np.linspace(0, 2*N_ELECTRONS, 10000)
yy_q2 = q2_pdf(xx_q2, r=R, q_th=Q_TH)
plt.plot(xx_q2, yy_q2)
plt.xlabel("q2 [electrons]")
plt.ylabel("PDF(q2)")
print("Area under q2 PDF:", pdf_area(xx_q2, yy_q2))


# Plot of r_recon PDF
plt.figure("r_recon PDF")
xx_recon = np.linspace(0, PITCH, 10000)
yy_recon = r_recon_pdf(xx_recon, r=R, q_th=Q_TH)
plt.plot(xx_recon, yy_recon)
plt.xlabel("r_recon [pitch units]")
plt.ylabel("PDF(r_recon)")
print("Area under r_recon PDF:", pdf_area(xx_recon, yy_recon))


# Plot of dr PDF
plt.figure("dr PDF")
xx_dr = np.linspace(0, PITCH, 10000)
yy_dr = dr_pdf(xx_dr, r=R, q_th=Q_TH)
plt.plot(xx_dr, yy_dr)
plt.xlabel("dr [pitch units]")
plt.ylabel("PDF(dr)")
print("Area under dr PDF:", pdf_area(xx_dr, yy_dr))


# Plot of marginalized dr PDF
plt.figure("Marginalized dr PDF")
xx_dr_marginalized = np.linspace(0, PITCH, 1000)
yy_dr_marginalized = dr_marginalized_pdf(xx_dr_marginalized, q_th=Q_TH)
plt.plot(xx_dr_marginalized, yy_dr_marginalized)
plt.xlabel("dr [pitch units]")
plt.ylabel("PDF_marginalized(dr)")
print("Area under marginalized dr PDF:", pdf_area(xx_dr_marginalized, yy_dr_marginalized))


# STUDYING THE RESOLUTION FOR DIFFERENT THRESHOLDS
mean = []
std = []
N = 100
nn = np.linspace(0, 10*NOISE_SIGMA, N)
for n_th in nn:
    yy_dr_marginalized = dr_marginalized_pdf(xx_dr_marginalized, q_th=n_th)
    mean.append( np.sum(xx_dr_marginalized * yy_dr_marginalized) / np.sum(yy_dr_marginalized) )
    std.append( np.sqrt( np.sum( (xx_dr_marginalized - mean[-1])**2 * yy_dr_marginalized ) / np.sum(yy_dr_marginalized) ) )
    # plt.plot(xx_dr_marginalized, yy_dr_marginalized, label=f'Th={n_th:.1f}')

plt.figure("Mean and Std vs Threshold")
plt.plot(nn, mean, label='Mean')
plt.plot(nn, std, label='Std')
plt.xlabel('Noise Threshold [electrons]')
plt.ylabel('Distance [pitch units]')
plt.legend()


plt.show()