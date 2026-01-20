import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

PITCH = 1.0
N_ELECTRONS = 1800
TRANS_DIFF_SIGMA = 0.1 * PITCH
TRANS_DIFF_SIGMA_RECON = 0.1
NOISE_SIGMA = 20

def q2_prob(r):
    return 1 - norm.cdf(PITCH * 0.5, loc=r, scale=TRANS_DIFF_SIGMA)

def q2_pdf(q2, r, noise_threshold):
    mu = N_ELECTRONS * q2_prob(r)
    p_below_th = norm.cdf(noise_threshold, loc=mu, scale=NOISE_SIGMA)
    
    pdf_cont = norm.pdf(q2, loc=mu, scale=NOISE_SIGMA)
    
    res = np.where(q2 > noise_threshold, pdf_cont, 0.0)
    res = np.where(np.isclose(q2, 0), p_below_th, res)
    return res


def r_recon_pdf(r_recon, r, noise_threshold):
    """PDF della posizione ricostruita r_recon."""
    z = (r_recon - 0.5 * PITCH) / (PITCH * TRANS_DIFF_SIGMA_RECON)
    q2 = N_ELECTRONS * norm.cdf(z)
    jacobian_r = N_ELECTRONS * norm.pdf(z) / (PITCH * TRANS_DIFF_SIGMA_RECON)

    pdf_q2_val = q2_pdf(q2, r, noise_threshold)
    p_below_th = q2_pdf(0, r, noise_threshold)
    
    res = np.where(q2 > noise_threshold, pdf_q2_val * jacobian_r, 0.0)
    res = np.where(np.isclose(r_recon, 0), p_below_th, res)
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

    integral = np.trapezoid(pdf_matrix, r_grid, axis=1)
    
    return integral / (0.5 * PITCH)

def r_recon_pdf_continuous(r_recon, r, noise_threshold):
    """Calcola solo la parte densità, ignorando la massa in zero."""
    z = (r_recon - 0.5 * PITCH) / (PITCH * TRANS_DIFF_SIGMA_RECON)
    q2 = N_ELECTRONS * norm.cdf(z)
    jacobian_r = N_ELECTRONS * norm.pdf(z) / (PITCH * TRANS_DIFF_SIGMA_RECON)
    
    mu_q2 = N_ELECTRONS * q2_prob(r)
    pdf_q2_val = norm.pdf(q2, loc=mu_q2, scale=NOISE_SIGMA)
    
    # Restituisce la densità solo sopra soglia, 0 altrove
    return np.where(q2 > noise_threshold, pdf_q2_val * jacobian_r, 0.0)

def dr_marginalized_pdf(dr_vals, noise_threshold):
    r_max = 0.5 * PITCH
    r_grid = np.linspace(0, r_max, 500)
    dr_vals = np.atleast_1d(dr_vals)
    
    # 1. Preparazione matrici per broadcasting
    dr_col = dr_vals[:, np.newaxis]
    r_row = r_grid[np.newaxis, :]
    
    # 2. Parte Continua
    pdf_matrix = r_recon_pdf_continuous(r_row + dr_col, r_row, noise_threshold) + \
                 r_recon_pdf_continuous(r_row - dr_col, r_row, noise_threshold)
    
    # Applichiamo il peso lineare r
    weights = r_grid 
    integral_cont = np.trapezoid(pdf_matrix * weights, r_grid, axis=1)
    
    # 3. Parte Discreta (La gobbetta)
    # Anche qui, ogni 'p_fail' deve essere pesata per il suo r (che è proprio dr_vals)
    p_fail_at_dr = q2_pdf(0, dr_vals, noise_threshold)
    integral_disc = np.where(dr_vals <= r_max, p_fail_at_dr * dr_vals, 0.0)
    
    # 4. Fattore di Normalizzazione
    # L'integrale di r dr da 0 a r_max è (r_max^2) / 2
    norm_factor = (r_max**2) / 2
    
    return (integral_cont + integral_disc) / norm_factor


R = 0.2

plt.figure("q2 PDF")
xx_q2 = np.linspace(0, N_ELECTRONS / 2, 1000)
yy_q2 = q2_pdf(xx_q2, r=R * PITCH, noise_threshold=60)
plt.plot(xx_q2, yy_q2)

# plt.figure("eta PDF")
# xx_eta = np.linspace(0, 0.5, 1000)
# yy_eta = eta_pdf(xx_eta, r=R * PITCH, noise_threshold=60)
# plt.plot(xx_eta, yy_eta)

plt.figure("r_recon PDF")
xx_r_recon = np.linspace(0, 0.5 * PITCH, 1000)
yy_r_recon = r_recon_pdf(xx_r_recon, r=R * PITCH, noise_threshold=60)
plt.plot(xx_r_recon, yy_r_recon)

plt.figure("dr PDF")
xx_dr = np.arange(0, 0.5 * PITCH, R * PITCH / 100)
yy_dr = dr_pdf(xx_dr, r=R * PITCH, noise_threshold=60)
plt.plot(xx_dr, yy_dr)

plt.figure("Marginalized dr PDF")
xx_dr_marginalized = np.linspace(0, 0.5 * PITCH, 200)
yy_dr_marginalized = dr_marginalized_pdf(xx_dr_marginalized, 60)
plt.plot(xx_dr_marginalized, yy_dr_marginalized)
area = np.trapezoid(yy_dr_marginalized, xx_dr_marginalized)
print(f"Area totale: {area}")

mean = []
std = []
N = 200
nn = np.linspace(0, 30*NOISE_SIGMA, N)
# plt.figure("Marginalized dr PDF for different thresholds")
for n_th in nn:
    yy_dr_marginalized = dr_marginalized_pdf(xx_dr_marginalized, n_th)
    mean.append( np.sum(xx_dr_marginalized * yy_dr_marginalized) / np.sum(yy_dr_marginalized) )
    std.append( np.sqrt( np.sum( (xx_dr_marginalized - mean[-1])**2 * yy_dr_marginalized ) / np.sum(yy_dr_marginalized) ) )
    # plt.plot(xx_dr_marginalized, yy_dr_marginalized, label=f'Th={n_th}')

plt.figure("Mean and Std vs Threshold")
plt.plot(nn, mean, label='Mean')
plt.plot(nn, std, label='Std')
plt.xlabel('Noise Threshold')
plt.ylabel('Distance')
plt.legend()

plt.show()