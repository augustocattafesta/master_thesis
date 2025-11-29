import numpy as np
import uncertainties
from aptapy.hist import Histogram1d
from aptapy.models import Fe55Forest
from aptapy.plotting import plt
from uncertainties import unumpy

E_SCALE = 0.2
SIGMA = 3.
AMP_R = 5.

def test_fit_forest(size):

    amp0 = []
    amp1 = []
    sigma = []
    e_scale = []
    for i in range(size):
        model = Fe55Forest()
        model.amplitude1.init(model.amplitude0.value / AMP_R)
        model.sigma.init(SIGMA)
        model.energy_scale.init(E_SCALE)
        hist = Histogram1d(xedges=np.arange(15.5, 45.5), xlabel='ADC Channel')
        hist.fill(model.rvs(size=10000))
        model.fit(hist)
        amp0.append(model.amplitude0.ufloat())
        amp1.append(model.amplitude1.ufloat())
        sigma.append(model.sigma.ufloat())
        e_scale.append(model.energy_scale.ufloat())
    
    amp0 = np.array(amp0)
    amp1 = np.array(amp1)
    
    plt.figure(f"Iteration {size}")
    hist.plot()
    model.plot(fit_output=True)
    plt.legend()

    plt.figure('Amp 1 relative error')
    amp1_err = abs(unumpy.std_devs(amp1) / unumpy.nominal_values(amp1))
    amp1_hist = Histogram1d(xedges=np.linspace(0, 4, 10), xlabel=r'|$\sigma_{a1}$ / a1|')
    amp1_hist.fill(amp1_err)
    amp1_hist.plot()

    plt.figure('Amp ratio relative error')
    amp_ratio = amp1 / amp0
    amp_ratio_err = abs(unumpy.std_devs(amp_ratio) / unumpy.nominal_values(amp_ratio))
    amp_ratio_hist = Histogram1d(xedges=np.linspace(0, 4, 10), xlabel=r'|$\sigma_{a0/a1}$ / (a0/a1)|')
    amp_ratio_hist.fill(amp_ratio_err)
    amp_ratio_hist.plot()

    plt.figure("Amp ratio")
    amp_rn = (unumpy.nominal_values(amp_ratio) - 1/AMP_R) / unumpy.std_devs(amp_ratio)
    amp_rn_hist = Histogram1d(xedges=np.linspace(min(amp_rn), max(amp_rn), 100), xlabel='Amp ratio')
    amp_rn_hist.fill(amp_rn)
    amp_rn_hist.plot()

    plt.figure("Sigma")
    sigma  = np.array(sigma)
    s = (unumpy.nominal_values(sigma) - SIGMA) / unumpy.std_devs(sigma)
    s_hist = Histogram1d(xedges=np.linspace(min(s), max(s), 100))
    s_hist.fill(s)
    s_hist.plot()



def test_model():
    model = Fe55Forest()
    corr_pars = [uncertainties.correlated_values([par for par in _model], _model.pcov) for _model in models]
test_fit_forest(1000)

# test_model()
plt.show()

