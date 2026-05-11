import numpy as np
import pandas as pd
from scipy import integrate

diameter = pd.Series(
    [0.12,0.088,0.062,0.044,0.03125],
    index=[0.602, 0.311, 0.165, 0.082, 0.042])
# diameter = np.arange(-2, 9)


def tgsd_func(log10eta, h_p):
    # tgsd provided by Costa et al. (2016)
    [sgm2, a_sgm1, b_sgm1, a_m13sgm1, b_m13sgm1, c_mu2mu1, b_mu2mu1, a_viscp1, b_viscp1] = [1.46, 0.67, 0.07, 0.96, 0.2,
                                                                                            1.62, 0.66, 1.61, 0.31]
    p1 = a_viscp1 * np.exp(-b_viscp1 * log10eta)
    p2 = 1 - p1
    # d = np.arange(-4, 6.5, 0.01)
    sgm1 = a_sgm1 + b_sgm1 * h_p
    mu1 = a_m13sgm1 + b_m13sgm1 * h_p - 3 * sgm1
    mu2 = c_mu2mu1 * log10eta ** b_mu2mu1 + mu1

    def f_tgsd(d):
        return p1 / (np.sqrt(2 * np.pi) * sgm1) * np.exp(-0.5 * (d - mu1) ** 2 / sgm1 ** 2) + \
               p2 / (np.sqrt(2 * np.pi) * sgm2) * np.exp(-0.5 * (d - mu2) ** 2 / sgm2 ** 2)

    # tgsd_phi = np.array([integrate.quad(tgsd_func, i - 0.5, i + 0.5)[0] for i in np.arange(-4, 6)])
    tgsd_array_psv = pd.Series(
        np.array([integrate.quad(f_tgsd, diameter.iloc[i], diameter.iloc[i + 1])[0] for i in
                  range(len(diameter) - 1)]), index=diameter.iloc[:-1])
    return tgsd_array_psv
