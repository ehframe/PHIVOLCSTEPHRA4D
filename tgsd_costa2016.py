import numpy as np
import pandas as pd
from scipy import integrate

diameter = pd.Series(
    [5.22, 4.42, 4.04, 3.78, 3.58, 3.41, 3.26, 3.13, 3, 2.89, 2.73, 2.53, 2.35, 2.18, 2.01, 1.77, 1.47, 1.19, 0.93,
     0.69, 0.33, -0.06, -0.41, -0.73, -1.01, -1.4, -1.82, -2.19, -2.51, -2.8, -3.17, -3.6],
    index=[0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.1, 1.3, 1.5, 1.7, 1.9,
           2.2, 2.6, 3, 3.4, 3.8, 4.4, 5.2, 6, 6.8, 7.6, 8.8, 10.4, 12, 13.6, 15.2, 17.6, 20.8])
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
