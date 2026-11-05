"""Synthetic photometric bandpass and temperature-sensitivity diagnostics."""
import numpy as np
from scipy.constants import h, c, k

def planck_lambda(wavelength_m, temperature_k):
    lam=np.asarray(wavelength_m,dtype=float)
    if temperature_k<=0 or np.any(lam<=0) or not np.all(np.isfinite(lam)):
        raise ValueError('positive temperature and wavelengths required')
    x=h*c/(lam*k*temperature_k)
    return (2*h*c*c/lam**5)/np.expm1(x)

def band_mean(wavelength_m, throughput, temperature_k):
    lam=np.asarray(wavelength_m,dtype=float)
    tr=np.asarray(throughput,dtype=float)
    if lam.ndim!=1 or tr.shape!=lam.shape or len(lam)<3 or np.any(np.diff(lam)<=0) or np.any(tr<0) or not np.all(np.isfinite(tr)):
        raise ValueError('invalid bandpass')
    norm=np.trapezoid(tr,lam)
    if norm<=0: raise ValueError('zero throughput')
    return np.trapezoid(tr*planck_lambda(lam,temperature_k),lam)/norm

def log_color_sensitivity(wave_a, tr_a, wave_b, tr_b, temperature_k, fractional_step=1e-4):
    if fractional_step<=0 or fractional_step>=1: raise ValueError('invalid step')
    def color(temp):
        return np.log(band_mean(wave_a,tr_a,temp)/band_mean(wave_b,tr_b,temp))
    return (color(temperature_k*(1+fractional_step))-color(temperature_k*(1-fractional_step)))/(2*fractional_step)
