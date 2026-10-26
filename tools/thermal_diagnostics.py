"""Numerically stable Planck-law and passband sensitivity diagnostics (SI units)."""
from __future__ import annotations

import math
import numpy as np

H = 6.62607015e-34
C = 299792458.0
K = 1.380649e-23
SIGMA_SB = 5.670374419e-8
WIEN_B = 2.897771955e-3


def planck_lambda(wavelength_m, temperature_k):
    """Spectral radiance B_lambda in W sr^-1 m^-3, stable at extreme x.

    The exp(-x)/(1-exp(-x)) form avoids exp(x) overflow; expm1
    avoids catastrophic cancellation in the Rayleigh-Jeans limit.
    """
    wl = np.asarray(wavelength_m, dtype=float)
    temp = np.asarray(temperature_k, dtype=float)
    if not np.all(np.isfinite(wl)) or not np.all(np.isfinite(temp)) or np.any(wl <= 0) or np.any(temp <= 0):
        raise ValueError("wavelength and temperature must be finite and positive")
    x = H*C/(wl*K*temp)
    inv = np.where(x < 40, 1.0/np.expm1(np.minimum(x, 40)), np.exp(-x)/(1-np.exp(-x)))
    return (2*H*C*C/wl**5)*inv


def bolometric_flux(temperature_k, *, samples=4097, x_min=1e-3, x_max=50.0):
    """Integrate pi*B_lambda d(lambda) on a dimensionless log-x grid.

    x = hc/(lambda*k*T); finite x limits exclude tails and are reported.
    """
    if not np.isfinite(temperature_k) or temperature_k <= 0 or samples < 16 or not (0 < x_min < x_max):
        raise ValueError("invalid temperature, sampling or integration bounds")
    x = np.geomspace(x_min, x_max, samples)
    wl = H*C/(x*K*temperature_k)
    integrand = np.pi*planck_lambda(wl, temperature_k)*wl
    return float(np.trapezoid(integrand, np.log(x)))


def top_hat_flux(temperature_k, low_nm, high_nm, *, samples=501):
    """Passband-integrated pi*B_lambda d(lambda), ideal unit-throughput top hat."""
    if not (0 < low_nm < high_nm) or samples < 8:
        raise ValueError("invalid passband")
    wavelength = np.linspace(low_nm*1e-9, high_nm*1e-9, samples)
    return float(np.trapezoid(np.pi*planck_lambda(wavelength, temperature_k), wavelength))


def colour_ratio(temperature_k, blue_nm=(400., 500.), red_nm=(650., 750.), *, samples=501):
    if not (0 < blue_nm[0] < blue_nm[1] <= red_nm[0] < red_nm[1]):
        raise ValueError("non-overlapping ordered blue and red bands required")
    return top_hat_flux(temperature_k, *blue_nm, samples=samples) / top_hat_flux(temperature_k, *red_nm, samples=samples)


def colour_temperature(ratio, *, bounds=(1500., 20000.), blue_nm=(400.,500.), red_nm=(650.,750.), samples=501):
    """Bisection inversion for ideal two-top-hat Planck colour, not stellar Teff."""
    if not np.isfinite(ratio) or ratio <= 0 or not (0 < bounds[0] < bounds[1]):
        raise ValueError("invalid colour ratio or temperature bounds")
    low, high = map(float, bounds)
    rlo = colour_ratio(low, blue_nm, red_nm, samples=samples)
    rhi = colour_ratio(high, blue_nm, red_nm, samples=samples)
    if not (rlo <= ratio <= rhi):
        raise ValueError("ratio outside the specified temperature bracket")
    for _ in range(65):
        mid = (low+high)/2
        if colour_ratio(mid, blue_nm, red_nm, samples=samples) < ratio:
            low = mid
        else:
            high = mid
    return (low+high)/2


def colour_temperature_uncertainty(blue_flux, red_flux, blue_sigma, red_sigma, *, correlation=0., bounds=(1500.,20000.)):
    """First-order propagation including correlated two-band flux errors."""
    vals = (blue_flux, red_flux, blue_sigma, red_sigma, correlation)
    if not all(np.isfinite(v) for v in vals) or min(blue_flux,red_flux) <= 0 or min(blue_sigma,red_sigma) < 0 or abs(correlation)>1:
        raise ValueError("invalid fluxes, uncertainties or correlation")
    ratio = blue_flux/red_flux
    temp = colour_temperature(ratio, bounds=bounds)
    variance = ratio**2*((blue_sigma/blue_flux)**2+(red_sigma/red_flux)**2-2*correlation*blue_sigma*red_sigma/(blue_flux*red_flux))
    dt = max(1e-3, temp*1e-4)
    derivative = (colour_ratio(temp+dt)-colour_ratio(temp-dt))/(2*dt)
    if derivative <= 0:
        raise ArithmeticError("colour ratio is locally insensitive to temperature")
    return {"temperature_k": float(temp), "sigma_temperature_k": float(math.sqrt(max(0.,variance))/derivative),
            "ratio": float(ratio), "sigma_ratio": float(math.sqrt(max(0.,variance))),
            "d_ratio_d_temperature": float(derivative)}
