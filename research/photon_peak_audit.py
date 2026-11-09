"""Independent blackbody photon-counting and peak-convention diagnostics.

Educational ideal blackbody; no instrument response or extinction included.
"""
import math

H = 6.62607015e-34
C = 299792458.0
K = 1.380649e-23


def spectral_radiance(wavelength_m: float, temperature_k: float, photons: bool = False) -> float:
    """Per-metre, per-steradian spectral radiance; photons toggles photon counts."""
    if not (math.isfinite(wavelength_m) and wavelength_m > 0
            and math.isfinite(temperature_k) and temperature_k > 0):
        raise ValueError("wavelength and temperature must be positive and finite")
    x = H * C / (wavelength_m * K * temperature_k)
    if x > 745:
        return 0.0
    occupation = 1.0 / math.expm1(x)
    energy = 2.0 * H * C**2 / wavelength_m**5 * occupation
    return energy * wavelength_m / (H * C) if photons else energy


def peak_root(power: int, iterations: int = 90) -> float:
    """Solve power*(1-exp(-x))=x for the nonzero wavelength peak."""
    if power not in (4, 5):
        raise ValueError("power must be 4 (photons) or 5 (energy)")
    low, high = 1.0, float(power)
    for _ in range(iterations):
        mid = (low + high) / 2
        if power * (-math.expm1(-mid)) > mid:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def peak_wavelength(temperature_k: float, photons: bool = False) -> float:
    if not math.isfinite(temperature_k) or temperature_k <= 0:
        raise ValueError("temperature must be positive and finite")
    return H * C / (K * temperature_k * peak_root(4 if photons else 5))


def fractional_temperature_sensitivity(wavelength_m: float, temperature_k: float) -> float:
    """d ln(B_lambda)/d ln(T), also valid for photon radiance."""
    if wavelength_m <= 0 or temperature_k <= 0:
        raise ValueError("positive inputs required")
    x = H * C / (wavelength_m * K * temperature_k)
    return x / (-math.expm1(-x)) if x < 700 else x
