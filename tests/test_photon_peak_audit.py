import math
import unittest
from research.photon_peak_audit import (
    peak_root, peak_wavelength, spectral_radiance, fractional_temperature_sensitivity
)


class BlackbodyDiagnostics(unittest.TestCase):
    def test_wien_energy(self):
        self.assertAlmostEqual(peak_wavelength(5800) * 5800, 2.897771955e-3, delta=2e-10)

    def test_photon_peak_longer(self):
        self.assertGreater(peak_wavelength(5800, True), peak_wavelength(5800))

    def test_photon_energy_conversion(self):
        wavelength, temperature = 550e-9, 5800
        energy = spectral_radiance(wavelength, temperature)
        photons = spectral_radiance(wavelength, temperature, True)
        self.assertAlmostEqual(photons / energy, wavelength / (6.62607015e-34 * 299792458.0))

    def test_temperature_jacobian(self):
        w, t, eps = 600e-9, 5000., 1e-5
        upper = spectral_radiance(w, t * (1 + eps))
        lower = spectral_radiance(w, t * (1 - eps))
        fd = (math.log(upper) - math.log(lower)) / (math.log1p(eps) - math.log1p(-eps))
        self.assertAlmostEqual(fd, fractional_temperature_sensitivity(w, t), places=8)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            peak_wavelength(0)
        with self.assertRaises(ValueError):
            spectral_radiance(-1, 5800)


if __name__ == "__main__":
    unittest.main()
