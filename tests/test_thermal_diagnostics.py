import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from thermal_diagnostics import (planck_lambda, bolometric_flux, top_hat_flux, colour_ratio,
                                 colour_temperature, colour_temperature_uncertainty, SIGMA_SB, WIEN_B)


class ThermalDiagnosticsTests(unittest.TestCase):
    def test_planck_reference(self):
        wl=500e-9; temp=5772.; h=6.62607015e-34;c=299792458.;k=1.380649e-23
        ref=2*h*c*c/wl**5/np.expm1(h*c/(wl*k*temp))
        self.assertAlmostEqual(float(planck_lambda(wl,temp))/ref,1,places=12)
    def test_rayleigh_jeans_limit(self):
        wl=.5;temp=5000
        approx=2*1.380649e-23*temp*299792458./wl**4
        self.assertAlmostEqual(float(planck_lambda(wl,temp))/approx,1,places=5)
    def test_wien_peak(self):
        temp=5772
        w=np.linspace(250e-9,1000e-9,50001)
        peak=w[np.argmax(planck_lambda(w,temp))]
        self.assertLess(abs(peak-WIEN_B/temp)/(WIEN_B/temp),3e-5)
    def test_stefan_boltzmann(self):
        for temp in (1000,5772,15000):
            self.assertLess(abs(bolometric_flux(temp)/(SIGMA_SB*temp**4)-1),1e-5)
    def test_bolometric_convergence(self):
        temp=5000
        f1=bolometric_flux(temp,samples=1025)
        f2=bolometric_flux(temp,samples=4097)
        self.assertLess(abs(f1/f2-1),2e-5)
    def test_top_hat_positive_scaling(self):
        self.assertGreater(top_hat_flux(7000,400,500),top_hat_flux(4000,400,500))
    def test_colour_inversion(self):
        for temp in (2500,5772,12000):
            recovered=colour_temperature(colour_ratio(temp))
            self.assertAlmostEqual(recovered,temp,places=6)
    def test_colour_monotonicity(self):
        self.assertLess(colour_ratio(3000),colour_ratio(6000))
        self.assertLess(colour_ratio(6000),colour_ratio(12000))
    def test_uncertainty_correlation(self):
        b=top_hat_flux(5772,400,500);r=top_hat_flux(5772,650,750)
        indep=colour_temperature_uncertainty(b,r,.01*b,.01*r,correlation=0)
        corr=colour_temperature_uncertainty(b,r,.01*b,.01*r,correlation=.8)
        self.assertAlmostEqual(indep['temperature_k'],5772,places=5)
        self.assertLess(corr['sigma_temperature_k'],indep['sigma_temperature_k'])
    def test_invalid_temperature(self):
        with self.assertRaises(ValueError): planck_lambda([500e-9],-1)
    def test_out_of_range_ratio(self):
        with self.assertRaises(ValueError): colour_temperature(100)
    def test_invalid_band_order(self):
        with self.assertRaises(ValueError): colour_ratio(5772,(650,750),(400,500))


if __name__ == '__main__': unittest.main()
