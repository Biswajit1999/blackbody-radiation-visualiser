import unittest
import numpy as np
from bandpass_robustness import planck_lambda, band_mean, log_color_sensitivity

class TestBandpass(unittest.TestCase):
    def test_rayleigh_jeans(self):
        lam=np.array([1e-3]); t=6000
        exact=planck_lambda(lam,t)[0]
        approx=2*1.380649e-23*t*299792458/lam[0]**4
        self.assertLess(abs(exact/approx-1),.002)
    def test_constant_band(self):
        wave=np.linspace(400e-9,500e-9,1001)
        self.assertAlmostEqual(band_mean(wave,np.ones(len(wave)),5800),np.trapezoid(planck_lambda(wave,5800),wave)/(wave[-1]-wave[0]),delta=1e-7)
    def test_temperature_monotonic(self):
        wave=np.linspace(450e-9,550e-9,101)
        self.assertGreater(band_mean(wave,np.ones(len(wave)),6000),band_mean(wave,np.ones(len(wave)),5000))
    def test_color_derivative(self):
        a=np.linspace(400e-9,500e-9,200); b=np.linspace(700e-9,800e-9,200)
        d1=log_color_sensitivity(a,np.ones(len(a)),b,np.ones(len(b)),5700,1e-3)
        d2=log_color_sensitivity(a,np.ones(len(a)),b,np.ones(len(b)),5700,1e-4)
        self.assertGreater(d1,0)
        self.assertAlmostEqual(d1,d2,delta=1e-5)
if __name__=='__main__': unittest.main()
