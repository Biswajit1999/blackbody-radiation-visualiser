# Photon-counting versus energy radiance (9 November 2026)

Independent, ideal-blackbody diagnostics; does not modify the interactive application.

The wavelength density of **energy** radiance has its maximum at x = 4.965114..., whereas the wavelength density of **photon-number** radiance has its maximum at x satisfying 4(1-exp(-x)) = x. Both use x=hc/(lambda kT). The photon peak is at a longer wavelength. Neither is the peak of the corresponding per-frequency density: peak positions depend on the spectral-density variable.

The second diagnostic is the local temperature elasticity d ln B_lambda/d ln T = x/(1-exp(-x)). This quantifies wavelength-dependent sensitivity to temperature, but not instrument noise or temperature-estimation uncertainty.

Run from repository root: `python -m unittest discover -s tests -p 'test_photon_peak_audit.py'`.

AI assistance was used for code and tests. The numerical model uses ideal blackbody assumptions, no extinction, finite bandpass, detector quantum efficiency or measured spectra.
