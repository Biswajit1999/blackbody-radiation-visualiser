"""Reproducible ideal-blackbody convergence and two-band sensitivity experiment."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from thermal_diagnostics import (bolometric_flux, colour_ratio, colour_temperature,
                                 colour_temperature_uncertainty, top_hat_flux, SIGMA_SB, WIEN_B)


def analyse(temperature=5772.0):
    fluxes = {str(n): bolometric_flux(temperature, samples=n) for n in (257, 1025, 4097)}
    blue = top_hat_flux(temperature, 400, 500)
    red = top_hat_flux(temperature, 650, 750)
    return {"model": "ideal blackbody, top-hat filters, independent or declared correlated Gaussian flux errors",
            "temperature_k": temperature, "wien_peak_nm": WIEN_B/temperature*1e9,
            "stefan_boltzmann_flux_w_m2": SIGMA_SB*temperature**4,
            "integrated_flux_w_m2": fluxes,
            "relative_flux_error_4097": fluxes["4097"]/(SIGMA_SB*temperature**4)-1,
            "blue_400_500_nm_w_m2": blue, "red_650_750_nm_w_m2": red,
            "colour_ratio": colour_ratio(temperature),
            "recovered_colour_temperature_k": colour_temperature(blue/red),
            "one_percent_uncorrelated_flux_errors": colour_temperature_uncertainty(blue,red,.01*blue,.01*red),
            "one_percent_flux_errors_rho_0_8": colour_temperature_uncertainty(blue,red,.01*blue,.01*red,correlation=.8)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--temperature",type=float,default=5772)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    result=json.dumps(analyse(args.temperature),indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(result+"\n",encoding="utf-8")
    else:
        print(result)


if __name__ == '__main__': main()
