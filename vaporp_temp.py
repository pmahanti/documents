#!/usr/bin/env python3
"""
Lunar South Pole Volatile Sublimation Rate Calculator

This application converts temperature values to sublimation rates for different
volatile species at the lunar south pole using the Hertz-Knudsen equation.
"""

import math
import argparse
import sys


class VolatileSpecies:
    """Class representing a volatile species with its thermodynamic properties."""

    def __init__(self, name, molecular_mass, A=None, B=None, C=0, T_triple=None,
                 use_clausius_clapeyron=False, latent_heat=None, P0=None, T0=None):
        """
        Initialize a volatile species.

        Parameters:
        -----------
        name : str
            Name of the species
        molecular_mass : float
            Molecular mass in kg/mol
        A, B, C : float
            Antoine equation coefficients for log10(P) = A - B/(T+C)
            where P is in Pa and T is in K
        T_triple : float
            Triple point temperature in K (optional, for validation)
        use_clausius_clapeyron : bool
            If True, use Clausius-Clapeyron equation instead of Antoine
        latent_heat : float
            Latent heat of sublimation in J/mol (for Clausius-Clapeyron)
        P0, T0 : float
            Reference pressure (Pa) and temperature (K) for Clausius-Clapeyron
        """
        self.name = name
        self.molecular_mass = molecular_mass  # kg/mol
        self.A = A
        self.B = B
        self.C = C
        self.T_triple = T_triple
        self.use_clausius_clapeyron = use_clausius_clapeyron
        self.latent_heat = latent_heat  # J/mol
        self.P0 = P0  # Reference pressure in Pa
        self.T0 = T0  # Reference temperature in K

    def vapor_pressure(self, temperature):
        """
        Calculate vapor pressure using Antoine or Clausius-Clapeyron equation.

        Parameters:
        -----------
        temperature : float
            Temperature in Kelvin

        Returns:
        --------
        float
            Vapor pressure in Pascals
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive (K)")

        if self.use_clausius_clapeyron and self.latent_heat and self.P0 and self.T0:
            # Clausius-Clapeyron: ln(P/P0) = -L/R * (1/T - 1/T0)
            # More accurate for water ice at low temperatures
            R = 8.314  # J/(mol·K)
            ln_p = math.log(self.P0) - (self.latent_heat / R) * (1/temperature - 1/self.T0)
            return math.exp(ln_p)
        else:
            # Antoine equation: log10(P) = A - B/(T+C)
            log_p = self.A - self.B / (temperature + self.C)
            return 10 ** log_p

    def sublimation_rate(self, temperature, alpha=1.0):
        """
        Calculate sublimation rate using Hertz-Knudsen equation.

        Parameters:
        -----------
        temperature : float
            Temperature in Kelvin
        alpha : float
            Sticking coefficient (default 1.0 for free sublimation)

        Returns:
        --------
        dict
            Dictionary containing sublimation rates in different units
        """
        R = 8.314  # Gas constant in J/(mol·K)

        # Get vapor pressure
        P_vap = self.vapor_pressure(temperature)

        # Hertz-Knudsen equation
        # J = α * P_vap * sqrt(M / (2 * π * R * T))
        flux_molecules = alpha * P_vap * math.sqrt(
            self.molecular_mass / (2 * math.pi * R * temperature)
        )  # kg/(m²·s)

        # Convert to different useful units
        results = {
            'temperature_K': temperature,
            'temperature_C': temperature - 273.15,
            'vapor_pressure_Pa': P_vap,
            'sublimation_rate_kg_m2_s': flux_molecules,
            'sublimation_rate_g_m2_s': flux_molecules * 1000,
            'sublimation_rate_g_m2_hr': flux_molecules * 1000 * 3600,
            'sublimation_rate_kg_m2_yr': flux_molecules * 365.25 * 24 * 3600,
            'sublimation_rate_mm_yr': self._to_mm_per_year(flux_molecules)
        }

        return results

    def _to_mm_per_year(self, flux_kg_m2_s):
        """
        Convert mass flux to depth loss in mm/year.
        Assumes ice density of ~920 kg/m³ for H2O, adjusted for other species.
        """
        # Rough density estimates
        density_map = {
            'H2O': 920,
            'CO2': 1560,
            'CO': 790,
            'CH4': 420,
            'NH3': 820,
            'SO2': 1460
        }
        density = density_map.get(self.name, 1000)  # kg/m³

        # flux (kg/m²/s) / density (kg/m³) = m/s
        # Convert to mm/year
        m_per_s = flux_kg_m2_s / density
        mm_per_year = m_per_s * 1000 * 365.25 * 24 * 3600

        return mm_per_year


# Database of volatile species relevant to the lunar south pole
# Updated with more accurate vapor pressure data from literature
VOLATILE_SPECIES = {
    'H2O': VolatileSpecies(
        name='H2O',
        molecular_mass=0.018015,  # kg/mol
        use_clausius_clapeyron=True,
        # Latent heat from Andreas et al. (2007), accurate for T < 273K
        latent_heat=51058,  # J/mol (sublimation)
        P0=611.657,  # Pa at T0
        T0=273.16,  # K (triple point)
        T_triple=273.16
    ),
    'CO2': VolatileSpecies(
        name='CO2',
        molecular_mass=0.04401,  # kg/mol
        use_clausius_clapeyron=True,
        latent_heat=25230,  # J/mol (Fray and Schmitt 2009)
        P0=518500,  # Pa at T0
        T0=216.58,  # K (triple point)
        T_triple=216.58
    ),
    'CO': VolatileSpecies(
        name='CO',
        molecular_mass=0.02801,  # kg/mol
        use_clausius_clapeyron=True,
        latent_heat=8140,  # J/mol
        P0=15400,  # Pa at T0
        T0=68.15,  # K (triple point)
        T_triple=68.15
    ),
    'CH4': VolatileSpecies(
        name='CH4',
        molecular_mass=0.01604,  # kg/mol
        use_clausius_clapeyron=True,
        latent_heat=9820,  # J/mol
        P0=11696,  # Pa at T0
        T0=90.69,  # K (triple point)
        T_triple=90.69
    ),
    'NH3': VolatileSpecies(
        name='NH3',
        molecular_mass=0.01703,  # kg/mol
        use_clausius_clapeyron=True,
        latent_heat=30800,  # J/mol
        P0=6091,  # Pa at T0
        T0=195.4,  # K (triple point)
        T_triple=195.4
    ),
    'SO2': VolatileSpecies(
        name='SO2',
        molecular_mass=0.06407,  # kg/mol
        use_clausius_clapeyron=True,
        latent_heat=25140,  # J/mol
        P0=1670,  # Pa at T0
        T0=197.64,  # K (triple point)
        T_triple=197.64
    )
}


def calculate_time_averaged_sublimation(species, temperature_array, weights=None, alpha=1.0):
    """
    Calculate time-averaged sublimation rate from a temperature time series.

    Parameters:
    -----------
    species : VolatileSpecies
        The volatile species object
    temperature_array : list or array
        Array of temperature values in Kelvin over time
    weights : list or array, optional
        Weighting factors for each temperature (e.g., time duration)
        If None, assumes equal weighting
    alpha : float
        Sticking coefficient (default 1.0)

    Returns:
    --------
    dict
        Dictionary with average temperature and time-averaged sublimation rates
    """
    if weights is None:
        weights = [1.0] * len(temperature_array)

    if len(temperature_array) != len(weights):
        raise ValueError("Temperature array and weights must have same length")

    total_weight = sum(weights)
    weighted_temp = sum(t * w for t, w in zip(temperature_array, weights)) / total_weight

    # Calculate sublimation rate for each temperature
    sublimation_rates = []
    for temp in temperature_array:
        try:
            result = species.sublimation_rate(temp, alpha=alpha)
            sublimation_rates.append(result['sublimation_rate_kg_m2_s'])
        except:
            sublimation_rates.append(0.0)

    # Time-averaged sublimation rate
    avg_rate = sum(r * w for r, w in zip(sublimation_rates, weights)) / total_weight

    results = {
        'average_temperature_K': weighted_temp,
        'average_temperature_C': weighted_temp - 273.15,
        'time_averaged_rate_kg_m2_s': avg_rate,
        'time_averaged_rate_g_m2_s': avg_rate * 1000,
        'time_averaged_rate_g_m2_hr': avg_rate * 1000 * 3600,
        'time_averaged_rate_kg_m2_yr': avg_rate * 365.25 * 24 * 3600,
        'time_averaged_rate_mm_yr': species._to_mm_per_year(avg_rate),
        'min_temperature_K': min(temperature_array),
        'max_temperature_K': max(temperature_array),
        'num_samples': len(temperature_array)
    }

    return results


def process_temperature_raster(input_raster, output_raster, species_name, alpha=1.0,
                                 input_scale=1.0, input_offset=0.0):
    """
    Convert a temperature raster to a sublimation rate raster.

    Parameters:
    -----------
    input_raster : str
        Path to input temperature raster file (GeoTIFF)
        Temperature should be in Kelvin, or use scale/offset to convert
    output_raster : str
        Path to output sublimation rate raster (GeoTIFF)
        Units: kg/(m²·yr)
    species_name : str
        Name of volatile species (e.g., 'H2O')
    alpha : float
        Sticking coefficient (default 1.0)
    input_scale : float
        Scale factor for input (Temperature_K = input_value * scale + offset)
    input_offset : float
        Offset for input temperature

    Returns:
    --------
    dict
        Statistics about the processing
    """
    try:
        from osgeo import gdal, gdal_array
        import numpy as np
        gdal.UseExceptions()
    except ImportError:
        raise ImportError("GDAL is required for raster processing. Install with: pip install gdal")

    # Get species
    if species_name not in VOLATILE_SPECIES:
        raise ValueError(f"Unknown species: {species_name}. Available: {list(VOLATILE_SPECIES.keys())}")

    species = VOLATILE_SPECIES[species_name]

    # Open input raster
    src_ds = gdal.Open(input_raster, gdal.GA_ReadOnly)
    if src_ds is None:
        raise ValueError(f"Could not open input raster: {input_raster}")

    # Read raster data
    band = src_ds.GetRasterBand(1)
    temp_array = band.ReadAsArray()
    nodata = band.GetNoDataValue()

    # Apply scale and offset to get temperature in Kelvin
    temp_kelvin = temp_array * input_scale + input_offset

    # Create output array
    sublimation_array = np.zeros_like(temp_kelvin, dtype=np.float32)

    # Calculate sublimation rate for each pixel
    valid_mask = np.ones_like(temp_kelvin, dtype=bool)
    if nodata is not None:
        valid_mask = (temp_array != nodata) & (temp_kelvin > 0)
    else:
        valid_mask = temp_kelvin > 0

    # Vectorized calculation for valid pixels
    valid_temps = temp_kelvin[valid_mask]

    if len(valid_temps) > 0:
        # Calculate vapor pressure for all valid temperatures
        R = 8.314
        if species.use_clausius_clapeyron and species.latent_heat:
            ln_p = np.log(species.P0) - (species.latent_heat / R) * (1/valid_temps - 1/species.T0)
            P_vap = np.exp(ln_p)
        else:
            log_p = species.A - species.B / (valid_temps + species.C)
            P_vap = 10 ** log_p

        # Hertz-Knudsen equation
        flux_kg_m2_s = alpha * P_vap * np.sqrt(
            species.molecular_mass / (2 * np.pi * R * valid_temps)
        )

        # Convert to kg/(m²·yr)
        flux_kg_m2_yr = flux_kg_m2_s * 365.25 * 24 * 3600

        # Assign to output array
        sublimation_array[valid_mask] = flux_kg_m2_yr

    # Create output raster
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_raster, src_ds.RasterXSize, src_ds.RasterYSize,
                            1, gdal.GDT_Float32, ['COMPRESS=LZW'])

    # Copy geotransform and projection
    out_ds.SetGeoTransform(src_ds.GetGeoTransform())
    out_ds.SetProjection(src_ds.GetProjection())

    # Write data
    out_band = out_ds.GetRasterBand(1)
    if nodata is not None:
        out_band.SetNoDataValue(0.0)
        sublimation_array[~valid_mask] = 0.0

    out_band.WriteArray(sublimation_array)
    out_band.SetDescription(f"{species_name} sublimation rate (kg/m²/yr)")

    # Calculate statistics
    stats = {
        'species': species_name,
        'input_file': input_raster,
        'output_file': output_raster,
        'min_temperature_K': float(np.min(valid_temps)) if len(valid_temps) > 0 else None,
        'max_temperature_K': float(np.max(valid_temps)) if len(valid_temps) > 0 else None,
        'mean_temperature_K': float(np.mean(valid_temps)) if len(valid_temps) > 0 else None,
        'min_sublimation_rate': float(np.min(sublimation_array[valid_mask])) if len(valid_temps) > 0 else None,
        'max_sublimation_rate': float(np.max(sublimation_array[valid_mask])) if len(valid_temps) > 0 else None,
        'mean_sublimation_rate': float(np.mean(sublimation_array[valid_mask])) if len(valid_temps) > 0 else None,
        'valid_pixels': int(np.sum(valid_mask)),
        'total_pixels': temp_array.size
    }

    # Clean up
    out_band.FlushCache()
    out_ds = None
    src_ds = None

    return stats


def format_results(species_name, results):
    """Format sublimation rate results for display."""
    output = []
    output.append(f"\n{'='*70}")
    output.append(f"Species: {species_name}")
    output.append(f"{'='*70}")

    if 'temperature_K' in results:
        # Single temperature result
        output.append(f"Temperature:           {results['temperature_K']:.2f} K ({results['temperature_C']:.2f} °C)")
        output.append(f"Vapor Pressure:        {results['vapor_pressure_Pa']:.2e} Pa")
        output.append(f"\nSublimation Rates:")
        output.append(f"  {results['sublimation_rate_kg_m2_s']:.2e} kg/(m²·s)")
        output.append(f"  {results['sublimation_rate_g_m2_s']:.2e} g/(m²·s)")
        output.append(f"  {results['sublimation_rate_g_m2_hr']:.2e} g/(m²·hr)")
        output.append(f"  {results['sublimation_rate_kg_m2_yr']:.2e} kg/(m²·yr)")
        output.append(f"  {results['sublimation_rate_mm_yr']:.2e} mm/yr (depth loss)")
    elif 'average_temperature_K' in results:
        # Time-averaged result
        output.append(f"Average Temperature:   {results['average_temperature_K']:.2f} K ({results['average_temperature_C']:.2f} °C)")
        output.append(f"Temperature Range:     {results['min_temperature_K']:.2f} - {results['max_temperature_K']:.2f} K")
        output.append(f"Number of Samples:     {results['num_samples']}")
        output.append(f"\nTime-Averaged Sublimation Rates:")
        output.append(f"  {results['time_averaged_rate_kg_m2_s']:.2e} kg/(m²·s)")
        output.append(f"  {results['time_averaged_rate_g_m2_s']:.2e} g/(m²·s)")
        output.append(f"  {results['time_averaged_rate_g_m2_hr']:.2e} g/(m²·hr)")
        output.append(f"  {results['time_averaged_rate_kg_m2_yr']:.2e} kg/(m²·yr)")
        output.append(f"  {results['time_averaged_rate_mm_yr']:.2e} mm/yr (depth loss)")

    output.append(f"{'='*70}\n")

    return '\n'.join(output)


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description='Calculate sublimation rates for lunar south pole volatiles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available species: {', '.join(VOLATILE_SPECIES.keys())}

Examples:
  # Calculate H2O sublimation at 110K
  python vaporp_temp.py -t 110 -s H2O

  # Calculate for multiple species
  python vaporp_temp.py -t 100 -s H2O CO2 CO

  # Temperature range for H2O
  python vaporp_temp.py -t 80 90 100 110 -s H2O

  # All species at 100K
  python vaporp_temp.py -t 100 --all
"""
    )

    parser.add_argument('-t', '--temperature',
                        type=float,
                        nargs='+',
                        required=True,
                        help='Temperature value(s) in Kelvin')

    parser.add_argument('-s', '--species',
                        nargs='+',
                        choices=list(VOLATILE_SPECIES.keys()),
                        help='Volatile species to calculate')

    parser.add_argument('--all',
                        action='store_true',
                        help='Calculate for all available species')

    parser.add_argument('-a', '--alpha',
                        type=float,
                        default=1.0,
                        help='Sticking coefficient (default: 1.0)')

    parser.add_argument('-o', '--output',
                        type=str,
                        help='Output file (optional, prints to stdout if not specified)')

    args = parser.parse_args()

    # Determine which species to process
    if args.all:
        species_list = list(VOLATILE_SPECIES.keys())
    elif args.species:
        species_list = args.species
    else:
        parser.error("Either --species or --all must be specified")

    # Process calculations
    results_text = []
    results_text.append(f"\nLunar South Pole Volatile Sublimation Calculator")
    results_text.append(f"Sticking coefficient (α): {args.alpha}")

    for temp in args.temperature:
        for species_name in species_list:
            try:
                species = VOLATILE_SPECIES[species_name]
                results = species.sublimation_rate(temp, alpha=args.alpha)
                results_text.append(format_results(species_name, results))
            except ValueError as e:
                results_text.append(f"\nError calculating {species_name} at {temp} K: {e}\n")
            except Exception as e:
                results_text.append(f"\nUnexpected error for {species_name} at {temp} K: {e}\n")

    # Output results
    output_str = '\n'.join(results_text)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_str)
        print(f"Results written to {args.output}")
    else:
        print(output_str)


if __name__ == '__main__':
    main()
