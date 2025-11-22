# Lunar South Pole Volatile Sublimation Calculator

A Python application to convert temperature values to sublimation rates for different volatile species at the lunar south pole. Includes support for time-averaged calculations and GeoTIFF raster processing.

## Overview

This tool calculates sublimation rates using the **Hertz-Knudsen equation**, which describes the flux of molecules leaving a surface in vacuum conditions (like the lunar surface). The calculator supports multiple volatile species commonly found or theorized to exist at the lunar south pole.

## Features

- **Accurate vapor pressure calculations** using Clausius-Clapeyron equation (Andreas et al. 2007 for H₂O)
- **Time-averaged sublimation rates** for diurnal/seasonal temperature variations
- **Micro cold trap analysis** accounting for surface roughness effects on volatile retention
- **GeoTIFF raster processing** to convert temperature maps to sublimation rate maps
- **Roughness-based processing** to model mixed-pixel sublimation in rough terrain
- **Multiple volatile species** support (H₂O, CO₂, CO, CH₄, NH₃, SO₂)
- **Flexible output units** (kg/m²/s, mm/yr depth loss, etc.)
- **Command-line and programmatic** interfaces

## Physics Background

### Hertz-Knudsen Equation

The sublimation rate is calculated using:

```
J = α × P_vap × √(M / (2πRT))
```

Where:
- `J` = sublimation flux (kg/m²/s)
- `α` = sticking coefficient (typically 1.0 for free sublimation)
- `P_vap` = vapor pressure (Pa)
- `M` = molecular mass (kg/mol)
- `R` = gas constant (8.314 J/mol/K)
- `T` = temperature (K)

### Vapor Pressure

Vapor pressure is calculated using the **Clausius-Clapeyron equation** for improved accuracy at low lunar temperatures:

```
ln(P/P₀) = -(L/R) × (1/T - 1/T₀)
```

Where:
- `L` = latent heat of sublimation (J/mol)
- `P₀`, `T₀` = reference pressure and temperature

This method is more accurate than Antoine equation for the extreme low temperatures found in permanently shadowed regions (PSRs).

### Micro Cold Traps

Surface roughness creates micro-scale permanently shadowed regions (micro cold traps) within pixels. These micro-PSRs can be significantly colder than the pixel-averaged temperature, enabling ice retention even in nominally "warm" locations.

**Key concepts:**
- Surface roughness from boulders, small craters, and terrain undulations creates shadows
- Rough terrain characterized by RMS slope (degrees) or RMS height (meters)
- Cold trap fraction: f_cold = (1 - cos(θ_RMS)) / 2 (cosine model)
- Temperature depression: T_cold ≈ 0.3 to 0.7 × T_illuminated (latitude-dependent)
- Mixed-pixel sublimation: J_mixed = f_illum × J_illum + f_cold × J_cold

**Impact:**
- 20-30° RMS slope can reduce sublimation by 100-1000×
- Extends ice stability boundary poleward by several degrees latitude
- Explains ice detection in "unexpected" warm locations
- Critical for accurate total ice inventory estimates

### Integrated Thermal Modeling

The `thermal_model.py` module integrates accurate 1-D thermal modeling with micro cold trap theory and sublimation physics. This provides a comprehensive framework for understanding lunar ice retention from kilometer to centimeter scales.

**Key components:**
- **heat1d thermal model** (Hayne et al. 2017) - 1-D heat conduction with temperature-dependent thermophysical properties
- **Hayne et al. (2021) micro cold trap theory** - Analytical crater shadows, rough surface models, lateral conduction limits
- **Multi-scale analysis** - Cold traps from 1 km down to 1 cm (10⁶ range in scale!)
- **Lateral heat conduction** - Eliminates cold traps < 1 cm due to thermal diffusion
- **Thermal skin depth** - δ = √(κP/π) where κ is thermal diffusivity, P is period

**Physical parameters:**
- Thermal conductivity: k = kc(1 + R350×T³) (temperature-dependent)
- Heat capacity: cp(T) from Ledlow et al. (1992)
- Diurnal skin depth: ~4.4 cm for lunar regolith
- Optimal RMS slope for cold trapping: σs ≈ 10-20°

**Total lunar cold trap area (Hayne et al. 2021):**
- ~40,000 km² total (~0.10% of lunar surface)
- South pole: ~23,000 km² (60%)
- North pole: ~17,000 km² (40%)
- Micro cold traps (<100m): ~2,500 km² (~10-20% of total)
- Most numerous cold traps: ~1 cm scale

**Usage:**
```python
from thermal_model import (integrated_sublimation_with_thermal,
                            LunarThermalProperties)
from vaporp_temp import VOLATILE_SPECIES

# Calculate sublimation accounting for thermal model + roughness
species = VOLATILE_SPECIES['H2O']
latitude = 85  # degrees
rms_slope = 20  # degrees
length_scale = 0.1  # meters

result = integrated_sublimation_with_thermal(
    species, latitude, rms_slope, length_scale
)

print(f"Cold trap fraction: {result['cold_trap_fraction']:.2%}")
print(f"Cold trap temp: {result['cold_trap_temp_K']:.1f} K")
print(f"Ice lifetime (1m deposit): {1000/result['cold_trap_only_rate_mm_yr']:.2e} years")
```

Run comprehensive thermal integration examples:
```bash
python example_thermal_integration.py
```

This demonstrates:
1. Crater shadow fractions at different d/D ratios
2. Lateral conduction limits on cold trap size
3. Rough surface cold trap fractions vs RMS slope
4. Scale-dependent cold trap areas
5. Integrated thermal + sublimation calculations
6. Validation against Hayne et al. (2021) published results

## Supported Volatile Species

- **H₂O** (Water ice) - Primary volatile of interest
- **CO₂** (Carbon dioxide)
- **CO** (Carbon monoxide)
- **CH₄** (Methane)
- **NH₃** (Ammonia)
- **SO₂** (Sulfur dioxide)

## Installation

### Basic Installation

Core functionality uses only Python standard library:
```bash
chmod +x vaporp_temp.py
```

### Optional Dependencies

For raster processing (GeoTIFF support):
```bash
pip install gdal numpy
```

Or using conda:
```bash
conda install -c conda-forge gdal numpy
```

## Usage

### 1. Basic Sublimation Calculations

Calculate H₂O sublimation at 110K:
```bash
python vaporp_temp.py -t 110 -s H2O
```

Calculate for multiple species at one temperature:
```bash
python vaporp_temp.py -t 100 -s H2O CO2 CO
```

Calculate H₂O at multiple temperatures:
```bash
python vaporp_temp.py -t 80 90 100 110 120 -s H2O
```

Calculate all species at 100K:
```bash
python vaporp_temp.py -t 100 --all
```

Specify custom sticking coefficient:
```bash
python vaporp_temp.py -t 110 -s H2O --alpha 0.8
```

Save results to file:
```bash
python vaporp_temp.py -t 110 -s H2O -o results.txt
```

### 2. Time-Averaged Sublimation Rates

Calculate time-averaged rates from temperature time series:

```bash
# Simple temperature array
python time_averaged_sublimation.py --temps 40 60 80 100 120 -s H2O

# From file (one temperature per line)
python time_averaged_sublimation.py --file temperatures.txt -s H2O

# With weights (e.g., hours at each temperature)
python time_averaged_sublimation.py --temps 100 120 140 --weights 12 8 4 -s H2O

# All species
python time_averaged_sublimation.py --temps 80 90 100 110 --all
```

**When to use time-averaging:**
- Modeling diurnal temperature cycles
- Seasonal variations at crater rims
- Long-term volatile stability analysis
- Locations with variable illumination

### 3. Raster Processing (GeoTIFF)

Convert temperature rasters to sublimation rate rasters:

```bash
# Basic raster conversion
python raster_sublimation.py -i temperature.tif -o h2o_sublim.tif -s H2O

# With custom sticking coefficient
python raster_sublimation.py -i temp.tif -o sublim.tif -s CO2 --alpha 0.8

# If input is in Celsius, convert to Kelvin
python raster_sublimation.py -i temp_celsius.tif -o sublim.tif -s H2O --offset 273.15

# Verbose output with statistics
python raster_sublimation.py -i temp.tif -o sublim.tif -s H2O -v
```

**Output format:**
- GeoTIFF with same projection/geotransform as input
- Units: kg/(m²·yr)
- LZW compression
- NoData values preserved

### 4. Micro Cold Trap Analysis

Analyze sublimation accounting for surface roughness and micro-scale permanently shadowed regions:

```bash
# Run comprehensive micro cold trap analysis
python micro_cold_trap_analysis.py

# Process rasters with temperature and roughness data
python raster_micro_coldtrap.py -t temp.tif -r slope_rms.tif -o sublim_mixed.tif -s H2O

# Use different roughness models
python raster_micro_coldtrap.py -t temp.tif -r slope.tif -o sublim.tif -s H2O --model exponential

# RMS height instead of slope
python raster_micro_coldtrap.py -t temp.tif -r height_rms.tif -o sublim.tif -s H2O --rtype height

# Fixed temperature depression
python raster_micro_coldtrap.py -t temp.tif -r slope.tif -o sublim.tif -s H2O --depression 50
```

**Programmatic usage:**
```python
from vaporp_temp import (VOLATILE_SPECIES, estimate_cold_trap_fraction,
                          calculate_mixed_pixel_sublimation)

# Estimate cold trap fraction from roughness
roughness_rms_slope = 25  # degrees
cold_trap_frac = estimate_cold_trap_fraction(roughness_rms_slope=roughness_rms_slope)

# Calculate mixed-pixel sublimation
species = VOLATILE_SPECIES['H2O']
illuminated_temp = 130  # K
cold_trap_temp = 70     # K

result = calculate_mixed_pixel_sublimation(
    species, illuminated_temp, cold_trap_frac, cold_trap_temp=cold_trap_temp
)

print(f"Mixed rate: {result['mixed_sublimation_rate_kg_m2_yr']:.2e} kg/(m²·yr)")
print(f"Cold trap only: {result['cold_trap_only_rate_kg_m2_yr']:.2e} kg/(m²·yr)")
```

**Roughness models:**
- `cosine`: f_cold = (1 - cos(slope)) / 2 (conservative, geometric)
- `linear`: f_cold = slope / 90° (simple linear)
- `exponential`: f_cold = 1 - exp(-slope/15°) (realistic for very rough terrain)

### Command Line Arguments

- `-t, --temperature` : Temperature value(s) in Kelvin (required)
- `-s, --species` : Species to calculate (choose from: H2O, CO2, CO, CH4, NH3, SO2)
- `--all` : Calculate for all available species
- `-a, --alpha` : Sticking coefficient (default: 1.0)
- `-o, --output` : Output file path (optional)

## Output Units

The calculator provides sublimation rates in multiple units:

- **kg/(m²·s)** - SI unit for mass flux
- **g/(m²·s)** - Grams per square meter per second
- **g/(m²·hr)** - Grams per square meter per hour
- **kg/(m²·yr)** - Kilograms per square meter per year
- **mm/yr** - Depth loss in millimeters per year (assumes ice density)

## Typical Lunar South Pole Temperatures

- **Permanently Shadowed Regions (PSRs)**: 30-110 K
- **Crater floors**: 40-100 K
- **Illuminated regions**: 200-400 K
- **Coldest recorded**: ~30 K (Shackleton crater)

## Example Output

```
======================================================================
Species: H2O
======================================================================
Temperature:           110.00 K (-163.15 °C)
Vapor Pressure:        1.26e-06 Pa

Sublimation Rates:
  2.91e-10 kg/(m²·s)
  2.91e-07 g/(m²·s)
  1.05e-03 g/(m²·hr)
  9.18e-03 kg/(m²·yr)
  9.98e-03 mm/yr (depth loss)
======================================================================
```

## Scientific Applications

- Modeling volatile loss from lunar cold traps (including micro-PSRs)
- Estimating ice stability in permanently shadowed regions
- Analyzing effect of surface roughness on ice retention
- Planning resource utilization missions (ice prospecting)
- Understanding volatile migration and distribution
- Analyzing LCROSS and other lunar mission data
- Calculating total ice inventory accounting for rough terrain

## Scientific References

### Vapor Pressure Data
- **Andreas, E. L. (2007).** "New estimates for the sublimation rate for ice on the Moon." *Icarus*, 186(1), 24-30.
  - Source for H₂O Clausius-Clapeyron parameters
- **Fray, N., & Schmitt, B. (2009).** "Sublimation of ices of astrophysical interest: A bibliographic review." *Planetary and Space Science*, 57(14-15), 2053-2080.
  - Source for CO₂ and other volatile parameters

### Temperature Data
- **Paige, D. A., et al. (2010).** "Diviner Lunar Radiometer Observations of Cold Traps in the Moon's South Polar Region." *Science*, 330(6003), 479-482.
  - LRO Diviner temperature measurements of PSRs

### Sublimation Physics
- **Hertz-Knudsen equation** for molecular flux from surfaces in vacuum
- **Clausius-Clapeyron equation** for vapor pressure as function of temperature
- **Zhang, J. A., & Paige, D. A. (2009).** "Cold-trapped organic compounds at the poles of the Moon and Mercury: Implications for origins." *Geophysical Research Letters*, 36(16).

### Thermal Modeling
- **Hayne, P. O., et al. (2017).** "Evidence for exposed water ice in the Moon's south polar regions from Lunar Reconnaissance Orbiter ultraviolet albedo and temperature measurements." *Icarus*, 255, 58-69.
  - Source for heat1d 1-D thermal model and lunar thermophysical properties
- **Mitchell, D. L., & de Pater, I. (1994).** "Microwave imaging of Mercury's thermal emission at wavelengths from 0.3 to 20.5 cm." *Icarus*, 110(1), 2-32.
  - Temperature-dependent thermal conductivity model
- **Ledlow, M. J., et al. (1992).** "Subsurface emissions from Mercury: VLA radio observations at 2 and 6 cm." *Astrophysical Journal*, 384, 640-655.
  - Heat capacity measurements for regolith

### Micro Cold Traps and Surface Roughness
- **Hayne, P. O., et al. (2021).** "Micro cold traps on the Moon." *Nature Astronomy*, 5(5), 462-467.
  - Micro-scale PSRs from surface roughness, multi-scale cold trap theory
- **Rubanenko, L., et al. (2019).** "Stability of ice on the Moon with rough topography." *Icarus*, 296, 99-109.
  - Demonstrates how surface roughness extends ice stability boundaries
- **Schorghofer, N., & Taylor, G. J. (2007).** "Subsurface migration of H₂O at lunar cold traps." *Journal of Geophysical Research*, 112(E2).
  - Early work on cold trap physics
- **Deutsch, A. N., et al. (2020).** "Analyzing the ages of south polar craters on the Moon: Implications for the sources and evolution of surface water ice." *Icarus*, 336, 113455.
  - Ice stability in rough polar terrain

## License

This tool is provided for scientific and educational purposes.

## Contributing

Feel free to add more volatile species or improve the thermodynamic models.
