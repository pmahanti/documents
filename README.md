# Lunar South Pole Volatile Sublimation Calculator

A Python application to convert temperature values to sublimation rates for different volatile species at the lunar south pole. Includes support for time-averaged calculations and GeoTIFF raster processing.

## Overview

This tool calculates sublimation rates using the **Hertz-Knudsen equation**, which describes the flux of molecules leaving a surface in vacuum conditions (like the lunar surface). The calculator supports multiple volatile species commonly found or theorized to exist at the lunar south pole.

## Features

- **Accurate vapor pressure calculations** using Clausius-Clapeyron equation (Andreas et al. 2007 for H₂O)
- **Time-averaged sublimation rates** for diurnal/seasonal temperature variations
- **GeoTIFF raster processing** to convert temperature maps to sublimation rate maps
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

- Modeling volatile loss from lunar cold traps
- Estimating ice stability in permanently shadowed regions
- Planning resource utilization missions
- Understanding volatile migration and distribution
- Analyzing LCROSS and other lunar mission data

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

## License

This tool is provided for scientific and educational purposes.

## Contributing

Feel free to add more volatile species or improve the thermodynamic models.
