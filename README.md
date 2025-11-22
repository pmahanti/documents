# Lunar South Pole Volatile Sublimation Calculator

A Python application to convert temperature values to sublimation rates for different volatile species at the lunar south pole.

## Overview

This tool calculates sublimation rates using the **Hertz-Knudsen equation**, which describes the flux of molecules leaving a surface in vacuum conditions (like the lunar surface). The calculator supports multiple volatile species commonly found or theorized to exist at the lunar south pole.

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

Vapor pressure is calculated using the **Antoine equation**:

```
log₁₀(P) = A - B/(T + C)
```

Species-specific coefficients are included for accurate calculations across relevant temperature ranges.

## Supported Volatile Species

- **H₂O** (Water ice) - Primary volatile of interest
- **CO₂** (Carbon dioxide)
- **CO** (Carbon monoxide)
- **CH₄** (Methane)
- **NH₃** (Ammonia)
- **SO₂** (Sulfur dioxide)

## Installation

No special dependencies required - uses only Python standard library.

```bash
chmod +x vaporp_temp.py
```

## Usage

### Basic Examples

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

### Advanced Options

Specify custom sticking coefficient:
```bash
python vaporp_temp.py -t 110 -s H2O --alpha 0.8
```

Save results to file:
```bash
python vaporp_temp.py -t 110 -s H2O -o results.txt
```

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

## References

- Hertz-Knudsen equation for sublimation in vacuum
- Antoine equation vapor pressure correlations
- Lunar surface temperature data from LRO Diviner
- Volatiles detected by LCROSS impact experiment

## License

This tool is provided for scientific and educational purposes.

## Contributing

Feel free to add more volatile species or improve the thermodynamic models.
