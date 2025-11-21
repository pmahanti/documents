# Impact Mechanism: Crater Scaling Calculator

Python implementation of impact crater scaling laws for computing impact parameters from observed crater dimensions.

## Overview

This tool uses **pi-group scaling relationships** (Holsapple 1993, Holsapple & Housen 2007) to perform inverse calculations: given a crater's size, compute the impactor properties that created it.

### Key Features

- ✓ **Inverse scaling** - From crater → impactor parameters
- ✓ **Multiple materials** - Predefined target materials (lunar regolith, rock, ice, soil)
- ✓ **Multiple impactor types** - Rocky asteroids, iron meteorites, comets
- ✓ **Regime detection** - Automatically identifies strength vs. gravity scaling
- ✓ **Velocity trade-offs** - Explore impactor size vs. velocity relationships
- ✓ **Batch processing** - Handle multiple craters efficiently

## Installation

### Requirements

```bash
pip install numpy scipy matplotlib pandas
```

For Jupyter notebook examples:
```bash
pip install jupyter
```

Or install all at once:
```bash
pip install -r requirements.txt
```

## Quick Start

### Command Line Example

```python
from impact_scaling import ImpactScaling, MATERIALS, IMPACTORS, format_results

# Set up materials
target = MATERIALS['lunar_regolith']
impactor = IMPACTORS['asteroid_rock']

# Create calculator
calc = ImpactScaling(target, impactor.density)

# Compute impactor parameters
results = calc.compute_impactor_params(
    D=100,      # Crater diameter (m)
    d=20,       # Crater depth (m)
    U=15000,    # Impact velocity (m/s)
    crater_type='simple'
)

# Display results
format_results(results, 100, 20, 15000, target.name, impactor.name)
```

### Example Output

```
======================================================================
IMPACT PARAMETER CALCULATION RESULTS
======================================================================

Input Parameters:
  Crater Diameter:        100.0 m (0.10 km)
  Crater Depth:           20.0 m
  Depth/Diameter Ratio:   0.200
  Impact Velocity:        15,000 m/s (15.0 km/s)
  Target Material:        Lunar Regolith
  Impactor Type:          Rocky Asteroid

──────────────────────────────────────────────────────────────────────
Computed Impactor Properties:
──────────────────────────────────────────────────────────────────────
  Impactor Diameter:      1.8 m
  Impactor Mass:          7.66e+03 kg (7.7 metric tons)
  Impact Energy:          8.62e+11 J (0.21 kt TNT)
  Impact Momentum:        1.15e+08 kg⋅m/s

──────────────────────────────────────────────────────────────────────
Scaling Analysis:
──────────────────────────────────────────────────────────────────────
  π₂ (gravity):           6.32e-09
  π₃ (strength):          2.96e-09
  π₄ (density ratio):     0.556
  Dominant Regime:        GRAVITY
  → Gravity controls crater size
```

## Available Materials

### Target Materials

| Material | Density (kg/m³) | Strength (Pa) | Gravity (m/s²) | Environment |
|----------|-----------------|---------------|----------------|-------------|
| `lunar_regolith` | 1500 | 1×10³ | 1.62 | Moon |
| `lunar_mare` | 3100 | 1×10⁷ | 1.62 | Moon |
| `lunar_highland` | 2800 | 5×10⁶ | 1.62 | Moon |
| `sandstone` | 2200 | 3×10⁷ | 9.81 | Earth |
| `granite` | 2750 | 2×10⁸ | 9.81 | Earth |
| `sand` | 1650 | 1×10³ | 9.81 | Earth |
| `dry_soil` | 1600 | 1×10⁴ | 9.81 | Earth |
| `water_ice` | 920 | 1×10⁶ | 1.62 | Icy bodies |

### Impactor Types

| Type | Density (kg/m³) | Description |
|------|-----------------|-------------|
| `asteroid_rock` | 2700 | Typical stony asteroid |
| `asteroid_metal` | 7800 | Iron meteorite |
| `comet_ice` | 500 | Porous cometary ice |
| `comet_ice_dense` | 1000 | Dense comet nucleus |

## Usage Examples

### 1. Basic Calculation

```python
from impact_scaling import ImpactScaling, MATERIALS, IMPACTORS

target = MATERIALS['lunar_mare']
impactor_density = IMPACTORS['asteroid_rock'].density

calc = ImpactScaling(target, impactor_density)

results = calc.compute_impactor_params(
    D=1000,     # 1 km crater
    d=200,      # 200 m deep
    U=20000,    # 20 km/s
    crater_type='simple'
)

print(f"Impactor diameter: {results['impactor_diameter']:.1f} m")
print(f"Impact energy: {results['impact_energy']/4.184e15:.2f} Mt TNT")
```

### 2. Velocity Trade-off Study

When impact velocity is uncertain, explore the range of possible impactor sizes:

```python
scan = calc.velocity_scan(
    D=500,                    # 500 m crater
    d=100,                    # 100 m depth
    U_range=(10000, 30000),   # 10-30 km/s range
    n_points=20
)

import matplotlib.pyplot as plt
plt.plot(scan['velocities']/1000, scan['impactor_diameters'])
plt.xlabel('Impact Velocity (km/s)')
plt.ylabel('Impactor Diameter (m)')
plt.title('Impactor Size vs. Velocity Trade-off')
plt.show()
```

### 3. Custom Materials

Define your own target material:

```python
from impact_scaling import Material, ImpactScaling

my_material = Material(
    name='Martian Regolith',
    density=1400,      # kg/m³
    strength=2e3,      # Pa
    gravity=3.71,      # m/s² (Mars)
    K1=0.132,
    mu=0.41,           # Granular material
    nu=0.40,
)

calc = ImpactScaling(my_material, impactor_density=2700)
```

### 4. Batch Processing Multiple Craters

```python
import pandas as pd

craters = pd.DataFrame({
    'name': ['Crater_A', 'Crater_B', 'Crater_C'],
    'diameter': [150, 500, 1200],
    'depth': [30, 100, 240],
})

results = []
for _, crater in craters.iterrows():
    res = calc.compute_impactor_params(
        D=crater['diameter'],
        d=crater['depth'],
        U=18000,  # Assume 18 km/s
        crater_type='simple'
    )
    results.append({
        'name': crater['name'],
        'impactor_size': res['impactor_diameter'],
        'energy_Mt': res['impact_energy'] / 4.184e15
    })

results_df = pd.DataFrame(results)
print(results_df)
```

## Theory

### Pi-Group Scaling

The code implements dimensionless pi-group scaling:

**Key Parameters:**
- **π₂ = gL/U²** - Gravity parameter (ratio of gravitational to kinetic energy)
- **π₃ = Y/(ρU²)** - Strength parameter (ratio of material strength to dynamic pressure)
- **π₄ = ρ_target/ρ_impactor** - Density ratio

**Scaling Regimes:**
- **Strength regime** (π₃ > π₂): Material strength controls crater size
  - Common for: Small craters, strong materials, high velocities
  - D/L ∝ (ρU²/Y)^(μ/(2+μ))

- **Gravity regime** (π₂ > π₃): Gravity controls crater size
  - Common for: Large craters, weak materials, lower velocities
  - D/L ∝ (U²/gL)^(ν/(2+ν))

**Scaling Exponents:**
- μ ≈ 0.41 for granular materials (sand, regolith)
- μ ≈ 0.55 for competent rock
- ν ≈ 0.40 for most materials

### Crater Morphology

**Simple Craters:**
- Bowl-shaped
- Depth/diameter ratio ≈ 0.2
- Minimal collapse from transient cavity
- D_final ≈ 1.25 × D_transient

**Complex Craters:**
- Central peaks or peak rings
- Terraced walls
- Depth/diameter ratio ≈ 0.1
- Significant collapse
- D_final ≈ 1.3 × D_transient

**Transition diameter:**
- Moon: ~15-20 km
- Earth: ~2-4 km (depends on gravity and target)

## Files

| File | Description |
|------|-------------|
| `impact_scaling.py` | Main module with scaling calculations |
| `impact_mechanism_examples.ipynb` | Jupyter notebook with interactive examples |
| `requirements.txt` | Python package dependencies |
| `README_impact_scaling.md` | This file |

## Running Examples

### Command line:
```bash
python impact_scaling.py
```

This runs built-in examples showing:
1. Small fresh lunar crater (100 m)
2. Large complex crater (10 km)
3. Velocity scan for trade-off analysis

### Jupyter notebook:
```bash
jupyter notebook impact_mechanism_examples.ipynb
```

Interactive notebook includes:
- Material database exploration
- Step-by-step calculations
- Visualization of scaling regimes
- Custom calculations
- Batch processing examples

## Scientific References

1. **Holsapple, K.A. (1993).** "The scaling of impact processes in planetary sciences."
   *Annual Review of Earth and Planetary Sciences*, 21, 333-373.
   - Foundational pi-group scaling theory

2. **Holsapple, K.A., & Housen, K.R. (2007).** "A crater and its ejecta: An interpretation of Deep Impact."
   *Icarus*, 187, 345-356.
   - Updated scaling parameters

3. **Collins, G.S., Melosh, H.J., & Marcus, R.A. (2005).** "Earth Impact Effects Program."
   *Meteoritics & Planetary Science*, 40, 817-840.
   - Practical scaling implementations

4. **Melosh, H.J. (1989).** *Impact Cratering: A Geologic Process.*
   Oxford University Press.
   - Comprehensive impact mechanics textbook

5. **Schmidt, R.M., & Housen, K.R. (1987).** "Some recent advances in the scaling of impact and explosion cratering."
   *International Journal of Impact Engineering*, 5, 543-560.
   - Experimental validation of scaling laws

## Applications

### Lunar Crater Studies
- Determine impactor flux over lunar history
- Characterize recent impact events
- Estimate age from crater degradation

### Planetary Defense
- Assess damage from potential Earth impactors
- Design mitigation strategies
- Estimate energy deposition

### Mars and Other Bodies
- Interpret crater populations
- Constrain surface ages
- Understand target properties from crater morphology

### Impact Experiments
- Scale laboratory impacts to planetary scales
- Design hypervelocity experiments
- Validate numerical simulations

## Limitations and Caveats

1. **Impact angle assumed vertical**: Oblique impacts (θ < 45°) produce elongated craters
2. **Velocity uncertainty**: ±5 km/s uncertainty → factor of 2-3 in impactor size
3. **Material properties**: Strength and density can vary significantly
4. **Layered targets**: Assumes homogeneous target (real targets often layered)
5. **Transient-to-final scaling**: Simplified relationships, actual collapse is complex
6. **Fresh craters only**: Degraded/modified craters require additional correction

## Future Enhancements

Potential additions:
- [ ] Oblique impact angles
- [ ] Layered target support
- [ ] Uncertainty propagation (Monte Carlo)
- [ ] Ejecta distribution calculations
- [ ] Thermal effects and melting
- [ ] Atmospheric entry (for Earth)

## Contributing

This tool was developed for lunar impact crater analysis. Contributions welcome:
- Additional material properties
- Validation against known impacts
- Enhanced uncertainty quantification
- Additional scaling formulations

## License

Academic/research use. Please cite Holsapple (1993) and this tool in publications.

## Contact

Part of the lunar d/D ratio analysis project using KPLO Shadowcam data.

---

**Last updated:** 2025-11-21
**Version:** 1.0
**Python:** 3.8+
