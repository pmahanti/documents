# Impact Crater Analysis Report Generator

Automated PDF report generation for impact crater analysis with comprehensive theoretical explanations, uncertainty quantification, and ejecta calculations.

## Overview

The `impact_report_generator.py` module creates professional PDF reports for impact crater analysis. Each report includes:

### Page 1: Executive Summary
- Crater location (latitude/longitude)
- Observed crater properties with uncertainties
- Computed impactor properties (diameter, mass, energy) with confidence intervals
- Material properties (target and impactor)
- Scaling regime analysis (strength vs. gravity)
- Excavation depth and ejecta volume estimates

### Pages 2+: Theoretical Methodology
- Overview of inverse scaling method
- Pi-group scaling theory with equations
- Detailed equation derivations with **substituted values**
- Regime determination explanation
- Transient-to-final crater conversion
- **Excavation depth calculations** with geometry
- **Ejecta volume and distribution** estimates
- Key assumptions and limitations
- Sources of uncertainty
- Scientific references

## Features

✓ **Uncertainty Quantification** - Monte Carlo sampling for error propagation
✓ **Excavation Depth** - Computes depth from which ejecta originates
✓ **Ejecta Volume** - Volume estimates with bulking factors
✓ **Ejecta Distribution** - Blanket thickness and range
✓ **Equation Transparency** - All formulas shown with actual values substituted
✓ **Professional Formatting** - 0.5" margins, 12pt font, tables, and structured layout
✓ **Multiple Materials** - Supports various target materials and impactor types

## Installation

```bash
# Install required packages
pip install numpy scipy matplotlib pandas reportlab

# Or use requirements file
pip install -r requirements.txt
```

## Quick Start

### Method 1: Python API

```python
from impact_report_generator import generate_impact_report

# Generate report
report = generate_impact_report(
    diameter=500,                      # meters
    depth=100,                         # meters
    velocity=15000,                    # m/s (15 km/s)
    target_material='lunar_regolith',
    impactor_material='asteroid_rock',
    latitude=-89.5,                    # degrees
    longitude=45.2,                    # degrees
    crater_name="Fresh Crater A",
    output_filename='crater_report.pdf'
)

print(f"Report saved to: {report}")
```

### Method 2: Command Line

```bash
python impact_report_generator.py \
  --diameter 500 \
  --depth 100 \
  --velocity 15000 \
  --target lunar_regolith \
  --impactor asteroid_rock \
  --latitude -89.5 \
  --longitude 45.2 \
  --crater-name "Fresh Crater A" \
  --output crater_report.pdf
```

### Method 3: Run Examples

```bash
# Generates 5 example reports
python generate_report_example.py
```

This creates:
- `lunar_crater_500m.pdf` - Simple fresh crater
- `lunar_basin_10km.pdf` - Large complex crater
- `iron_meteorite_crater.pdf` - Metallic impactor
- `polar_ice_crater.pdf` - Ice-rich target
- `shadowcam_crater_report.pdf` - Template for your data

## Parameters

### Required Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `diameter` | float | Crater diameter (m) | 500.0 |
| `depth` | float | Crater depth (m) | 100.0 |
| `velocity` | float | Impact velocity (m/s) | 15000 |
| `target_material` | str | Target material name | 'lunar_regolith' |
| `impactor_material` | str | Impactor type | 'asteroid_rock' |
| `latitude` | float | Latitude (decimal degrees) | -89.5 |
| `longitude` | float | Longitude (decimal degrees) | 45.2 |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `diameter_uncertainty` | float | 5% of diameter | Uncertainty in diameter (m) |
| `depth_uncertainty` | float | 10% of depth | Uncertainty in depth (m) |
| `velocity_uncertainty` | float | 2000 | Uncertainty in velocity (m/s) |
| `crater_type` | str | 'simple' | 'simple' or 'complex' |
| `crater_name` | str | None | Optional crater name |
| `output_filename` | str | 'impact_analysis_report.pdf' | Output file path |

## Available Materials

### Target Materials

Use these values for the `target_material` parameter:

| Material ID | Description | Density (kg/m³) | Strength (Pa) | Body |
|------------|-------------|-----------------|---------------|------|
| `lunar_regolith` | Loose lunar soil | 1500 | 1×10³ | Moon |
| `lunar_mare` | Mare basalt | 3100 | 1×10⁷ | Moon |
| `lunar_highland` | Highland anorthosite | 2800 | 5×10⁶ | Moon |
| `sandstone` | Sedimentary rock | 2200 | 3×10⁷ | Earth |
| `granite` | Igneous rock | 2750 | 2×10⁸ | Earth |
| `limestone` | Sedimentary rock | 2600 | 5×10⁷ | Earth |
| `sand` | Loose sand | 1650 | 1×10³ | Earth |
| `dry_soil` | Dry terrestrial soil | 1600 | 1×10⁴ | Earth |
| `wet_soil` | Wet terrestrial soil | 1800 | 5×10⁴ | Earth |
| `water_ice` | Pure water ice | 920 | 1×10⁶ | Icy moons |
| `ice_regolith_mix` | Ice-regolith mixture | 1200 | 5×10⁵ | Polar regions |

### Impactor Types

Use these values for the `impactor_material` parameter:

| Impactor ID | Description | Density (kg/m³) |
|------------|-------------|-----------------|
| `asteroid_rock` | Stony asteroid | 2700 |
| `asteroid_metal` | Iron meteorite | 7800 |
| `comet_ice` | Porous comet | 500 |
| `comet_ice_dense` | Dense comet nucleus | 1000 |

## Example Use Cases

### 1. Analyze Your Lunar Crater Data

```python
import pandas as pd
from impact_report_generator import generate_impact_report

# Load your crater measurements
craters = pd.read_csv('my_crater_measurements.csv')

# Generate reports for each crater
for idx, crater in craters.iterrows():
    generate_impact_report(
        diameter=crater['diameter_m'],
        depth=crater['depth_m'],
        velocity=16000,  # Typical lunar impact
        target_material='lunar_regolith',
        impactor_material='asteroid_rock',
        latitude=crater['lat'],
        longitude=crater['lon'],
        crater_name=f"Crater_{crater['id']}",
        output_filename=f"reports/crater_{crater['id']}.pdf"
    )
```

### 2. Compare Different Impact Velocities

```python
# Same crater, different assumed velocities
velocities = [12000, 15000, 18000, 21000]

for vel in velocities:
    generate_impact_report(
        diameter=1000,
        depth=200,
        velocity=vel,
        target_material='lunar_mare',
        impactor_material='asteroid_rock',
        latitude=-88.0,
        longitude=120.0,
        output_filename=f'crater_velocity_{vel//1000}kms.pdf'
    )
```

### 3. Analyze Polar Craters

```python
# Polar crater that may have impacted ice
generate_impact_report(
    diameter=750,
    depth=150,
    velocity=15000,
    target_material='ice_regolith_mix',  # Ice-rich target
    impactor_material='asteroid_rock',
    latitude=-89.9,  # Near pole
    longitude=0.0,
    diameter_uncertainty=30,
    depth_uncertainty=15,
    crater_name="Polar PSR Crater",
    output_filename='polar_psr_analysis.pdf'
)
```

## Report Contents Detail

### Page 1 Summary

**Crater Location Section**
- Latitude and longitude in decimal degrees
- Coordinate system specification

**Observed Properties Table**
- Crater diameter (with uncertainty)
- Crater depth (with uncertainty)
- Depth/diameter ratio
- Crater type (simple/complex)
- Assumed impact velocity (with uncertainty)

**Computed Impactor Properties Table**
- Impactor diameter (with 16th-84th percentile range)
- Impactor mass
- Impact energy (in Joules and Mt TNT equivalent)
- Impact momentum
- Includes uncertainty ranges from Monte Carlo analysis

**Material Properties Table**
- Target material name, density, strength, surface gravity
- Impactor type and density

**Scaling Regime Analysis**
- Identifies strength vs. gravity regime
- Shows π₂, π₃, π₄ parameter values
- Explains which factor controls crater size

**Excavation and Ejecta Table**
- Excavation depth (average and maximum)
- Excavated volume
- Ejecta volume (with bulking factor)
- Continuous ejecta range
- Ejecta thickness (average and at rim)

### Pages 2+ Theory

**Section 1: Overview**
- Explanation of inverse scaling approach
- Relationship between crater and impactor

**Section 2: Pi-Group Scaling Theory**
- Definition of π₂ (gravity parameter)
- Definition of π₃ (strength parameter)
- Definition of π₄ (density ratio)
- **Actual numerical values substituted** into equations

**Section 3: Regime Determination**
- Comparison of π₂ and π₃
- Identification of dominant regime
- Physical interpretation

**Section 4: Scaling Law Application**
- Appropriate scaling equation (strength or gravity regime)
- **Step-by-step calculation with actual numbers**
- Exponent calculations
- Final crater diameter prediction

**Section 5: Transient to Final Crater**
- Modification process explanation
- Expansion factors for simple vs. complex craters
- Transient vs. final diameter calculations

**Section 6: Excavation Depth Calculations**
- Definition of excavation depth
- Empirical relations used
- **Computed values with formulas**
- Radial variation of excavation depth

**Section 7: Ejecta Volume and Distribution**
- Excavation volume calculation
- Bulking factor application
- Ejecta blanket extent
- Thickness distribution
- **Power-law decrease with distance**

**Section 8: Assumptions and Limitations**
- Vertical impact assumption
- Velocity assumptions
- Homogeneous target assumption
- Fresh crater assumption
- Material property assumptions
- Scaling law validity ranges

**Section 9: Uncertainty Sources**
- Measurement uncertainties
- Physical uncertainties
- Model uncertainties
- Monte Carlo propagation method

**Section 10: References**
- Key scientific papers cited

## Technical Details

### Uncertainty Quantification

The module performs **Monte Carlo sampling** (default: 1000 samples) to propagate uncertainties:

1. Samples diameter, depth, and velocity from normal distributions
2. Computes impactor parameters for each sample
3. Calculates mean, standard deviation, and percentiles
4. Reports 16th-84th percentile range (±1σ equivalent)

### Excavation Depth Formula

```
d_excavation = 0.10 × D_final    (simple craters)
d_excavation = 0.08 × D_final    (complex craters)

d_excavation_max = 1.5 × d_excavation (at center)

V_excavation = (π/8) × D² × d_excavation
```

### Ejecta Volume Calculation

```
V_ejecta = f_bulk × V_excavation

where f_bulk ≈ 1.2 (bulking factor for fractured material)
```

### Ejecta Thickness Distribution

```
Continuous ejecta extends to r ≈ 1.5 × D

Thickness decreases as: t(r) ∝ r^(-3)

Average thickness: t_avg = V_ejecta / A_blanket

At rim: t_rim ≈ 0.14 × d_crater
```

## Customization

### Adding Custom Materials

You can add custom materials to `impact_scaling.py`:

```python
from impact_scaling import Material

custom_material = Material(
    name='My Custom Material',
    density=2000,      # kg/m³
    strength=1e6,      # Pa
    gravity=1.62,      # m/s²
    K1=0.132,
    mu=0.45,           # Scaling exponent
    nu=0.40,
)

# Then use in report
generate_impact_report(
    target_material='my_custom_material',
    # ... other parameters
)
```

### Modifying Report Style

Edit `_setup_custom_styles()` in `impact_report_generator.py` to change:
- Font sizes
- Colors
- Spacing
- Margins (in `SimpleDocTemplate` call)

## Integration with Your Workflow

### Example: Batch Processing from Excel

```python
import pandas as pd
from impact_report_generator import generate_impact_report

# Load your measurements from fnnames_sslatlon_time.xlsx
df = pd.read_excel('fnnames_sslatlon_time.xlsx')

# Load crater measurements (you'd have this from DEM analysis)
craters = pd.read_csv('crater_measurements.csv')

# Merge on image_name or other key
merged = df.merge(craters, on='image_name')

# Generate reports
for idx, row in merged.iterrows():
    try:
        generate_impact_report(
            diameter=row['crater_diameter_m'],
            depth=row['crater_depth_m'],
            velocity=16000,  # Typical
            target_material='lunar_regolith',
            impactor_material='asteroid_rock',
            latitude=row['sslat'],
            longitude=row['sslon'],
            crater_name=row['image_name'],
            output_filename=f"reports/{row['image_name']}_report.pdf"
        )
        print(f"✓ Generated report for {row['image_name']}")
    except Exception as e:
        print(f"✗ Failed for {row['image_name']}: {e}")
```

## Troubleshooting

### Common Issues

**"KeyError: style already defined"**
- Fixed in current version - styles are now updated if they exist

**"Optimization did not converge"**
- Occurs with extreme parameter combinations
- Try adjusting velocity or crater size
- Check that crater type ('simple' vs 'complex') is appropriate

**"ReportLab not installed"**
```bash
pip install reportlab
```

**Small impactor diameter (< 0.1 m)**
- May indicate incompatible parameters
- Check velocity (should be 5-50 km/s)
- Verify crater measurements are in meters

**Large uncertainty ranges**
- Common when velocity uncertainty is high
- Reduce `velocity_uncertainty` if velocity is better constrained
- Consider constraining impactor type (metal vs. rock makes big difference)

## Output Files

Reports are saved as PDF with:
- **Page size**: US Letter (8.5" × 11")
- **Margins**: 0.5" on all sides
- **Font**: Helvetica (body), Helvetica-Bold (headings), Courier (equations)
- **Font size**: 12pt (body), 14pt (section headings), 16pt (title)
- **File size**: Typically 10-20 KB

## Performance

- Single report generation: ~2-5 seconds
- Includes Monte Carlo sampling (1000 iterations)
- Batch processing: ~3 seconds per report

## Version History

**v1.0** (2025-11-21)
- Initial release
- Comprehensive report generation
- Monte Carlo uncertainty quantification
- Excavation and ejecta calculations
- Full equation transparency with substituted values

## Contributing

Suggestions for improvements:
- Additional material properties
- Alternative uncertainty methods
- Enhanced visualizations
- Multi-language support
- LaTeX equation rendering

## References

The theoretical foundations come from:

1. **Holsapple, K.A. (1993)** - Pi-group scaling theory
2. **Melosh, H.J. (1989)** - Impact cratering mechanics
3. **Collins et al. (2005)** - Practical implementations
4. **Schmidt & Housen (1987)** - Experimental validation

## License

Academic/research use. Cite Holsapple (1993) and this tool in publications.

## Contact

Part of the KPLO Shadowcam lunar crater d/D analysis project.

---

**Created**: 2025-11-21
**Python**: 3.8+
**Dependencies**: numpy, scipy, reportlab
