# Crater Report Generator

Comprehensive Python script for generating professional crater impact analysis reports in multiple formats:
- **PDF Report**: Enhanced 14-page scientific report with full Bayesian analysis
- **IEEE LaTeX Paper**: Publication-ready LaTeX source with proper citations
- **High-Quality Figures**: All figures exported separately in PNG and PDF formats
- **Animated Quadchart**: Crater formation visualization

## Features

- **Bayesian Inverse Modeling**: Constrains projectile parameters from observed crater morphology
- **Monte Carlo Uncertainty**: Rigorous uncertainty quantification with credible intervals
- **Multiple Output Formats**: PDF, LaTeX, and individual figures
- **Simulation-Specific Values**: All numbers in reports are unique to your crater parameters
- **Professional Formatting**: IEEE-style LaTeX template ready for journal submission

## Installation

### Requirements

```bash
pip install numpy scipy matplotlib
```

Existing crater analysis modules must be in the same directory:
- `impact_backcalculator_enhanced.py`
- `lunar_impact_simulation.py`

## Usage

### Basic Usage

```bash
python crater_report_generator.py \
    --lat 15.5 \
    --lon 45.2 \
    --terrain mare \
    --diameter 350
```

### Full Parameter Specification

```bash
python crater_report_generator.py \
    --lat 15.5 \
    --lon 45.2 \
    --terrain mare \
    --diameter 350 \
    --depth 68.6 \
    --ejecta 25000 \
    --n-samples 2000 \
    --dpi 300 \
    --frames 60 \
    --output my_crater_analysis
```

### Quick Analysis (Faster)

```bash
python crater_report_generator.py \
    --lat 20.0 \
    --lon 30.0 \
    --terrain highland \
    --diameter 500 \
    --n-samples 500 \
    --dpi 150 \
    --frames 20
```

## Parameters

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--lat` | Crater latitude (degrees N) | `15.5` |
| `--lon` | Crater longitude (degrees E) | `45.2` |
| `--terrain` | Terrain type: `mare` or `highland` | `mare` |
| `--diameter` | Observed crater diameter (m) | `350` |

### Optional Crater Measurements

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--depth` | Observed crater depth (m) | Auto (0.196×D) |
| `--ejecta` | Maximum ejecta extent (m) | Auto |

### Analysis Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--n-samples` | Monte Carlo samples for uncertainty | `2000` |
| `--velocity-guess` | Initial velocity guess (km/s) | `20.0` |

### Output Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--output` | Output directory name | `crater_analysis_output` |
| `--dpi` | Figure resolution (dots per inch) | `300` |
| `--frames` | Animation frames | `60` |
| `--fps` | Animation frames per second | `15` |

## Output Files

The script generates a complete analysis package:

```
crater_analysis_output/
├── crater_analysis_report.pdf          # 14-page PDF report
├── crater_formation_quadchart.gif       # Animated visualization
├── figures/
│   ├── fig1_location_map.png/.pdf       # Orthographic crater location
│   ├── fig2_posterior_distributions.png/.pdf  # MC posterior plots
│   ├── fig3_parameter_correlations.png/.pdf   # Correlation matrix
│   ├── fig4_sensitivity_analysis.png/.pdf     # Parameter sensitivity
│   ├── fig5_crater_cross_section.png/.pdf     # Crater profile
│   ├── fig6_ejecta_distribution.png/.pdf      # Ejecta analysis
│   └── fig7_process_diagram.png/.pdf          # Workflow diagram
└── latex/
    ├── crater_analysis_paper.tex        # IEEE format LaTeX source
    ├── README.md                         # LaTeX compilation instructions
    └── fig*.pdf                          # All figures (PDF versions)
```

## Generated Report Contents

### PDF Report (14 Pages)

1. **Title & Summary**: Crater location, key findings
2. **Observations**: Detailed crater characteristics
3-4. **Theory**: Expanded Bayesian inverse problem formulation
5. **Process Diagram**: Complete workflow visualization
6. **Results**: Maximum likelihood parameters with uncertainties
7. **Monte Carlo**: Convergence analysis and credible intervals
8. **Sensitivity**: Parameter sensitivity plots
9. **Plan Views**: Orthographic projection with ejecta distribution
10. **Validation**: Forward model comparison
11. **References**: Complete bibliography
12-14. **Appendix**: Detailed methodology blocks

### LaTeX Paper (IEEE Format)

- **Abstract**: Concise summary with numerical results
- **Introduction**: Impact cratering background
- **Theory**: Scaling laws and Bayesian formulation
- **Methodology**: Study site and inversion approach
- **Results**: Parameter estimates with figures
- **Discussion**: Physical interpretation
- **Conclusions**: Key findings summary
- **References**: Properly formatted citations

All numbers (projectile diameter, velocity, angle, density, credible intervals) are **automatically filled** with simulation-specific values.

### Figures (High-Quality)

All 7 figures exported in both:
- **PNG**: High-resolution raster (DPI customizable)
- **PDF**: Vector graphics for publication

## Example Results

For a 350m crater at 15.5°N, 45.2°E (mare terrain):

**Back-Calculated Impact Parameters:**
- Projectile diameter: 3.29 ± 0.19 m
- Impact velocity: 20.0 ± 1.1 km/s
- Impact angle: 45.0° ± 6.8°
- Projectile density: 2800 ± 300 kg/m³

**95% Credible Intervals:**
- Diameter: [2.93, 3.67] m
- Velocity: [17.9, 22.2] km/s

**Kinetic Energy:** ~10 TJ (~2.4 kilotons TNT)

## LaTeX Compilation

To compile the IEEE format paper:

```bash
cd crater_analysis_output/latex
pdflatex crater_analysis_paper.tex
bibtex crater_analysis_paper
pdflatex crater_analysis_paper.tex
pdflatex crater_analysis_paper.tex
```

Output: `crater_analysis_paper.pdf`

## Methodology

### Bayesian Inversion

The script implements probabilistic parameter estimation:

1. **Forward Model**: Crater scaling laws (Holsapple 1993)
   ```
   D = K₁ × L × (ρₚ/ρₜ)^(1/3) × (v²/(g×L))^0.3 × sin(θ)^(1/3)
   ```

2. **Likelihood**: Gaussian measurement error model

3. **Priors**: Physically motivated distributions
   - Diameter: Log-normal
   - Velocity: Normal (μ=20 km/s, σ=5 km/s)
   - Angle: Uniform [30°, 90°]
   - Density: Bimodal (rocky vs. iron)

4. **Optimization**: Nelder-Mead maximum likelihood

5. **Uncertainty**: Monte Carlo sampling from posterior

### Physical Models

- **Crater Scaling**: Π-group dimensional analysis
- **Ejecta**: Maxwell Z-model with ballistic trajectories
- **Morphology**: Flow-field excavation models
- **Target Properties**: Terrain-specific (mare vs. highland)

## Performance

Typical runtimes (Intel i7, 16GB RAM):

| Configuration | N_samples | DPI | Frames | Time |
|---------------|-----------|-----|--------|------|
| Quick | 500 | 150 | 20 | ~2 min |
| Standard | 1000 | 200 | 40 | ~4 min |
| High Quality | 2000 | 300 | 60 | ~8 min |
| Publication | 5000 | 600 | 100 | ~20 min |

Most time is spent on:
- Monte Carlo sampling (30%)
- Ejecta simulations (25%)
- Animation generation (20%)
- Figure rendering (15%)
- PDF/LaTeX generation (10%)

## Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution**: Ensure all dependencies are installed and crater analysis modules are in the same directory.

### Issue: Very slow execution
**Solution**: Reduce `--n-samples`, `--frames`, or `--dpi`:
```bash
python crater_report_generator.py --lat 15.5 --lon 45.2 --terrain mare --diameter 350 --n-samples 300 --dpi 100 --frames 15
```

### Issue: LaTeX compilation fails
**Solution**:
1. Ensure LaTeX distribution is installed (TeX Live, MiKTeX, or MacTeX)
2. Install IEEEtran document class
3. Check compilation instructions in `latex/README.md`

### Issue: Out of memory
**Solution**: Reduce number of samples and animation frames.

## Scientific References

The methodology is based on:

1. **Holsapple, K. A. (1993)**: Impact crater scaling laws
   - Ann. Rev. Earth Planet. Sci., 21, 333-373

2. **Collins et al. (2005)**: Impact simulation modeling
   - Meteorit. Planet. Sci., 40(6), 817-840

3. **Pike, R. J. (1977)**: Lunar crater morphology
   - Impact and Explosion Cratering, 489-509

4. **Melosh, H. J. (1989)**: Impact Cratering: A Geologic Process
   - Oxford University Press

5. **Luo et al. (2025)**: Topography degradation models
   - JGR Planets, doi: 10.1029/2025JE008937

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{crater_report_generator,
  title = {Crater Report Generator: Bayesian Impact Parameter Inversion},
  author = {Crater Analysis Toolkit},
  year = {2025},
  version = {1.0},
  url = {https://github.com/pmahanti/documents}
}
```

## License

This software is provided for research and educational purposes.

## Version History

- **v1.0.0** (2025-11-19): Initial release
  - Bayesian inverse modeling
  - PDF and LaTeX report generation
  - High-quality figure export
  - Animated quadchart visualization
  - Comprehensive uncertainty quantification

## Contact

For questions, bug reports, or feature requests, please open an issue on GitHub.

## Acknowledgments

This tool utilizes:
- Enhanced Bayesian back-calculator (`impact_backcalculator_enhanced.py`)
- Lunar impact simulation framework (`lunar_impact_simulation.py`)
- Crater scaling laws following Holsapple (1993) and Collins et al. (2005)
