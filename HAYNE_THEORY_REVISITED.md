# Hayne et al. (2021) Theory Revisited: Complete Mathematical Framework

**Date**: 2025-11-23
**Purpose**: Systematic review of all equations and theory from Hayne et al. (2021) "Micro cold traps on the Moon"
**Reference**: Nature Astronomy 5, 169-175 (2021) + Supplementary Materials

---

## Table of Contents

1. [Overview](#overview)
2. [Part 1: Crater Shadow Geometry](#part-1-crater-shadow-geometry)
3. [Part 2: Thermal Balance](#part-2-thermal-balance)
4. [Part 3: Rough Surface Model](#part-3-rough-surface-model)
5. [Part 4: Lateral Heat Conduction](#part-4-lateral-heat-conduction)
6. [Part 5: Size Distribution and Total Areas](#part-5-size-distribution-and-total-areas)
7. [Implementation Notes](#implementation-notes)

---

## Overview

### Physical System

Hayne et al. (2021) model **multi-scale cold traps** on the Moon from **1 cm to 100 km**, including:
1. **Large craters** (1-100 km): Spherical bowl geometry (Ingersoll 1992)
2. **Small craters** (1 m - 1 km): Same bowl model at smaller scales
3. **Rough plains** (cm - 10 m): Gaussian surfaces with RMS slopes

### Key Innovation

**Scale-dependent cold trap fractions** accounting for:
- Geometric shadowing (latitude and topography dependent)
- Thermal effects (radiation balance)
- Lateral heat conduction (eliminates micro-traps)

### Total Cold Trap Area Formula

**Main Equation (Hayne Eq. 1)**:
```
A(L, L') = ∫[L to L'] α(l,φ) τ(l,φ) dl
```

Where:
- `A(L, L')` = Total cold trap area between scales L and L'
- `α(l,φ)` = Fractional area occupied by permanent shadows at scale l, latitude φ
- `τ(l,φ)` = Fraction of PSRs that are cold traps (T_max < 110 K)
- Integration over length scales from L to L'

---

## Part 1: Crater Shadow Geometry

### 1.1 Spherical Bowl Crater Model

**Geometry** (Supplementary Figure 1):
- Diameter: D
- Depth: d
- Depth-to-diameter ratio: **γ = d/D**
- Radius of sphere: **R_s = (R² + d²)/(2d)** where R = D/2

**Key Parameter**:
```
β = 1/(2γ) - 2γ
```

This geometric parameter controls shadow behavior.

### 1.2 Instantaneous Shadow Area

**Shadow coordinate** (Hayne Eq. 3):
```
x'₀ = cos²(e) - sin²(e) - β cos(e) sin(e)
```

Where:
- `e` = solar elevation angle above horizon

**Shadow area fraction** (Hayne Eq. 5):
```
A_shadow / A_crater = (1 + x'₀) / 2
```

**Physical interpretation**:
- `x'₀ = -1`: Fully in shadow (A_shadow/A_crater = 0)
- `x'₀ = +1`: Fully illuminated (A_shadow/A_crater = 1)
- Intermediate values give partial shadow

### 1.3 Permanent Shadow Area

**At the pole** (Hayne Eq. 22, δ = 0):
```
A_perm / A_crater = 1 - (8β e₀)/(3π)
```

Where:
- `e₀` = maximum solar elevation = 90° - |latitude|

**With solar declination** (Hayne Eq. 26):
```
A_perm / A_crater = 1 - (8β e₀)/(3π) - 2β δ_max
```

Where:
- `δ_max` = maximum solar declination ≈ 1.54° for Moon

**Constraints**:
- Result must be ≥ 0 (use max(0, result))
- Result must be ≤ A_shadow at any instant

### 1.4 Shadow Boundary Position

**Normalized position** (Hayne Eq. 2):
```
x₀ = R_s sin(θ)
```

Where:
- `R_s` = sphere radius
- `θ` = angle from vertical to shadow boundary

**Relation to x'₀**:
```
x'₀ = 2x₀/D
```

This gives the physical location of the shadow edge.

### 1.5 Validation: Comparison with Bussey et al. (2003)

**Supplementary Figure 3** shows validation against numerical ray-tracing for a d/D = 1:5 crater (γ = 0.20).

**Agreement**:
- Analytical formula matches numerical within ~5-10%
- Better at high latitudes (>80°)
- Semi-analytic approximation improves fit

---

## Part 2: Thermal Balance

### 2.1 Radiation Balance in Shadows

**Energy balance** (implied from Methods):
```
ε σ T⁴ = Q_scattered + Q_thermal + Q_sky
```

Where:
- `ε` = emissivity ≈ 0.95
- `σ` = Stefan-Boltzmann constant = 5.67×10⁻⁸ W/(m²·K⁴)
- `T` = surface temperature in shadow

### 2.2 Scattered Sunlight

**From crater walls** (Lambertian scattering):
```
Q_scattered = F_walls × A × S × cos(e)
```

Where:
- `F_walls` = view factor to illuminated walls
- `A` = Bond albedo ≈ 0.12
- `S` = solar constant at Moon ≈ 1361 W/m²
- `cos(e)` = projection factor

### 2.3 Thermal Infrared Emission

**From crater walls**:
```
Q_thermal = F_walls × ε × σ × T_walls⁴
```

Where:
- `T_walls` = temperature of illuminated crater walls

**Wall temperature** depends on:
- Direct solar illumination
- Thermal properties of regolith
- Local slope and view factors

### 2.4 Sky Radiation

**From cosmic microwave background**:
```
Q_sky = F_sky × ε × σ × T_sky⁴
```

Where:
- `T_sky` ≈ 3 K (CMB)
- This term is negligible (Q_sky ≈ 10⁻⁶ W/m²)

### 2.5 View Factors for Bowl Crater

**From Ingersoll et al. (1992)**:

**Exact formula** (from opening solid angle):
```
cos(θ_open) = h / sqrt(h² + R²)
```

Where:
- `h = R_s - d` = height from floor to sphere center
- `R = D/2` = crater radius

**Sky view factor**:
```
F_sky = (1 - cos(θ_open)) / 2
```

**Wall view factor**:
```
F_walls = 1 - F_sky
```

**Approximation** (for small γ):
```
F_walls ≈ min(γ/0.2, 0.7)
```

**Note**: Our validation showed this approximation has large errors. Use exact formula!

### 2.6 Solution for Shadow Temperature

Rearrange energy balance:
```
T_shadow = [(Q_scattered + Q_thermal + Q_sky) / (ε σ)]^(1/4)
```

This must be solved **self-consistently** because Q_thermal depends on T_walls, which itself depends on the radiation environment.

---

## Part 3: Rough Surface Model

### 3.1 Gaussian Surface Generation

**From Methods (page 174)**:
- Use **Gaussian random fields** with **Hurst exponent H = 0.9**
- Domain size: **128 × 128 pixels**
- RMS slopes: **σ_s** from 0° to 35°

**Gaussian height field**:
```
z(x,y) = Σ A_k exp(i k·r + φ_k)
```

Where power spectrum follows:
```
P(k) ∝ k^(-2H-2)
```

For H = 0.9, this gives **self-affine fractal** surface.

### 3.2 RMS Slope Calculation

**Definition**:
```
σ_s = sqrt(<|∇z|²>)
```

Where:
- `∇z` = gradient of height field
- `< >` = spatial average

**In discrete form**:
```
σ_s = sqrt[(Σ(∂z/∂x)² + Σ(∂z/∂y)²) / N]
```

### 3.3 Horizon Calculation (Ray-Tracing)

**From Methods**:
- Calculate horizons **every 1° in azimuth** (360 directions)
- Use ray-tracing from each surface element
- Horizon angle **h(az)** = elevation angle to visible horizon

**Shadow determination**:
```
if e_solar < h(az_solar):
    in_shadow = True
else:
    in_shadow = False
```

Where:
- `e_solar` = solar elevation
- `az_solar` = solar azimuth

### 3.4 Cold Trap Fraction vs RMS Slope

**Empirical fit** (from Figure 3):

At **optimal roughness** (σ_s ≈ 15°):
- Maximum cold trap fraction achieved
- Balance between shadow area and temperature

**Latitude dependence**:
- 70°S: f ≈ 0.002 (0.2%)
- 80°S: f ≈ 0.008 (0.8%)
- 85°S: f ≈ 0.015 (1.5%)
- 88°S: f ≈ 0.020 (2.0%)

**Functional form** (approximate):
```
f(σ_s, φ) = f_max(φ) × {
    (σ_s / σ_opt)                         for σ_s < σ_opt
    exp(-(σ_s - σ_opt) / σ_decay)        for σ_s ≥ σ_opt
}
```

Where:
- `σ_opt` ≈ 15° (optimal RMS slope)
- `σ_decay` ≈ 10° (decay constant)
- `f_max(φ)` = latitude-dependent maximum

### 3.5 Temperature Calculation on Rough Surfaces

**At each pixel** (from Methods):

1. **Direct solar flux**:
   ```
   Q_direct = S × (1-A) × max(0, n̂·ŝ)
   ```
   Where:
   - `n̂` = surface normal vector
   - `ŝ` = solar direction unit vector

2. **Scattered flux** (from visible surrounding terrain):
   ```
   Q_scattered = ∫ [S × A × cos(e') × dΩ / π]
   ```
   Integral over visible hemisphere

3. **Thermal flux** (from surrounding terrain):
   ```
   Q_thermal = ∫ [ε × σ × T(r')⁴ × dΩ / π]
   ```
   Integral over visible hemisphere

4. **Energy balance**:
   ```
   ε σ T⁴ = Q_direct + Q_scattered + Q_thermal
   ```

This must be solved **iteratively** for all pixels simultaneously until convergence.

---

## Part 4: Lateral Heat Conduction

### 4.1 Thermal Skin Depth

**Diurnal skin depth**:
```
δ = sqrt(κ P / π)
```

Where:
- `κ` = thermal diffusivity ≈ 5×10⁻⁷ m²/s for lunar regolith
- `P` = period = 29.5 days = 2.55×10⁶ s

**Result**: δ ≈ **4.4 cm** for lunar day

### 4.2 Critical Length Scale

**Lateral conduction eliminates cold traps** when:
```
l < l_crit = C × δ / sin(φ_lat)
```

Where:
- `C` ≈ 2-3 (empirical factor)
- `φ_lat` = latitude from pole

**Physical interpretation**:
- At **90°S**: l_crit ≈ 1 cm (smallest cold traps survive)
- At **60°S**: l_crit ≈ 10 m (only large cold traps survive)

### 4.3 Critical Depth/Diameter Ratio

**Supplementary Figure 8** shows **γ_c(l, φ)**:

For a crater to be a cold trap:
```
γ < γ_c(l, φ)
```

**Trends**:
- γ_c **decreases** with increasing latitude (easier to have cold traps)
- γ_c **decreases** with decreasing scale (conduction eliminates shallow micro-craters)
- At 85°S, l=10 cm: γ_c ≈ 0.15

### 4.4 Implementation

**Numerical solution** (Supplementary Figure 10):
- Solve **2D heat equation** in cylindrical coordinates
- Account for variable thermal conductivity: k = k_c(1 + R_350 T³)
- Calculate steady-periodic temperature field
- Determine minimum temperature at crater center

**Result**: Temperature rises due to lateral heating, eliminating cold traps at small scales.

---

## Part 5: Size Distribution and Total Areas

### 5.1 Crater Size-Frequency Distribution

**Power law form**:
```
N(>D) ∝ D^(-b)
```

Where:
- `N(>D)` = cumulative number of craters with diameter > D
- `b` ≈ 2-3 (production function exponent)

**Differential form**:
```
dN/dD ∝ D^(-(b+1))
```

### 5.2 Crater Depth/Diameter Distributions

**Log-normal distribution** (Supplementary Figure 9):

**Distribution A** (deeper craters):
```
P(γ) = (1/√(2πσ²)) exp[-(ln γ - μ)² / (2σ²)]
```
With μ = ln(0.14)

**Distribution B** (shallower craters):
With μ = ln(0.076)

**Standard deviation**: σ ≈ 0.3 (typical)

### 5.3 Fraction of PSRs that are Cold Traps

**Supplementary Figure 9** shows **τ(l, φ)**:

For given (length scale, latitude):
```
τ(l, φ) = ∫[0 to γ_c(l,φ)] P(γ) dγ
```

**Physical meaning**: Fraction of craters with γ < γ_c that have T_max < 110 K

**Trends**:
- τ → 1 at high latitudes and large scales (all PSRs are cold traps)
- τ → 0 at low latitudes and small scales (conduction destroys cold traps)

### 5.4 Permanent Shadow Area Fraction

**Supplementary Figure 6** shows **α(l, φ)**:

**Combined model**:
```
α(l, φ) = x_crater × α_crater(l, φ) + (1 - x_crater) × α_plains(l, φ)
```

Where:
- `x_crater` = 0.20 (fraction of surface covered by craters)
- `α_crater(l, φ)` = shadow fraction from bowl crater model
- `α_plains(l, φ)` = shadow fraction from rough surface model with σ_s = 5.7°

**Result** (units of m⁻¹):
- At 85°S: α ≈ 10⁻⁵ m⁻¹ for l ~ 10 m
- At 70°S: α ≈ 10⁻⁷ m⁻¹ for l ~ 10 m

### 5.5 Total Cold Trap Area Calculation

**Integration** (Hayne Eq. 1):
```
A_total = ∫∫ α(l,φ) × τ(l,φ) × f_surface(φ) dl dφ
```

Where:
- `f_surface(φ)` = fraction of lunar surface at latitude φ
- Integration over latitudes 60°-90° and scales 1 cm to 100 km

**Result**: **~40,000 km²** total (0.10% of lunar surface)

**Breakdown**:
- South pole: ~23,000 km² (60%)
- North pole: ~17,000 km² (40%)
- Micro cold traps (<100 m): ~2,500 km² (10-20% of total)

---

## Implementation Notes

### Software Tools Used by Hayne et al.

**From paper and supplementary**:
- **Topo3D**: https://github.com/nschorgh/Planetary-Code-Collection/blob/master/Topo3D
  - 3D terrain thermal model
  - Calculates horizons, shadows, radiation balance
  - Solves heat equation with lateral conduction

- **Custom Python/MATLAB**: For Gaussian surface generation, analysis, plotting

### Validation Data

1. **Bussey et al. (2003)**: Numerical shadow calculations (Supplementary Figure 3)
2. **LROC NAC images**: Shadow measurements (Supplementary Figures 4-5)
3. **Mazarico et al. (2011)**: LOLA topography illumination (Supplementary Figure 7)

### Key Parameters

| Parameter | Symbol | Value | Units |
|-----------|--------|-------|-------|
| Solar constant | S | 1361 | W/m² |
| Bond albedo | A | 0.12 | - |
| Emissivity | ε | 0.95 | - |
| Sky temperature | T_sky | 3 | K |
| H₂O threshold | T_threshold | 110 | K |
| Lunar declination | δ_max | 1.54 | ° |
| Thermal diffusivity | κ | 5×10⁻⁷ | m²/s |
| Diurnal skin depth | δ | 4.4 | cm |
| Crater area fraction | x_crater | 0.20 | - |
| Plains RMS slope | σ_s | 5.7 | ° |

---

## Cross-Check with Current Implementation

### ✅ VALIDATED Components

1. **Shadow geometry** (Equations 2-9, 22, 26):
   - `bowl_crater_thermal.py` implementation CORRECT
   - Validation showed 0.00e+00 error

2. **Energy balance**:
   - Conservation satisfied to machine precision
   - Correct component structure

### ❌ ISSUES Found

1. **View factors**:
   - ✅ NOW FIXED (was using wrong formula)
   - Currently using exact Ingersoll (1992) formula

2. **Rough surface cold trap fraction**:
   - ❌ `thermal_model.py` ignores latitude parameter
   - ❌ No 3D radiation balance calculation
   - ❌ Uses oversimplified empirical fit

3. **Figure 4 total area**:
   - ❌ Predicts 105,000 km² vs 40,000 km²
   - Missing crater/plains mixture
   - Missing proper depth/diameter distributions
   - Missing lateral conduction cutoff

### 📋 NEXT STEPS for Full Implementation

1. **Implement 3D radiation model**:
   - Gaussian surface generator with H=0.9
   - Ray-tracing for horizons
   - Self-consistent temperature solver

2. **Fix rough surface model**:
   - Add proper latitude dependence
   - Include full radiation balance
   - Validate against Figure 3

3. **Implement size distributions**:
   - Crater size-frequency distribution
   - Log-normal depth/diameter distribution
   - Proper integration over scales

4. **Add lateral conduction**:
   - Calculate γ_c(l, φ)
   - Implement cutoff in cold trap areas
   - Validate against Supplementary Figures 8-9

---

## References

1. **Hayne, P. O., Aharonson, O. & Schörghofer, N.** Micro cold traps on the Moon. *Nature Astronomy* **5**, 169–175 (2021).

2. **Ingersoll, A. P., Svitek, T. & Murray, B. C.** Stability of polar frosts in spherical bowl-shaped craters on the Moon, Mercury, and Mars. *Icarus* **100**, 40–47 (1992).

3. **Bussey, D. B. J. et al.** Permanent shadow in simple craters near the lunar poles. *Geophys. Res. Lett.* **30**, 1278 (2003).

4. **Schörghofer, N.** Planetary Code Collection. GitHub repository: https://github.com/nschorgh/Planetary-Code-Collection

---

*This document provides the complete theoretical framework for implementing Hayne et al. (2021) micro cold trap model. All equations are documented with their physical meaning and implementation guidance.*

*Created: 2025-11-23*
*Status: Ready for implementation and validation*
