# Critical Issues in Hayne et al. (2021) Validation

## Date: 2025-11-23
## Status: VALIDATION FAILED - Multiple Critical Bugs Found

---

## Executive Summary

The current microPSR implementation has **critical bugs** that prevent accurate replication of Hayne et al. (2021) results:

1. ❌ **Figure 3**: Cold trap fraction shows NO latitude dependence (all curves identical)
2. ❌ **Figure 4**: Total area is ~105,000 km² vs Hayne's ~40,000 km² (2.6× too high)
3. ❌ **Table 1**: Values don't match published results
4. ❌ **Core Bug**: `rough_surface_cold_trap_fraction()` ignores latitude parameter

---

## Detailed Analysis of Bugs

### BUG #1: Missing Latitude Dependence in Cold Trap Fraction

**Location**: `thermal_model.py`, line 210-271

**Current Code**:
```python
def rough_surface_cold_trap_fraction(rms_slope_deg, latitude_deg, ...):
    ...
    if model == 'hayne2021':
        sigma_optimal = 15.0
        if sigma_s < sigma_optimal:
            frac = 0.02 * sigma_s / sigma_optimal  # ← No latitude_deg used!
        else:
            frac = 0.02 * np.exp(-(sigma_s - sigma_optimal) / 10.0)  # ← No latitude_deg used!
```

**Problem**:
- The function accepts `latitude_deg` parameter but **never uses it**
- Result: All latitudes (70°S, 75°S, 80°S, 85°S, 88°S) produce identical curves
- This is **fundamentally incorrect** physics

**What Hayne Actually Did**:
From the paper (Methods section):
1. Used 3D topography model with Gaussian rough surfaces
2. Calculated **direct illumination** based on solar elevation (which varies with latitude)
3. Calculated **infrared emission** from surrounding terrain
4. Calculated **scattered visible light**
5. Solved **radiation balance** to get temperature
6. Determined which shadows are < 110K (cold traps)

**Expected Behavior** (from Hayne Fig. 3):
- At 70°S, σs=15°: f ≈ 0.2% (0.002)
- At 80°S, σs=15°: f ≈ 0.8% (0.008)
- At 85°S, σs=15°: f ≈ 1.5% (0.015)
- At 88°S, σs=15°: f ≈ 2.0% (0.020)

**Actual Output** (current bug):
- At ALL latitudes, σs=15°: f ≈ 2.0% (0.020)

**Impact**:
- Cannot validate against Hayne's work
- Predictions at lower latitudes are severely wrong (10× too high at 70°S)
- Ice stability estimates are incorrect

---

### BUG #2: Oversized Cold Trap Area Estimates

**Location**: Figure 4 generation

**Current Output**:
- Total cold trap area: **105,257 km²** (0.278% of lunar surface)
- Northern hemisphere: 42,103 km²
- Southern hemisphere: 63,154 km²

**Hayne et al. (2021) Values**:
- Total cold trap area: **~40,000 km²** (0.10% of lunar surface)
- Southern: ~23,000 km²
- Northern: ~17,000 km²

**Discrepancy**:
- Current model predicts **2.6× more cold trap area** than Hayne
- This propagates from Bug #1: overestimating cold trap fractions at lower latitudes

---

### BUG #3: Ingersoll Bowl Model Implementation

**Need to Verify**:
1. Shadow area fractions (Hayne Eqs. 2-9)
2. View factors (Ingersoll 1992)
3. Radiation balance equations
4. Temperature calculations
5. Lateral conduction treatment

**Status**: NOT YET VALIDATED

The Ingersoll model must be validated equation-by-equation against:
- Ingersoll et al. (1992) original paper
- Hayne et al. (2021) Methods section (Eqs. 2-9, 22-27)

---

## What Hayne et al. (2021) Actually Did

### For Figure 2: Synthetic Rough Surface Temperatures

**Methodology** (from Methods, page 174):
1. Created Gaussian surfaces with specific σs values (5.7°, 26.6°)
2. Used 128×128 pixel domains
3. Calculated horizons using ray-tracing (every 1° in azimuth)
4. Calculated **direct solar flux** at each surface element
5. Calculated **scattered flux** (Lambertian scattering, albedo = 0.12)
6. Calculated **infrared flux** (emissivity = 0.95)
7. Solved for **equilibrium surface temperatures**
8. Displayed as spatial temperature maps

**Key Parameters**:
- Latitude: 85°S
- Solar declination: 0° and 1.5°
- Hurst exponent: 0.9
- Albedo: 0.12
- Emissivity: 0.95

### For Figure 3: Cold Trap Fraction vs RMS Slope

**Methodology**:
1. Used same 3D model as Figure 2
2. Varied σs from 0° to 35°
3. Varied latitude from 70°S to 90°S
4. For each combination:
   - Calculated temperatures over full lunar day
   - Identified pixels with T_max < 110K
   - Calculated fractional area
5. Accounted for lateral heat conduction (eliminates cold traps < 1 cm)

**Critical**: This requires **full thermal model**, not just geometric shadowing!

### For Figure 4: Size Distribution

**Methodology**:
1. Combined crater model (~20% by area) and rough plains (~80% by area)
2. Crater size distribution from crater surveys
3. For craters: used Ingersoll bowl model with depth/diameter distributions
4. For plains: used rough surface model with σs ≈ 5.7°
5. Integrated over all length scales from 1 cm to 100 km
6. Accounted for lateral conduction at small scales

**Key Equation** (Hayne Eq. 1):
```
A(L, L') = ∫[L to L'] α(l,φ) τ(l,φ) dl
```
where:
- α(l,φ) = fractional surface area occupied by permanent shadows
- τ(l,φ) = fraction of PSRs that are cold traps (T < 110K)

### For Table 1: Total Areas

**Methodology**:
1. Integrated Figure 4 results over all length scales
2. Separated by latitude bands
3. Compared with Watson et al. (1961) classical analysis
4. Reported as percentage of surface area

---

## Correct Implementation Plan

### Phase 1: Fix Core Thermal Model ✓ PRIORITY

1. **Implement proper radiation balance for rough surfaces**:
   ```
   εσT⁴ = Q_direct + Q_scattered + Q_thermal + Q_sky
   ```
   where each component depends on:
   - Solar elevation (latitude-dependent)
   - View factors (geometry-dependent)
   - Surface temperatures (self-consistent)

2. **Fix `rough_surface_cold_trap_fraction()` to use latitude**:
   - Calculate maximum solar elevation from latitude
   - Determine shadow fractions (geometric)
   - Calculate shadow temperatures (radiation balance)
   - Determine which shadows are < 110K

3. **Validate Ingersoll bowl model equations**:
   - Check Hayne Eqs. 2-9 implementation
   - Verify view factor calculations
   - Test against Ingersoll (1992) analytical solutions

### Phase 2: Replicate Hayne Figures

1. **Figure 2**: Synthetic rough surface temperature maps
   - Generate Gaussian surfaces
   - Implement full 3D radiation model
   - Produce side-by-side comparison with Hayne Fig. 2

2. **Figure 3**: Cold trap fraction vs RMS slope
   - Use corrected thermal model
   - Ensure proper latitude dependence
   - Match Hayne's curves exactly

3. **Figure 4**: PSR and cold trap size distributions
   - Implement proper crater/plains mixture
   - Use correct depth/diameter distributions
   - Match total area (~40,000 km²)

4. **Table 1**: Total cold trap areas
   - Integrate Figure 4 results
   - Match Hayne's values

### Phase 3: Documentation

1. Create validation report showing:
   - Side-by-side figure comparisons
   - Quantitative agreement metrics
   - Remaining uncertainties

2. Document all equations used
3. Provide references to specific equations in Hayne paper

---

## Key Equations from Hayne et al. (2021)

### Shadow Area in Bowl Crater

**Instantaneous shadow** (Eq. 5):
```
A_shadow / A_crater = (1 + x₀') / 2
```
where (Eq. 3):
```
x₀' = cos²(e) - sin²(e) - β cos(e) sin(e)
```
and:
```
β = 1/(2γ) - 2γ
γ = d/D  (depth-to-diameter ratio)
```

**Permanent shadow** (Eq. 22 with Eq. 26 correction):
```
A_perm / A_crater ≈ 1 - (8β e₀)/(3π) - 2β δ_max
```
where:
- e₀ = colatitude = π/2 - φ  (φ = latitude)
- δ_max = maximum solar declination ≈ 1.54° for Moon

### Thermal Balance

From Hayne Methods section (page 174):
```
ε σ T⁴ = F_walls × ε σ T_walls⁴ + F_sky × ε σ T_sky⁴ + Q_scattered
```

### Lateral Conduction Limit

From paper discussion:
- Conduction eliminates cold traps from ~1 cm near pole to ~10 m at 60° latitude
- Skin depth: L ≈ √(κ P / π) where κ = thermal diffusivity, P = lunar day

---

## References

1. **Hayne et al. (2021)**: Micro cold traps on the Moon. Nature Astronomy 5, 169-175.
   - Main methodology paper
   - Defines all equations and parameters

2. **Ingersoll et al. (1992)**: Stability of polar frosts in spherical bowl-shaped craters. Icarus 100, 40-47.
   - Original bowl crater analytical theory
   - View factor derivations

3. **Hayne et al. (2017)**: Global regolith thermophysical properties of the Moon. JGR Planets 122, 2371-2400.
   - Heat1d thermal model
   - Lunar thermal properties

---

## Action Items

- [ ] Fix `rough_surface_cold_trap_fraction()` to properly use latitude
- [ ] Implement full radiation balance for rough surfaces
- [ ] Validate Ingersoll bowl equations line-by-line
- [ ] Create proper 3D synthetic surface model for Figure 2
- [ ] Rerun all validations
- [ ] Document remaining differences from Hayne's results

---

*This validation analysis identifies critical bugs that prevent accurate replication of Hayne et al. (2021). All microPSR results should be considered invalid until these issues are fixed.*
