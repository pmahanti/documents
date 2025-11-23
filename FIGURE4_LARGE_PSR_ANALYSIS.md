# Figure 4 Large PSR Analysis

**Date:** 2025-11-23
**Issue:** Figure 4 top panel doesn't match paper at large scales (≥1km)
**Investigation:** Checking if code uses large PSRs from geodata package

---

## Executive Summary

The current Figure 4 implementation uses a **purely synthetic power-law distribution** for all scales. Investigation reveals that:

1. **Geodata contains 8,039 observed PSRs** with Diviner temperatures
2. **521 large PSRs (D≥1km)** totaling 1,688.75 km²
3. **Observed PSRs follow N ∝ L^(-2.14)**, close to self-similar crater distribution
4. **Synthetic model uses N ∝ L^(-2.8)**, which underestimates large PSR counts

However, incorporating actual large PSRs reveals a critical issue: **most large PSRs are NOT cold traps** due to their latitude distribution.

---

## Key Findings

### 1. Observed Large PSR Distribution

From `psr_with_temperatures.csv` (geodata package + Diviner temperatures):

| Parameter | North | South | Total |
|-----------|-------|-------|-------|
| Total PSRs | 5,655 | 2,384 | 8,039 |
| Total area | 1,305 km² | 1,422 km² | 2,727 km² |
| **Large PSRs (D≥1km)** | **264** | **257** | **521** |
| **Large PSR area** | **660 km²** | **1,029 km²** | **1,689 km²** |
| Diviner cold traps (<110K) | 0 | 41 | 41 |
| Cold trap area | 0.00 km² | 26.66 km² | 26.66 km² |

**Key observation:** Only 41/8,039 (0.5%) of PSRs are cold traps by Diviner measurements!

### 2. Size-Frequency Distribution Comparison

| Approach | Power-law slope | Implication |
|----------|----------------|-------------|
| Observed data | **-2.14** | Self-similar, more large PSRs |
| Synthetic (original code) | **-2.8** | Steeper, fewer large PSRs |
| Lunar craters (typical) | **-2.0** | Self-similar |

The observed distribution has **significantly more large PSRs** than the synthetic model predicts.

### 3. Why Large PSRs Aren't Cold Traps

#### Latitude Distribution of Large PSRs (D≥1km):
- Mean |latitude|: **81.4°**
- Median |latitude|: **81.5°**

| Latitude Band | PSR Count | Total Area | Cold Traps | Cold Trap Area |
|---------------|-----------|------------|------------|----------------|
| 70-80° | 189 | 416 km² | 0 | 0.00 km² |
| 80-85° | 227 | 795 km² | 0 | 0.00 km² |
| 85-90° | 105 | 479 km² | 3 | 19.61 km² |

**Critical finding:** Most large PSRs are at 70-85° latitude where temperatures exceed 110K.

#### Hayne Model Cold Trap Fractions by Latitude:

| Latitude | Hayne f_ct (σs=10°) | Interpretation |
|----------|---------------------|----------------|
| 70°S | 0.0025 (0.25%) | Very few cold traps |
| 75°S | 0.0050 (0.50%) | Few cold traps |
| 80°S | 0.0080 (0.80%) | Some cold traps |
| 85°S | 0.0115 (1.15%) | Moderate cold traps |
| 88°S | 0.0160 (1.60%) | More cold traps |

For large PSRs with mean |lat| = 81.4°:
- **Hayne model estimate:** f_ct ≈ 0.75% → 1,689 km² × 0.0075 = **12.7 km² cold traps**
- **Diviner measurement:** **19.6 km² cold traps**
- **Ratio:** 0.65 (Hayne underestimates by 35%)

---

## Hybrid Approaches Tested

### Version 1: Geodata PSRs with Diviner Temperatures

**Method:** Use actual large PSR counts and Diviner-measured cold trap areas

**Results:**
- Large PSRs (≥300m): 4,126 PSRs
- Cold trap area from large PSRs: **26 km²**
- Total cold trap area: **13,658 km²**
- **Problem:** Too low compared to paper's ~40,000 km²

**Saved:** `figure4_hybrid.png`

### Version 2: Geodata PSRs with Hayne Model Applied

**Method:** Use actual large PSR counts, but apply Hayne model cold trap fractions

**Results:**
- Large PSRs (≥300m): 4,126 PSRs
- Estimated cold trap area from large PSRs: **11 km²**
- Total cold trap area: **13,644 km²**
- **Problem:** Even lower! Hayne model gives smaller f_ct at these latitudes

**Saved:** `figure4_hybrid_v2.png`

### Original: Pure Synthetic Distribution

**Method:** Power-law size-frequency with N ∝ L^(-2.8)

**Results:**
- Total cold trap area: **40,987 km²**
- Matches paper target
- **Problem:** Doesn't use actual observed large PSRs

**Saved:** `figure4_verified.png`

---

## Analysis of Discrepancy

### Why does the synthetic model give ~41,000 km² while observed data gives ~14,000 km²?

1. **Latitude assumption difference:**
   - Synthetic model: Evaluates at representative polar latitude (85°) for all PSRs
   - Observed data: PSRs distributed across 70-90° with mean at 81.4°

2. **Size distribution difference:**
   - Synthetic: N ∝ L^(-2.8) (fewer large PSRs)
   - Observed: N ∝ L^(-2.14) (more large PSRs, but at lower latitudes)

3. **Scale integration:**
   - Synthetic model: Integrates from 100 μm to 100 km at 85° latitude
   - Observed approach: Uses actual latitude distribution of PSRs

### What is the paper's Figure 4 actually showing?

The paper's Figure 4 likely represents a **theoretical size-frequency distribution evaluated at representative polar latitudes**, not the actual observed PSR distribution across all latitudes.

This is a model prediction answering: *"If we had a crater size-frequency distribution at 85° latitude, what would the cumulative cold trap area be?"* rather than *"What is the observed cold trap area from all PSRs?"*

---

## Recommendations

### For matching the paper's Figure 4:

**Use the synthetic power-law approach (current `remake_figure4_verified.py`)** because:
1. Figure 4 appears to be a **theoretical model curve**, not observational data
2. It's evaluated at representative polar latitudes (85°)
3. It demonstrates the multi-scale physics from 100 μm to 100 km

### For validating with observations:

**Use the hybrid approach** to compare:
1. **Model prediction at 85°:** ~41,000 km² (from synthetic distribution)
2. **Observed cold traps (Diviner):** ~27 km² from large PSRs
3. **Discrepancy:** Factor of ~1,500× difference

This huge discrepancy suggests:
- The synthetic model may overestimate small-scale (<1km) cold traps
- OR the model is predicting undiscovered micro-scale cold traps
- OR the model is evaluated at the wrong latitudes/conditions

### Critical question for verification:

**Does the paper intend Figure 4 to show:**
1. **Theoretical model** at representative polar latitudes? → Use synthetic approach
2. **Observed PSR distribution** across all latitudes? → Use hybrid with geodata
3. **Something else?** → Need to clarify paper's methodology

---

## Data Summary

### Geodata Files Available:
- `psr_database.gpkg` - PSR geometries (north and south layers)
- `psr_with_temperatures.csv/.gpkg` - PSRs with Diviner temperatures
- `polar_north_80_summer_max-float.tif` - Diviner temperature map (north)
- `polar_south_80_summer_max-float.tif` - Diviner temperature map (south)

### Code Files Created:
1. `remake_figure4_verified.py` - Pure synthetic (matches paper ~41,000 km²)
2. `remake_figure4_hybrid.py` - Observed PSRs + Diviner temps (~13,658 km²)
3. `remake_figure4_hybrid_v2.py` - Observed PSRs + Hayne model (~13,644 km²)

---

## Conclusions

1. **Observed large PSRs ARE in the codebase** via geodata package
2. **Diviner temperatures ARE being used** to identify cold traps
3. **The discrepancy at large scales is REAL and significant:**
   - Synthetic model: Assumes high-latitude crater distribution
   - Observed data: PSRs distributed across many latitudes, mostly NOT cold traps
4. **The paper's Figure 4 is likely a theoretical model**, not observational

**Next steps:**
- Clarify whether Figure 4 should match observed data or remain theoretical
- If theoretical: Keep current synthetic approach
- If observational: Use hybrid approach and explain the much lower cold trap area

**Status:** Hybrid implementations created and documented. Both approaches are now available in the codebase.
