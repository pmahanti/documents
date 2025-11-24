# Verification of Paper Text with 1 cm Conduction Limit

**Date:** 2025-11-24
**Configuration:** LATERAL_CONDUCTION_LIMIT = 1 cm (0.01 m)

---

## Paper Claims to Verify

The paper states:

> "Owing to their distinct topographic slope distributions (see above and Supplementary Fig. 7), the northern and southern hemispheres display different cold-trap areas, the south having the greater area overall. This topographic dichotomy also leads to differences in the dominant scales of cold traps: the north polar region has more cold traps of size ~1 m–10 km, whereas the south polar region has more cold traps of >10 km. Since the largest cold traps dominate the surface area, the South has greater overall cold-trapping area (~23,000 km2) compared with the north (~17,000 km2). The south-polar estimate is roughly twice as large as an earlier estimate derived from Diviner data poleward of 80° S18, due to our inclusion of all length scales and latitudes. About 2,500 km2 of cold-trapping area exists in shadows smaller than 100 m in size, and ~700 km2 of cold-trapping area is contributed by shadows smaller than 1 m in size."

---

## Model Results (1 cm Conduction Limit)

### Overall Totals

| Hemisphere | Model Result | Paper Claim | Match? |
|------------|--------------|-------------|--------|
| **North** | **7,275 km²** | ~17,000 km² | **✗ NO** |
| **South** | **11,426 km²** | ~23,000 km² | **✗ NO** |
| **TOTAL** | **18,701 km²** | ~40,000 km² | **✗ NO** |

**Discrepancy:** Model predicts about **half** the total cold trap area claimed in the paper.

### Cold Trap Area by Size Range

| Size Range | North | South | Total | Paper Claim |
|------------|-------|-------|-------|-------------|
| **< 1 m** | 2,532 km² | 3,798 km² | **6,331 km²** | ~700 km² |
| **1 m - 10 m** | 2,033 km² | 3,049 km² | **5,082 km²** | (included in < 100m) |
| **10 m - 100 m** | 2,710 km² | 4,066 km² | **6,776 km²** | (included in < 100m) |
| **< 100 m TOTAL** | 7,275 km² | 10,913 km² | **18,188 km²** | ~2,500 km² |
| **100 m - 1 km** | 0 km² | 19 km² | **19 km²** | |
| **1 km - 10 km** | 0 km² | 494 km² | **494 km²** | |
| **> 10 km** | 0 km² | 0 km² | **0 km²** | |

---

## Specific Claim Verification

### ✗ Claim 1: Total Areas
- **Paper:** South ~23,000 km², North ~17,000 km²
- **Model:** South 11,426 km², North 7,275 km²
- **Status:** **MISMATCH** - Model predicts ~50% of claimed values

### ✗ Claim 2: North has more cold traps of size ~1m-10km
- **Paper:** North > South in 1m-10km range
- **Model:** North 5.30×10⁸ cold traps, South 7.95×10⁸ cold traps
- **Status:** **MISMATCH** - Model shows South has MORE, not fewer

### ✗ Claim 3: South has more cold traps > 10km
- **Paper:** South > North for cold traps > 10km
- **Model:** North 0, South 0
- **Status:** **MISMATCH** - Model finds NO cold traps > 10km in either hemisphere

### ✗ Claim 4: ~2,500 km² in shadows < 100m
- **Paper:** ~2,500 km²
- **Model:** 18,188 km²
- **Status:** **MISMATCH** - Model predicts **7.3× MORE** area

### ✗ Claim 5: ~700 km² in shadows < 1m
- **Paper:** ~700 km²
- **Model:** 6,331 km²
- **Status:** **MISMATCH** - Model predicts **9× MORE** area

---

## Analysis

### Why the Mismatches?

1. **Scale factor calibration**: The power-law scale factor K = 2×10¹¹ may not be calibrated correctly for the paper's methodology

2. **Small-scale dominance**: With 1 cm conduction limit, the model allows cold traps down to 1 cm diameter, creating enormous numbers of microscopic features that dominate the total area

3. **Different methodologies**: The paper may use:
   - Different terrain roughness parameters
   - Different hemisphere asymmetry factors
   - Different power-law exponents
   - Additional physics or constraints not in the current model

4. **Observed PSR data**: The model finds very few large cold traps (> 1 km) from the Diviner data, suggesting the observed PSR dataset may be incomplete or processed differently

### Key Observations

- **South/North ratio**: Model gives 1.57, which is qualitatively correct (South > North)
- **Size distribution**: Model heavily weights small features, with 97% of cold trap area in features < 100m
- **Large cold traps**: Model finds almost no cold traps > 1 km, with only 513 km² total in this range
- **Conduction limit impact**: The 1 cm limit enables millions of microscopic cold traps that may not be physically realistic

---

## Conclusion

**The quoted text from the paper is NOT correctly reproduced by the current model with a 1 cm conduction limit.**

### Major Discrepancies:

1. Total cold trap area is **~50% lower** than claimed (18,701 km² vs ~40,000 km²)
2. Area in shadows < 100m is **7.3× higher** than claimed (18,188 km² vs 2,500 km²)
3. Area in shadows < 1m is **9× higher** than claimed (6,331 km² vs 700 km²)
4. No cold traps > 10 km are found (contradicts paper's claim about South having more large cold traps)
5. South has more cold traps than North in ALL size ranges (contradicts claim about North having more in 1m-10km range)

### Possible Resolutions:

1. **Different conduction limit**: The paper may use a different (larger) conduction limit
2. **Different model parameters**: K, b, hemisphere asymmetry, terrain roughness, etc.
3. **Different PSR dataset**: More complete observed PSR data for large features
4. **Different methodology**: The paper's approach may differ substantially from the Hayne model implementation

**Recommendation:** The model parameters and/or methodology need significant adjustment to match the paper's quantitative claims.

---

## Technical Details

- **Conduction limit**: 1 cm (0.01 m)
- **Transition scale**: 1 km (1000 m)
- **Size range modeled**: 0.1 mm to 100 km (8 orders of magnitude)
- **Cold traps enabled**: Features ≥ 1 cm diameter
- **Smallest cold trap found**: ~1 cm diameter
- **Largest cold trap found**: 8,879 m diameter
- **Total number of cold traps**: 4.22×10¹⁴
- **Total number of PSRs**: 2.02×10¹⁸

---

**Last updated:** 2025-11-24
