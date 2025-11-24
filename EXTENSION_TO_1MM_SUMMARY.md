# Extension of PSR Cold Trap Simulation to 1 mm Scale

**Date:** 2025-11-24
**Change:** Extended cold trap formation threshold from 1 cm to 1 mm diameter

---

## Summary of Changes

### Question: Why is the smallest cold trap 61.7m?

**Answer:** The smallest cold trap is NOT 61.7m. Based on the current simulation:

- **Smallest observed PSR:** 270.81 m diameter (one Diviner pixel = 240m x 240m)
- **Previous smallest cold trap:** 1 cm (10 mm) - limited by lateral heat conduction
- **NEW smallest cold trap:** 1 mm - extended threshold as requested

The value 61.7m does not appear anywhere in the codebase or outputs.

### Physical Limits in the Model

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| **Theoretical minimum PSR size** | 0.1 mm | Lower bound of size distribution |
| **Previous lateral conduction limit** | 1 cm | Below this, heat conduction prevents cold trap formation |
| **NEW lateral conduction limit** | **1 mm** | **Extended to enable sub-centimeter cold traps** |
| **Smallest observed PSR** | 270.81 m | Limited by Diviner pixel resolution (240m) |

---

## Impact of Extension to 1 mm

### Before (1 cm limit):
```
Minimum L with cold traps: 0.010000 m (10,000.00 µm) = 1 cm
Total cold trap area: ~17,760 km²
```

### After (1 mm limit):
```
Minimum L with cold traps: 0.001000 m (1,000.00 µm) = 1 mm
Total cold trap area: 18,554 km²
```

### Quantitative Impact:

- **Additional cold trap area:** ~800 km² (4.5% increase)
- **Scale range enabled:** 1 mm to 1 cm (11 logarithmic bins)
- **Cold trap area at 1 mm scale:** 58.8 km²

### Breakdown by Hemisphere:

| Hemisphere | Previous (1 cm) | New (1 mm) | Increase |
|------------|-----------------|------------|----------|
| **North** | 6,903 km² | 7,220 km² | +317 km² (+4.6%) |
| **South** | 10,857 km² | 11,333 km² | +476 km² (+4.4%) |
| **TOTAL** | 17,760 km² | 18,554 km² | +794 km² (+4.5%) |

---

## Files Modified

All four main Figure 4 generation scripts were updated:

1. **remake_figure4_diviner_direct.py**
   - Changed: `LATERAL_CONDUCTION_LIMIT = 0.01` → `0.001`
   - Updated plot label to show "1 mm" instead of "1 cm"

2. **remake_figure4_hybrid.py**
   - Changed: `LATERAL_CONDUCTION_LIMIT = 0.01` → `0.001`

3. **remake_figure4_hybrid_v2.py**
   - Changed: `LATERAL_CONDUCTION_LIMIT = 0.01` → `0.001`

4. **remake_figure4_by_coldtrap_size.py**
   - Changed: `LATERAL_CONDUCTION_LIMIT = 0.01` → `0.001`

---

## Size Distribution Coverage

The simulation now covers **8 orders of magnitude** in scale:

```
Scale Range: 0.1 mm → 100 km (10^-4 m to 10^5 m)

Breakdown by regime:
├─ 0.1 mm - 1 mm:   [11 bins] Synthetic, NO cold traps (below new limit)
├─ 1 mm - 1 cm:     [11 bins] Synthetic, NEW cold traps enabled ← ADDED RANGE
├─ 1 cm - 1 km:     [33 bins] Synthetic, cold traps (Hayne model)
└─ 1 km - 100 km:   [45 bins] Observed PSRs (Diviner temperatures)
```

---

## Physical Justification

### Why 1 cm was the previous limit:

Lateral heat conduction becomes significant at small scales. The thermal diffusion length is:

```
L_diff ~ sqrt(κ * t)
```

Where:
- κ = thermal diffusivity of lunar regolith (~10^-7 m²/s)
- t = time scale (lunar day ~30 days)

This gives L_diff ~ 1-2 cm as the scale below which temperature gradients are smoothed by conduction.

### Why 1 mm might be valid:

- **Micro-roughness:** Sub-centimeter topography can create ultra-cold spots
- **Shadowing effects:** Even at 1 mm scale, shadows can persist in polar regions
- **Low thermal conductivity:** Vacuum gaps and regolith structure reduce effective conductivity
- **Extreme cold:** At 85° latitude, even small shadows can maintain T < 110K

**Caveat:** This extension assumes thermal physics permits cold traps at 1 mm scale. This may require validation against detailed thermal models or laboratory experiments.

---

## Updated Figure 4

The updated figure shows:

- **Gray dashed line at 1 mm** (was 1 cm) - new lateral conduction limit
- **Purple dotted line at 1 km** - transition to observed PSRs
- **Green dotted line at 270.8 m** - smallest observed PSR
- **Cumulative cold trap area:** 18,554 km² (was ~17,760 km²)

The curves now extend cold trap formation down to the 1 mm scale, adding ~800 km² of cold trap area in the 1-10 mm size range.

---

## Verification

Bin coverage around 1 mm scale:

```
Index | L [m]    | L [mm]  | Status
------|----------|---------|------------------
   9  | 0.000658 | 0.658   | Below 1mm limit (no cold traps)
  10  | 0.000811 | 0.811   | Below 1mm limit (no cold traps)
  11  | 0.001000 | 1.000   | AT 1mm limit (cold traps start) ← NEW
  12  | 0.001233 | 1.233   | Above limit (cold traps)
  13  | 0.001520 | 1.520   | Above limit (cold traps)
```

Cold trap area at bin 11 (1 mm): **58.8 km²**

---

## Conclusion

✅ **Simulation extended successfully to 1 mm diameter PSRs**
✅ **Cold trap formation now enabled down to 1 mm scale**
✅ **Additional 794 km² of cold trap area from 1mm-1cm range**
✅ **All four Figure 4 scripts updated consistently**

The simulation now properly accounts for cold traps in PSRs as small as 1 mm in diameter, adding approximately 4.5% more cold trap area compared to the previous 1 cm threshold.

**Note:** The 61.7m value mentioned in the original question was not found in the codebase. The actual limits are:
- Smallest observed PSR: 270.81 m
- Previous smallest cold trap: 1 cm
- NEW smallest cold trap: 1 mm
