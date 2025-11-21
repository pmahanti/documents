"""
Impact Crater Scaling Laws - Inverse Calculation Module

This module implements pi-group scaling relationships (Holsapple & Schmidt 1982,
Holsapple 1993, Holsapple & Housen 2007) to compute impact parameters from
observed crater dimensions.

Given:
    - Crater diameter (D) and depth (d)
    - Target material properties
    - Impactor material properties

Computes:
    - Impactor diameter (L)
    - Impact velocity (U)
    - Impact energy
    - Impact momentum
    - Scaling regime (strength vs. gravity)

References:
    Holsapple, K.A. (1993). "The scaling of impact processes in planetary sciences."
        Annual Review of Earth and Planetary Sciences, 21, 333-373.
    Holsapple, K.A., & Housen, K.R. (2007). "A crater and its ejecta: An interpretation
        of Deep Impact." Icarus, 187, 345-356.
    Melosh, H.J. (1989). "Impact Cratering: A Geologic Process."
"""

import numpy as np
from scipy.optimize import fsolve, minimize_scalar
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import warnings


@dataclass
class Material:
    """Material properties for impact scaling calculations."""
    name: str
    density: float  # kg/m³
    strength: float  # Pa (cohesion/yield strength)
    gravity: float  # m/s² (surface gravity of body)
    K1: float  # Coupling parameter (strength regime)
    mu: float  # Strength scaling exponent
    nu: float  # Gravity scaling exponent (coupling parameter component)

    def __repr__(self):
        return (f"Material(name='{self.name}', "
                f"ρ={self.density:.0f} kg/m³, "
                f"Y={self.strength:.2e} Pa, "
                f"g={self.gravity:.2f} m/s²)")


# Material Database
# -----------------
# Based on Holsapple (1993), Housen & Holsapple (2003), and other sources

MATERIALS = {
    # Lunar materials
    'lunar_regolith': Material(
        name='Lunar Regolith',
        density=1500,  # kg/m³ (loose regolith)
        strength=1e3,  # Pa (very weak cohesion)
        gravity=1.62,  # m/s² (Moon)
        K1=0.132,  # Strength coupling parameter (sand-like)
        mu=0.41,  # Strength exponent (sand/soil)
        nu=0.40,  # Gravity exponent (sand/soil)
    ),

    'lunar_mare': Material(
        name='Lunar Mare Basalt',
        density=3100,  # kg/m³
        strength=1e7,  # Pa (weak rock)
        gravity=1.62,  # m/s²
        K1=0.132,
        mu=0.55,  # Rock-like
        nu=0.40,
    ),

    'lunar_highland': Material(
        name='Lunar Highland Anorthosite',
        density=2800,  # kg/m³
        strength=5e6,  # Pa
        gravity=1.62,  # m/s²
        K1=0.132,
        mu=0.55,
        nu=0.40,
    ),

    # Earth materials
    'sandstone': Material(
        name='Sandstone',
        density=2200,  # kg/m³
        strength=3e7,  # Pa
        gravity=9.81,  # m/s²
        K1=0.132,
        mu=0.55,
        nu=0.40,
    ),

    'granite': Material(
        name='Granite',
        density=2750,  # kg/m³
        strength=2e8,  # Pa
        gravity=9.81,  # m/s²
        K1=0.132,
        mu=0.55,
        nu=0.40,
    ),

    'limestone': Material(
        name='Limestone',
        density=2600,  # kg/m³
        strength=5e7,  # Pa
        gravity=9.81,  # m/s²
        K1=0.132,
        mu=0.55,
        nu=0.40,
    ),

    'dry_soil': Material(
        name='Dry Soil',
        density=1600,  # kg/m³
        strength=1e4,  # Pa
        gravity=9.81,  # m/s²
        K1=0.132,
        mu=0.41,
        nu=0.40,
    ),

    'wet_soil': Material(
        name='Wet Soil',
        density=1800,  # kg/m³
        strength=5e4,  # Pa
        gravity=9.81,  # m/s²
        K1=0.132,
        mu=0.41,
        nu=0.40,
    ),

    'sand': Material(
        name='Sand',
        density=1650,  # kg/m³
        strength=1e3,  # Pa
        gravity=9.81,  # m/s²
        K1=0.132,
        mu=0.41,
        nu=0.40,
    ),

    # Ice targets (Mars, icy moons)
    'water_ice': Material(
        name='Water Ice',
        density=920,  # kg/m³
        strength=1e6,  # Pa (temperature dependent)
        gravity=1.62,  # m/s² (Moon gravity as default, change as needed)
        K1=0.132,
        mu=0.55,
        nu=0.40,
    ),

    'ice_regolith_mix': Material(
        name='Ice-Regolith Mix',
        density=1200,  # kg/m³
        strength=5e5,  # Pa
        gravity=1.62,  # m/s²
        K1=0.132,
        mu=0.50,
        nu=0.40,
    ),
}

# Impactor types
IMPACTORS = {
    'asteroid_rock': Material(
        name='Rocky Asteroid',
        density=2700,  # kg/m³
        strength=0,  # Not used for impactor
        gravity=0,  # Not used for impactor
        K1=0,
        mu=0,
        nu=0,
    ),

    'asteroid_metal': Material(
        name='Metallic Asteroid (Iron)',
        density=7800,  # kg/m³
        strength=0,
        gravity=0,
        K1=0,
        mu=0,
        nu=0,
    ),

    'comet_ice': Material(
        name='Cometary Ice',
        density=500,  # kg/m³ (porous)
        strength=0,
        gravity=0,
        K1=0,
        mu=0,
        nu=0,
    ),

    'comet_ice_dense': Material(
        name='Dense Cometary Ice',
        density=1000,  # kg/m³
        strength=0,
        gravity=0,
        K1=0,
        mu=0,
        nu=0,
    ),
}


class ImpactScaling:
    """
    Impact crater scaling calculator using pi-group formalism.

    Implements Holsapple-Schmidt scaling to relate crater size to
    impact parameters.
    """

    def __init__(self, target: Material, impactor_density: float = 2700):
        """
        Initialize impact scaling calculator.

        Parameters:
        -----------
        target : Material
            Target material properties
        impactor_density : float
            Impactor density in kg/m³ (default: 2700 for rocky asteroid)
        """
        self.target = target
        self.impactor_density = impactor_density

    def pi2(self, a: float, U: float) -> float:
        """
        Gravity pi-group: π₂ = ga/U²

        Ratio of gravitational to kinetic energy.
        """
        return (self.target.gravity * a) / (U**2)

    def pi3(self, U: float) -> float:
        """
        Strength pi-group: π₃ = Y/(ρU²)

        Ratio of strength to kinetic energy (dynamic pressure).
        """
        return self.target.strength / (self.target.density * U**2)

    def pi4(self) -> float:
        """
        Density ratio: π₄ = ρ_target / ρ_impactor
        """
        return self.target.density / self.impactor_density

    def crater_volume_simple(self, D: float, d: float,
                            shape: str = 'paraboloid') -> float:
        """
        Compute crater volume from diameter and depth.

        Parameters:
        -----------
        D : float
            Crater diameter (m)
        d : float
            Crater depth (m)
        shape : str
            Crater shape ('paraboloid', 'conical', 'bowl')

        Returns:
        --------
        V : float
            Crater volume (m³)
        """
        if shape == 'paraboloid':
            # V = (π/8) * D² * d
            return (np.pi / 8) * D**2 * d
        elif shape == 'conical':
            # V = (π/12) * D² * d
            return (np.pi / 12) * D**2 * d
        elif shape == 'bowl':
            # Spherical cap approximation
            # V ≈ (π/6) * d * (3*(D/2)² + d²)
            r = D / 2
            return (np.pi / 6) * d * (3 * r**2 + d**2)
        else:
            raise ValueError(f"Unknown shape: {shape}")

    def transient_to_final(self, D_trans: float, regime: str = 'simple') -> Tuple[float, float]:
        """
        Convert transient crater to final crater dimensions.

        For simple craters: minimal modification
        For complex craters: significant collapse and enlargement

        Parameters:
        -----------
        D_trans : float
            Transient crater diameter (m)
        regime : str
            'simple' or 'complex'

        Returns:
        --------
        D_final, d_final : tuple
            Final crater diameter and depth (m)
        """
        if regime == 'simple':
            # Simple craters: modest modification
            # D_final ≈ 1.25 * D_trans (Melosh 1989)
            D_final = 1.25 * D_trans
            # d/D ratio for simple fresh craters ≈ 0.2
            d_final = 0.2 * D_final
        else:  # complex
            # Complex craters: D_final ≈ 1.3 * D_trans
            D_final = 1.3 * D_trans
            # d/D ratio for complex craters ≈ 0.1 or less
            d_final = 0.1 * D_final

        return D_final, d_final

    def scaling_law(self, a: float, U: float, m: float) -> float:
        """
        Holsapple pi-group scaling law for crater diameter.

        Uses comprehensive pi-scaling with proper exponent formulation
        based on Holsapple (1993) and Collins et al. (2005).

        Parameters:
        -----------
        a : float
            Impactor radius (m)
        U : float
            Impact velocity (m/s)
        m : float
            Impactor mass (kg)

        Returns:
        --------
        D : float
            Transient crater diameter (m)
        """
        L = 2 * a  # Impactor diameter
        nu = self.target.nu
        mu = self.target.mu

        # Dimensionless groups
        p2 = self.pi2(a, U)  # = gL/U²
        p3 = self.pi3(U)     # = Y/(ρU²)
        p4 = self.pi4()      # = ρ_target / ρ_impactor

        # Density scaling: (ρ_imp/ρ_tgt)^(1/3)
        density_factor = (1.0 / p4)**(1.0/3.0)

        # Coupling parameters (empirically determined)
        # These give D/L ratios consistent with experiments
        K1 = 1.25  # Constant for strength regime
        K2 = 1.61  # Constant for gravity regime

        # Determine regime and compute scaling
        # Transition when gL/U² ≈ Y/(ρU²), i.e., when ρgL ≈ Y

        if p3 > p2:
            # Strength regime dominates
            # D/L = K1 * (ρ_imp/ρ_tgt)^(1/3) * (ρU²/Y)^(μ/(2+μ))
            # Since π₃ = Y/(ρU²), we have (ρU²/Y) = 1/π₃
            exponent_strength = mu / (2.0 + mu)
            D = L * K1 * density_factor * (1.0/p3)**exponent_strength
        else:
            # Gravity regime dominates
            # D/L = K2 * (ρ_imp/ρ_tgt)^(1/3) * (U²/gL)^(ν/(2+ν))
            # Since π₂ = gL/U², we have (U²/gL) = 1/π₂
            exponent_gravity = nu / (2.0 + nu)
            D = L * K2 * density_factor * (1.0/p2)**exponent_gravity

        return D

    def scaling_law_volume(self, a: float, U: float, m: float) -> float:
        """
        Compute crater volume from diameter using simple scaling.

        Parameters:
        -----------
        a : float
            Impactor radius (m)
        U : float
            Impact velocity (m/s)
        m : float
            Impactor mass (kg)

        Returns:
        --------
        V : float
            Transient crater volume (m³)
        """
        D = self.scaling_law(a, U, m)
        # Simple crater: d/D ≈ 0.28 for transient cavity
        d = 0.28 * D
        # Paraboloid volume
        V = (np.pi / 8) * D**2 * d
        return V

    def compute_impactor_params(self, D: float, d: float,
                                U: float,
                                shape: str = 'paraboloid',
                                crater_type: str = 'simple') -> Dict:
        """
        Compute impactor parameters given crater size and impact velocity.

        Parameters:
        -----------
        D : float
            Final crater diameter (m)
        d : float
            Final crater depth (m)
        U : float
            Impact velocity (m/s)
        shape : str
            Crater shape for volume calculation
        crater_type : str
            'simple' or 'complex' - affects transient crater estimation

        Returns:
        --------
        results : dict
            Dictionary containing:
            - impactor_diameter : float (m)
            - impactor_mass : float (kg)
            - impact_energy : float (J)
            - impact_momentum : float (kg⋅m/s)
            - pi_2 : float (gravity parameter)
            - pi_3 : float (strength parameter)
            - pi_4 : float (density ratio)
            - regime : str ('strength' or 'gravity')
            - crater_volume : float (m³)
        """
        # Final crater diameter
        D_final = D

        # Estimate transient crater diameter (before collapse/modification)
        if crater_type == 'simple':
            # Simple crater: modest expansion during final modification
            # D_final ≈ 1.25 * D_transient, so D_transient ≈ D_final / 1.25
            D_transient = D_final / 1.25
        else:
            # Complex crater: significant collapse and enlargement
            # D_final ≈ 1.3 * D_transient
            D_transient = D_final / 1.3

        # Now solve for impactor size using scaling law
        # D_transient = f(a, U) where a and m are related by density

        def objective(log_a):
            """Objective function to find impactor radius."""
            a = np.exp(log_a)
            m = (4/3) * np.pi * a**3 * self.impactor_density
            D_predicted = self.scaling_law(a, U, m)
            return (np.log(D_predicted) - np.log(D_transient))**2

        # Initial guess: use simple scaling D ~ L (impactor diameter)
        # Rough estimate: D/L ~ 20 for typical impacts
        L_guess = D_transient / 20.0
        a_guess = L_guess / 2.0
        log_a_guess = np.log(max(a_guess, 0.01))  # Ensure positive

        # Optimize
        from scipy.optimize import minimize
        result = minimize(objective, log_a_guess, method='Nelder-Mead')

        if not result.success:
            warnings.warn("Optimization did not converge perfectly")

        a_impactor = np.exp(result.x[0])
        L_impactor = 2 * a_impactor
        m_impactor = (4/3) * np.pi * a_impactor**3 * self.impactor_density

        # Compute dimensionless parameters
        p2 = self.pi2(a_impactor, U)
        p3 = self.pi3(U)
        p4 = self.pi4()

        # Determine regime
        if p3 > p2:
            regime = 'strength'
        else:
            regime = 'gravity'

        # Impact energy and momentum
        E_impact = 0.5 * m_impactor * U**2
        p_impact = m_impactor * U

        # Compute volumes
        V_final = self.crater_volume_simple(D_final, d, shape)
        d_transient = 0.28 * D_transient  # Transient d/D ratio
        V_transient = self.crater_volume_simple(D_transient, d_transient, shape)

        return {
            'impactor_diameter': L_impactor,
            'impactor_radius': a_impactor,
            'impactor_mass': m_impactor,
            'impact_energy': E_impact,
            'impact_momentum': p_impact,
            'pi_2': p2,
            'pi_3': p3,
            'pi_4': p4,
            'regime': regime,
            'crater_diameter_final': D_final,
            'crater_diameter_transient': D_transient,
            'crater_volume_final': V_final,
            'crater_volume_transient': V_transient,
        }

    def velocity_scan(self, D: float, d: float,
                     U_range: Tuple[float, float] = (5e3, 50e3),
                     n_points: int = 50,
                     shape: str = 'paraboloid',
                     crater_type: str = 'simple') -> Dict:
        """
        Scan over range of impact velocities to show possible impactor sizes.

        Useful when impact velocity is unknown - shows trade-off between
        impactor size and velocity.

        Parameters:
        -----------
        D : float
            Final crater diameter (m)
        d : float
            Final crater depth (m)
        U_range : tuple
            (U_min, U_max) velocity range in m/s
        n_points : int
            Number of velocity points to sample
        shape : str
            Crater shape for volume calculation
        crater_type : str
            'simple' or 'complex'

        Returns:
        --------
        results : dict
            Arrays of:
            - velocities
            - impactor_diameters
            - impactor_masses
            - impact_energies
            - regimes
        """
        U_values = np.linspace(U_range[0], U_range[1], n_points)

        L_values = []
        m_values = []
        E_values = []
        regimes = []

        for U in U_values:
            res = self.compute_impactor_params(D, d, U, shape, crater_type)
            L_values.append(res['impactor_diameter'])
            m_values.append(res['impactor_mass'])
            E_values.append(res['impact_energy'])
            regimes.append(res['regime'])

        return {
            'velocities': U_values,
            'impactor_diameters': np.array(L_values),
            'impactor_masses': np.array(m_values),
            'impact_energies': np.array(E_values),
            'regimes': regimes,
        }


def print_materials():
    """Print available target materials."""
    print("\n=== Available Target Materials ===")
    for key, mat in MATERIALS.items():
        print(f"\n'{key}':")
        print(f"  {mat}")

    print("\n\n=== Available Impactor Types ===")
    for key, imp in IMPACTORS.items():
        print(f"\n'{key}':")
        print(f"  Name: {imp.name}")
        print(f"  Density: {imp.density:.0f} kg/m³")


def format_results(results: Dict, crater_D: float, crater_d: float,
                  velocity: float, target_name: str, impactor_name: str):
    """
    Pretty-print impact calculation results.

    Parameters:
    -----------
    results : dict
        Results from compute_impactor_params()
    crater_D : float
        Crater diameter (m)
    crater_d : float
        Crater depth (m)
    velocity : float
        Impact velocity (m/s)
    target_name : str
        Target material name
    impactor_name : str
        Impactor type name
    """
    print("\n" + "="*70)
    print("IMPACT PARAMETER CALCULATION RESULTS")
    print("="*70)

    print(f"\nInput Parameters:")
    print(f"  Crater Diameter:        {crater_D:,.1f} m ({crater_D/1000:.2f} km)")
    print(f"  Crater Depth:           {crater_d:,.1f} m")
    print(f"  Depth/Diameter Ratio:   {crater_d/crater_D:.3f}")
    print(f"  Impact Velocity:        {velocity:,.0f} m/s ({velocity/1000:.1f} km/s)")
    print(f"  Target Material:        {target_name}")
    print(f"  Impactor Type:          {impactor_name}")

    print(f"\n{'─'*70}")
    print(f"Computed Impactor Properties:")
    print(f"{'─'*70}")

    L = results['impactor_diameter']
    m = results['impactor_mass']
    E = results['impact_energy']

    print(f"  Impactor Diameter:      {L:,.1f} m ({L/1000:.3f} km)")
    print(f"  Impactor Mass:          {m:.2e} kg ({m/1e12:.2f} trillion kg)")
    print(f"  Impact Energy:          {E:.2e} J")
    print(f"  Impact Energy:          {E/4.184e15:.2f} Mt TNT")
    print(f"  Impact Momentum:        {results['impact_momentum']:.2e} kg⋅m/s")

    print(f"\n{'─'*70}")
    print(f"Scaling Analysis:")
    print(f"{'─'*70}")
    print(f"  π₂ (gravity):           {results['pi_2']:.2e}")
    print(f"  π₃ (strength):          {results['pi_3']:.2e}")
    print(f"  π₄ (density ratio):     {results['pi_4']:.3f}")
    print(f"  Dominant Regime:        {results['regime'].upper()}")

    if results['regime'] == 'strength':
        print(f"  → Material strength controls crater size")
    else:
        print(f"  → Gravity controls crater size")

    print(f"\n{'─'*70}")
    print(f"Crater Volumes:")
    print(f"{'─'*70}")
    print(f"  Final Crater Volume:    {results['crater_volume_final']:.2e} m³")
    print(f"  Transient Volume:       {results['crater_volume_transient']:.2e} m³")

    print("\n" + "="*70 + "\n")


# Example usage
if __name__ == "__main__":
    print("Impact Crater Scaling Calculator")
    print("="*70)

    # Show available materials
    print_materials()

    # Example 1: Lunar crater
    print("\n\n" + "="*70)
    print("EXAMPLE 1: Small Fresh Lunar Crater")
    print("="*70)

    target = MATERIALS['lunar_regolith']
    impactor = IMPACTORS['asteroid_rock']

    calc = ImpactScaling(target, impactor.density)

    # Crater: 100 m diameter, 20 m deep, typical impact velocity 15 km/s
    results = calc.compute_impactor_params(
        D=100,  # m
        d=20,   # m
        U=15000,  # m/s (15 km/s - typical lunar impact)
        crater_type='simple'
    )

    format_results(results, 100, 20, 15000,
                  target.name, impactor.name)

    # Example 2: Large lunar crater
    print("\n\n" + "="*70)
    print("EXAMPLE 2: Large Lunar Crater (Complex)")
    print("="*70)

    target = MATERIALS['lunar_mare']
    impactor = IMPACTORS['asteroid_rock']

    calc = ImpactScaling(target, impactor.density)

    # Crater: 10 km diameter, 1 km deep
    results = calc.compute_impactor_params(
        D=10000,  # m
        d=1000,   # m
        U=20000,  # m/s (20 km/s)
        crater_type='complex'
    )

    format_results(results, 10000, 1000, 20000,
                  target.name, impactor.name)

    # Example 3: Velocity trade-off study
    print("\n\n" + "="*70)
    print("EXAMPLE 3: Velocity Scan - 500m Crater")
    print("="*70)
    print("\nShowing impactor size vs. velocity trade-off...")

    target = MATERIALS['lunar_regolith']
    impactor = IMPACTORS['asteroid_rock']
    calc = ImpactScaling(target, impactor.density)

    scan = calc.velocity_scan(
        D=500,  # m
        d=100,  # m
        U_range=(10000, 30000),  # 10-30 km/s
        n_points=5
    )

    print(f"\n{'Velocity (km/s)':<20} {'Impactor Diameter (m)':<25} {'Energy (Mt TNT)':<20} {'Regime':<15}")
    print("─" * 80)
    for i, U in enumerate(scan['velocities']):
        L = scan['impactor_diameters'][i]
        E = scan['impact_energies'][i]
        regime = scan['regimes'][i]
        print(f"{U/1000:<20.1f} {L:<25.2f} {E/4.184e15:<20.3f} {regime:<15}")

    print("\n" + "="*70)
    print("Note: Smaller impactors at higher velocities produce same crater size")
    print("="*70)
