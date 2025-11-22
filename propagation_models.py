"""
Lunar Surface Communication Propagation Models

Comprehensive collection of RF propagation models for lunar surface
communications including free-space, multipath, diffraction, and
terrain-specific effects.

References:
- ITU-R P.525: Free-space attenuation
- ITU-R P.526: Propagation by diffraction
- ITU-R P.1546: Point-to-area prediction
- Two-ray ground reflection model
- Okumura-Hata modifications for lunar environment
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class PropagationParameters:
    """Parameters for propagation calculations."""
    frequency_mhz: float
    tx_power_dbm: float
    tx_gain_dbi: float
    tx_height_m: float
    rx_gain_dbi: float
    rx_height_m: float
    distance_km: float
    polarization: str = 'vertical'
    ground_permittivity: float = 3.0  # Lunar regolith
    ground_conductivity: float = 1e-4  # S/m


class LunarPropagationModels:
    """Collection of propagation models for lunar communications."""

    # Physical constants
    C = 299792458.0  # Speed of light (m/s)
    R_MOON = 1737.4  # Moon radius (km)

    @staticmethod
    def free_space_path_loss(distance_km: float, frequency_mhz: float) -> float:
        """
        Calculate free-space path loss (Friis equation).

        FSPL(dB) = 20*log10(d) + 20*log10(f) + 32.45
        where d is in km and f is in MHz

        Args:
            distance_km: Distance in kilometers
            frequency_mhz: Frequency in MHz

        Returns:
            Path loss in dB
        """
        if distance_km <= 0 or frequency_mhz <= 0:
            return np.inf

        fspl = (20 * np.log10(distance_km) +
                20 * np.log10(frequency_mhz) +
                32.45)
        return fspl

    @staticmethod
    def two_ray_ground_reflection(params: PropagationParameters) -> Tuple[float, Dict]:
        """
        Two-ray ground reflection model for multipath propagation.

        Accounts for direct path and ground-reflected path.
        Valid for lunar regolith surface.

        Args:
            params: Propagation parameters

        Returns:
            Tuple of (path_loss_db, details_dict)
        """
        d = params.distance_km * 1000  # Convert to meters
        f = params.frequency_mhz * 1e6  # Convert to Hz
        h_t = params.tx_height_m
        h_r = params.rx_height_m

        # Wavelength
        wavelength = LunarPropagationModels.C / f

        # Direct path distance
        d_direct = np.sqrt(d**2 + (h_t - h_r)**2)

        # Reflected path distance
        d_reflected = np.sqrt(d**2 + (h_t + h_r)**2)

        # Path difference
        delta_d = d_reflected - d_direct

        # Phase difference
        phase_diff = (2 * np.pi * delta_d) / wavelength

        # Fresnel reflection coefficient for lunar regolith
        # Using simplified model for grazing angle
        theta = np.arctan((h_t + h_r) / d)  # Grazing angle

        # Reflection coefficient (magnitude)
        if params.polarization == 'vertical':
            # Vertical polarization
            epsilon_r = params.ground_permittivity
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)

            num = epsilon_r * sin_theta - np.sqrt(epsilon_r - cos_theta**2)
            den = epsilon_r * sin_theta + np.sqrt(epsilon_r - cos_theta**2)
            gamma = np.abs(num / den) if den != 0 else 0.9
        else:
            # Horizontal polarization
            epsilon_r = params.ground_permittivity
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)

            num = sin_theta - np.sqrt(epsilon_r - cos_theta**2)
            den = sin_theta + np.sqrt(epsilon_r - cos_theta**2)
            gamma = np.abs(num / den) if den != 0 else 0.9

        # Field strength components
        # Direct ray
        E_direct = wavelength / (4 * np.pi * d_direct)

        # Reflected ray (with phase inversion for ground reflection)
        E_reflected = gamma * wavelength / (4 * np.pi * d_reflected)

        # Total field (phasor addition)
        E_total_real = E_direct + E_reflected * np.cos(phase_diff + np.pi)
        E_total_imag = E_reflected * np.sin(phase_diff + np.pi)
        E_total = np.sqrt(E_total_real**2 + E_total_imag**2)

        # Power ratio
        power_ratio = (E_total / E_direct)**2

        # Path loss relative to free space
        fspl = LunarPropagationModels.free_space_path_loss(params.distance_km, params.frequency_mhz)

        # Additional loss/gain from multipath
        multipath_effect = 10 * np.log10(power_ratio) if power_ratio > 0 else -40

        total_path_loss = fspl - multipath_effect

        details = {
            'd_direct_m': d_direct,
            'd_reflected_m': d_reflected,
            'path_difference_m': delta_d,
            'phase_difference_deg': np.degrees(phase_diff),
            'reflection_coefficient': gamma,
            'grazing_angle_deg': np.degrees(theta),
            'multipath_effect_db': multipath_effect,
            'fspl_db': fspl,
            'constructive': multipath_effect > 0
        }

        return total_path_loss, details

    @staticmethod
    def knife_edge_diffraction(h_obstacle: float, d1: float, d2: float,
                               frequency_mhz: float) -> float:
        """
        Knife-edge diffraction loss (ITU-R P.526).

        Args:
            h_obstacle: Height of obstacle above line-of-sight (m)
            d1: Distance from TX to obstacle (m)
            d2: Distance from obstacle to RX (m)
            frequency_mhz: Frequency in MHz

        Returns:
            Diffraction loss in dB
        """
        if d1 <= 0 or d2 <= 0:
            return 0.0

        wavelength = LunarPropagationModels.C / (frequency_mhz * 1e6)

        # Fresnel-Kirchhoff diffraction parameter
        v = h_obstacle * np.sqrt(2 * (d1 + d2) / (wavelength * d1 * d2))

        # Diffraction loss
        if v > -0.78:
            loss = 6.9 + 20 * np.log10(np.sqrt((v - 0.1)**2 + 1) + v - 0.1)
        else:
            loss = 0.0

        # Cap maximum loss
        loss = min(loss, 40.0)

        return max(0.0, loss)

    @staticmethod
    def crater_diffraction_loss(crater_depth: float, crater_radius: float,
                                tx_position: Tuple[float, float],
                                rx_position: Tuple[float, float],
                                frequency_mhz: float) -> float:
        """
        Calculate diffraction loss for communication from inside a crater.

        Models the crater rim as a knife-edge obstacle.

        Args:
            crater_depth: Crater depth (m)
            crater_radius: Crater radius (m)
            tx_position: (distance_from_center, height_above_floor) in meters
            rx_position: (distance_from_tx_horizontal, height) in meters
            frequency_mhz: Frequency in MHz

        Returns:
            Diffraction loss in dB
        """
        tx_dist, tx_h = tx_position
        rx_dist_horiz, rx_h = rx_position

        # Calculate if TX is inside crater
        if tx_dist < crater_radius:
            # TX inside crater
            # Height of rim above TX
            rim_height = crater_depth - tx_h

            # Distance to rim
            d1 = np.sqrt((crater_radius - tx_dist)**2)

            # Assume RX is far enough that d2 >> d1
            d2 = rx_dist_horiz * 1000  # Convert km to m

            # Obstacle height above line-of-sight
            # Simplification: use rim height
            h_obs = rim_height

            return LunarPropagationModels.knife_edge_diffraction(
                h_obs, d1, d2, frequency_mhz
            )
        else:
            # TX outside crater, no additional diffraction
            return 0.0

    @staticmethod
    def plane_earth_loss(params: PropagationParameters) -> float:
        """
        Plane earth loss model (far-field approximation).

        Valid when d >> h_t + h_r.

        Args:
            params: Propagation parameters

        Returns:
            Path loss in dB
        """
        d = params.distance_km * 1000  # meters
        h_t = params.tx_height_m
        h_r = params.rx_height_m

        if d <= 0 or h_t <= 0 or h_r <= 0:
            return np.inf

        # Plane earth loss
        loss_linear = (d**2) / (h_t * h_r)**2
        loss_db = 10 * np.log10(loss_linear) if loss_linear > 0 else np.inf

        return loss_db

    @staticmethod
    def egli_model(params: PropagationParameters, terrain_factor: float = 1.0) -> float:
        """
        Modified Egli model for lunar surface.

        Originally for terrestrial VHF/UHF, adapted for lunar vacuum.

        Args:
            params: Propagation parameters
            terrain_factor: Terrain roughness factor (1.0 = smooth, >1 = rough)

        Returns:
            Path loss in dB
        """
        d = params.distance_km
        f = params.frequency_mhz
        h_t = params.tx_height_m
        h_r = params.rx_height_m

        if d <= 0 or f <= 0 or h_t <= 0 or h_r <= 0:
            return np.inf

        # Modified Egli model (no atmospheric absorption on Moon)
        loss = (20 * np.log10(f) +
                40 * np.log10(d) -
                20 * np.log10(h_t) -
                20 * np.log10(h_r) +
                76.3)

        # Add terrain factor
        loss += 10 * np.log10(terrain_factor)

        return loss

    @staticmethod
    def longley_rice_lunar(params: PropagationParameters,
                          climate: int = 7) -> float:
        """
        Simplified Longley-Rice model adapted for lunar environment.

        Climate 7 represents "vacuum" condition.

        Args:
            params: Propagation parameters
            climate: Climate code (7 for lunar vacuum)

        Returns:
            Path loss in dB
        """
        # For lunar surface, use free-space with terrain correction
        fspl = LunarPropagationModels.free_space_path_loss(
            params.distance_km, params.frequency_mhz
        )

        # No atmospheric absorption (major difference from Earth)
        atmospheric_loss = 0.0

        # Terrain roughness factor (simplified)
        # On Moon, regolith scattering is minimal at shallow angles
        terrain_loss = 0.0

        return fspl + atmospheric_loss + terrain_loss

    @staticmethod
    def cost231_hata_lunar(params: PropagationParameters,
                          area_type: str = 'open') -> float:
        """
        COST-231 Hata model adapted for lunar surface.

        Modified to remove atmospheric effects.

        Args:
            params: Propagation parameters
            area_type: 'open', 'suburban', 'urban' (for terrain roughness)

        Returns:
            Path loss in dB
        """
        d = params.distance_km
        f = params.frequency_mhz
        h_t = params.tx_height_m
        h_r = params.rx_height_m

        # Base calculation (modified Hata without atmospheric terms)
        a_hr = (1.1 * np.log10(f) - 0.7) * h_r - (1.56 * np.log10(f) - 0.8)

        loss = (46.3 +
                33.9 * np.log10(f) -
                13.82 * np.log10(max(h_t, 30)) -
                a_hr +
                (44.9 - 6.55 * np.log10(max(h_t, 30))) * np.log10(d))

        # Terrain correction for lunar surface
        if area_type == 'open':
            terrain_corr = 0  # Smooth mare
        elif area_type == 'suburban':
            terrain_corr = 5  # Rough highlands
        else:  # urban equivalent = very rough, boulder fields
            terrain_corr = 10

        return loss + terrain_corr

    @staticmethod
    def rain_attenuation_lunar(distance_km: float, frequency_mhz: float) -> float:
        """
        Rain attenuation for lunar environment.

        Returns 0.0 as there is no atmosphere on the Moon.
        Included for completeness and Earth-based relay scenarios.

        Args:
            distance_km: Distance in km
            frequency_mhz: Frequency in MHz

        Returns:
            Attenuation in dB (always 0.0 for Moon)
        """
        return 0.0

    @staticmethod
    def scattering_loss(params: PropagationParameters,
                       surface_roughness_m: float = 0.1) -> float:
        """
        Surface scattering loss from lunar regolith.

        Args:
            params: Propagation parameters
            surface_roughness_m: RMS surface roughness in meters

        Returns:
            Scattering loss in dB
        """
        f = params.frequency_mhz * 1e6  # Hz
        wavelength = LunarPropagationModels.C / f

        # Rayleigh criterion for surface roughness
        # Scattering becomes significant when roughness > λ/8

        roughness_ratio = surface_roughness_m / wavelength

        if roughness_ratio < 0.125:
            # Smooth surface, minimal scattering
            return 0.0
        else:
            # Rough surface, apply scattering loss
            # Empirical model
            scatter_loss = 10 * np.log10(1 + (roughness_ratio - 0.125)**2)
            return min(scatter_loss, 10.0)  # Cap at 10 dB

    @staticmethod
    def lunar_horizon_distance(height_m: float) -> float:
        """
        Calculate radio horizon distance on lunar surface.

        Args:
            height_m: Antenna height in meters

        Returns:
            Horizon distance in kilometers
        """
        R = LunarPropagationModels.R_MOON * 1000  # meters
        h = height_m

        # Geometric horizon
        if h <= 0:
            return 0.0

        d = np.sqrt(2 * R * h + h**2) / 1000  # km

        return d

    @staticmethod
    def link_budget(params: PropagationParameters,
                   path_loss_db: float,
                   additional_losses_db: float = 0.0) -> Dict[str, float]:
        """
        Calculate complete link budget.

        Args:
            params: Propagation parameters
            path_loss_db: Path loss in dB
            additional_losses_db: Additional system losses

        Returns:
            Dictionary with link budget components
        """
        # EIRP
        eirp_dbm = params.tx_power_dbm + params.tx_gain_dbi

        # Received power
        rx_power_dbm = (eirp_dbm +
                       params.rx_gain_dbi -
                       path_loss_db -
                       additional_losses_db)

        return {
            'eirp_dbm': eirp_dbm,
            'path_loss_db': path_loss_db,
            'rx_power_dbm': rx_power_dbm,
            'additional_losses_db': additional_losses_db
        }


def get_model_description(model_name: str) -> str:
    """Get description of propagation model."""
    descriptions = {
        'free_space': 'Free-Space Path Loss (FSPL) - Ideal vacuum propagation',
        'two_ray': 'Two-Ray Ground Reflection - Includes multipath from surface',
        'knife_edge': 'Knife-Edge Diffraction - Single obstacle diffraction',
        'crater': 'Crater Diffraction - Communication from inside crater',
        'plane_earth': 'Plane Earth Loss - Far-field approximation',
        'egli': 'Egli Model - Modified for lunar terrain',
        'longley_rice': 'Longley-Rice (Lunar) - Irregular terrain model',
        'cost231': 'COST-231 Hata (Lunar) - Adapted empirical model',
        'scattering': 'Surface Scattering - Regolith roughness effects'
    }
    return descriptions.get(model_name, 'Unknown model')


def list_available_models() -> List[str]:
    """List all available propagation models."""
    return [
        'free_space',
        'two_ray',
        'knife_edge',
        'crater',
        'plane_earth',
        'egli',
        'longley_rice',
        'cost231',
        'scattering'
    ]


if __name__ == "__main__":
    # Example usage
    print("Lunar Surface Communication Propagation Models")
    print("=" * 60)
    print("\nAvailable Models:")
    for model in list_available_models():
        print(f"  - {model}: {get_model_description(model)}")

    # Example calculation
    params = PropagationParameters(
        frequency_mhz=2600.0,
        tx_power_dbm=40.0,
        tx_gain_dbi=12.0,
        tx_height_m=10.0,
        rx_gain_dbi=8.0,
        rx_height_m=2.0,
        distance_km=10.0
    )

    print(f"\n\nExample: 10 km link at 2.6 GHz")
    print(f"TX: {params.tx_power_dbm} dBm, {params.tx_gain_dbi} dBi, {params.tx_height_m} m")
    print(f"RX: {params.rx_gain_dbi} dBi, {params.rx_height_m} m")

    # Free space
    fspl = LunarPropagationModels.free_space_path_loss(
        params.distance_km, params.frequency_mhz
    )
    print(f"\nFree-Space Path Loss: {fspl:.2f} dB")

    # Two-ray
    two_ray_loss, details = LunarPropagationModels.two_ray_ground_reflection(params)
    print(f"Two-Ray Path Loss: {two_ray_loss:.2f} dB")
    print(f"  Multipath effect: {details['multipath_effect_db']:+.2f} dB")
    print(f"  {'Constructive' if details['constructive'] else 'Destructive'} interference")
