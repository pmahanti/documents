#!/usr/bin/env python3
"""
Generate Theoretical Comparison Figures for Bowl vs Cone Craters

Creates comprehensive figures showing:
1. Crater geometry cross-sections
2. View factor diagrams
3. Shadow geometry evolution
4. Radiation balance illustrations
5. Temperature vs depth profiles
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Arc, FancyBboxPatch, Circle, Wedge
import matplotlib
matplotlib.use('Agg')

# Physical constants
SIGMA_SB = 5.67051e-8


def plot_crater_geometry_comparison():
    """
    Figure 1: Side-by-side cross-sections of bowl vs cone craters.
    Shows geometric differences clearly.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Common parameters
    D = 1000  # diameter in meters
    d = 100   # depth in meters (gamma = 0.1)
    gamma = d / D

    # BOWL CRATER (left)
    ax = axes[0]

    # Calculate spherical bowl profile
    R = D / 2.0
    R_sphere = (R**2 + d**2) / (2.0 * d)

    # Generate bowl profile
    theta_max = np.arcsin(R / R_sphere)
    theta = np.linspace(-theta_max, theta_max, 100)

    # Bowl coordinates (centered at crater rim)
    x_bowl = R_sphere * np.sin(theta)
    y_bowl = -(R_sphere * np.cos(theta) - (R_sphere - d))

    # Plot bowl crater
    ax.fill_between(x_bowl, y_bowl, -d-50, color='lightgray', alpha=0.3, label='Crater')
    ax.plot(x_bowl, y_bowl, 'b-', linewidth=3, label='Bowl profile')

    # Surface line
    ax.plot([-R-100, R+100], [0, 0], 'brown', linewidth=2, label='Surface')

    # Annotations
    # Diameter
    ax.annotate('', xy=(R, 20), xytext=(-R, 20),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(0, 30, f'D = {D} m', ha='center', fontsize=12, color='red', fontweight='bold')

    # Depth
    ax.annotate('', xy=(R+80, 0), xytext=(R+80, -d),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax.text(R+120, -d/2, f'd = {d} m', ha='left', fontsize=12, color='green', fontweight='bold')

    # Radius of curvature
    center_x = 0
    center_y = -(R_sphere - d)
    ax.plot([center_x], [center_y], 'ko', markersize=8)
    ax.plot([center_x, 0], [center_y, -d], 'k--', linewidth=1, alpha=0.5)
    ax.text(20, center_y + 50, f'$R_{{sphere}}$ = {R_sphere:.0f} m', fontsize=11, color='black')

    # gamma annotation
    ax.text(0, -d-30, f'γ = d/D = {gamma:.3f}', ha='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    ax.set_xlim([-R-150, R+200])
    ax.set_ylim([-d-80, 60])
    ax.set_aspect('equal')
    ax.set_xlabel('Radial Distance (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
    ax.set_title('A. BOWL-SHAPED CRATER\n(Spherical Cap - Hayne/Ingersoll)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)

    # CONE CRATER (right)
    ax = axes[1]

    # Generate cone profile
    x_cone = np.array([-R, 0, R])
    y_cone = np.array([0, -d, 0])

    # Plot cone crater
    ax.fill_between(x_cone, y_cone, -d-50, color='lightgray', alpha=0.3, label='Crater')
    ax.plot(x_cone, y_cone, 'r-', linewidth=3, label='Cone profile')

    # Surface line
    ax.plot([-R-100, R+100], [0, 0], 'brown', linewidth=2, label='Surface')

    # Annotations
    # Diameter
    ax.annotate('', xy=(R, 20), xytext=(-R, 20),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(0, 30, f'D = {D} m', ha='center', fontsize=12, color='red', fontweight='bold')

    # Depth
    ax.annotate('', xy=(R+80, 0), xytext=(R+80, -d),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax.text(R+120, -d/2, f'd = {d} m', ha='left', fontsize=12, color='green', fontweight='bold')

    # Wall slope angle
    theta_wall = np.arctan(2*gamma) * 180 / np.pi
    arc = Arc((0, 0), 100, 100, angle=0, theta1=180+theta_wall, theta2=180,
              color='purple', linewidth=2)
    ax.add_patch(arc)
    ax.text(-80, -15, f'θ$_w$ = {theta_wall:.1f}°', fontsize=11, color='purple', fontweight='bold')

    # gamma annotation
    ax.text(0, -d-30, f'γ = d/D = {gamma:.3f}', ha='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Wall slope annotation (right side)
    mid_x = R/2
    mid_y = -d/2
    ax.plot([mid_x-30, mid_x+30], [mid_y-30, mid_y+30], 'purple', linewidth=2, alpha=0.5)
    ax.text(mid_x+50, mid_y, 'Constant\nslope', fontsize=10, color='purple')

    ax.set_xlim([-R-150, R+200])
    ax.set_ylim([-d-80, 60])
    ax.set_aspect('equal')
    ax.set_xlabel('Radial Distance (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
    ax.set_title('B. CONICAL CRATER\n(Inverted Cone - This Work)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)

    plt.suptitle('Figure 1: Crater Geometry Comparison (Cross-Section)\n' +
                 'Same D and d, Different Curvature',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('/home/user/documents/fig1_crater_geometry.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fig1_crater_geometry.png")
    plt.close()


def plot_view_factor_diagrams():
    """
    Figure 2: View factor diagrams showing radiation exchange geometry.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    gamma = 0.1

    # BOWL VIEW FACTORS (left)
    ax = axes[0]
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.axis('off')

    # Draw simplified bowl crater
    theta = np.linspace(0, np.pi, 50)
    x_bowl = 5 + 2.5 * np.cos(theta)
    y_bowl = 2.5 + 1.5 * np.sin(theta)
    ax.fill_between(x_bowl, 0.5, y_bowl, color='lightgray', alpha=0.3)
    ax.plot(x_bowl, y_bowl, 'b-', linewidth=3)

    # Point at crater floor
    ax.plot([5], [1.0], 'ro', markersize=15, label='Observer point')

    # View to sky (cone of radiation)
    sky_cone_x = [5, 3, 7, 5]
    sky_cone_y = [1.0, 9, 9, 1.0]
    ax.fill(sky_cone_x, sky_cone_y, color='cyan', alpha=0.3, label='View to sky')
    ax.plot([5, 3], [1.0, 9], 'c--', linewidth=2)
    ax.plot([5, 7], [1.0, 9], 'c--', linewidth=2)

    # View to walls
    ax.annotate('', xy=(3, 2.5), xytext=(5, 1.0),
                arrowprops=dict(arrowstyle='->', color='orange', lw=3))
    ax.annotate('', xy=(7, 2.5), xytext=(5, 1.0),
                arrowprops=dict(arrowstyle='->', color='orange', lw=3))
    ax.text(3.5, 1.5, 'Wall\nradiation', fontsize=10, color='orange', fontweight='bold')

    # Labels
    ax.text(5, 9.5, 'SKY (3 K)', ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.7))
    ax.text(5, 0.3, 'CRATER FLOOR', ha='center', fontsize=11, fontweight='bold')

    # View factor values
    f_sky_bowl = 1.0 - min(gamma/0.2, 0.7)
    f_walls_bowl = min(gamma/0.2, 0.7)

    ax.text(5, 7, f'$F_{{sky}}$ ≈ {f_sky_bowl:.2f}', ha='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='cyan', lw=2))
    ax.text(7.5, 1.8, f'$F_{{walls}}$ ≈ {f_walls_bowl:.2f}', ha='left', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='orange', lw=2))

    ax.text(5, -0.5, 'Approximate\n(Empirical)', ha='center', fontsize=11, style='italic',
            color='blue')

    ax.set_title('A. BOWL CRATER VIEW FACTORS', fontsize=13, fontweight='bold')

    # CONE VIEW FACTORS (right)
    ax = axes[1]
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.axis('off')

    # Draw cone crater
    cone_x = [2.5, 5, 7.5, 2.5]
    cone_y = [2.5, 1.0, 2.5, 2.5]
    cone_y_fill = [0.5, 0.5, 0.5, 0.5]
    ax.fill_between([2.5, 5, 7.5], cone_y_fill[:3], [2.5, 1.0, 2.5], color='lightgray', alpha=0.3)
    ax.plot([2.5, 5, 7.5], [2.5, 1.0, 2.5], 'r-', linewidth=3)

    # Point at crater floor
    ax.plot([5], [1.0], 'ro', markersize=15, label='Observer point')

    # View to sky (much larger cone)
    alpha_deg = np.arctan(1.0/(2*gamma)) * 180 / np.pi
    sky_width = 4.8
    sky_cone_x = [5, 5-sky_width, 5+sky_width, 5]
    sky_cone_y = [1.0, 9, 9, 1.0]
    ax.fill(sky_cone_x, sky_cone_y, color='cyan', alpha=0.3, label='View to sky')
    ax.plot([5, 5-sky_width], [1.0, 9], 'c--', linewidth=2)
    ax.plot([5, 5+sky_width], [1.0, 9], 'c--', linewidth=2)

    # Opening half-angle annotation
    arc = Arc((5, 1.0), 1.5, 1.5, angle=0, theta1=90-alpha_deg, theta2=90+alpha_deg,
              color='purple', linewidth=2)
    ax.add_patch(arc)
    ax.text(5, 2.0, f'α = {alpha_deg:.1f}°', ha='center', fontsize=10, color='purple',
            fontweight='bold')

    # Minimal view to walls
    ax.annotate('', xy=(2.6, 2.4), xytext=(5, 1.0),
                arrowprops=dict(arrowstyle='->', color='orange', lw=2))
    ax.text(2.2, 2.0, 'Small wall\nradiation', fontsize=9, color='orange', ha='right')

    # Labels
    ax.text(5, 9.5, 'SKY (3 K)', ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.7))
    ax.text(5, 0.3, 'CRATER FLOOR', ha='center', fontsize=11, fontweight='bold')

    # View factor values (exact)
    f_sky_cone = 1.0 / (1.0 + 4*gamma**2)
    f_walls_cone = 4*gamma**2 / (1.0 + 4*gamma**2)

    ax.text(5, 7, f'$F_{{sky}}$ = {f_sky_cone:.3f}', ha='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='cyan', lw=2))
    ax.text(7.8, 1.8, f'$F_{{walls}}$ = {f_walls_cone:.3f}', ha='left', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='orange', lw=2))

    ax.text(5, -0.5, 'Exact Analytical\n$F_{{sky}} = 1/(1+4γ^2)$', ha='center', fontsize=11,
            style='italic', color='red')

    ax.set_title('B. CONE CRATER VIEW FACTORS', fontsize=13, fontweight='bold')

    plt.suptitle('Figure 2: View Factor Comparison (γ = 0.1)\n' +
                 f'Cone sees {f_sky_cone/f_sky_bowl:.1f}× more sky than bowl!',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('/home/user/documents/fig2_view_factors.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fig2_view_factors.png")
    plt.close()


def plot_view_factor_curves():
    """
    Figure 3: View factors as function of gamma for both geometries.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    gamma_vals = np.linspace(0.05, 0.25, 100)

    # Calculate view factors
    bowl_f_sky = np.array([1.0 - min(g/0.2, 0.7) for g in gamma_vals])
    bowl_f_walls = np.array([min(g/0.2, 0.7) for g in gamma_vals])

    cone_f_sky = 1.0 / (1.0 + 4*gamma_vals**2)
    cone_f_walls = 4*gamma_vals**2 / (1.0 + 4*gamma_vals**2)

    # Panel A: F_sky comparison
    ax = axes[0]
    ax.plot(gamma_vals, bowl_f_sky, 'b-', linewidth=3, label='Bowl (approximate)')
    ax.plot(gamma_vals, cone_f_sky, 'r-', linewidth=3, label='Cone (exact)')
    ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5, label='γ = 0.1 (typical)')
    ax.axhline(y=0.5, color='blue', linestyle=':', alpha=0.5)
    ax.axhline(y=0.962, color='red', linestyle=':', alpha=0.5)

    ax.set_xlabel('Depth-to-diameter ratio (γ = d/D)', fontsize=12, fontweight='bold')
    ax.set_ylabel('View Factor to Sky ($F_{sky}$)', fontsize=12, fontweight='bold')
    ax.set_title('A. View Factor to Sky vs γ', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.05, 0.25])
    ax.set_ylim([0, 1.05])

    # Annotations at γ = 0.1
    idx_010 = np.argmin(np.abs(gamma_vals - 0.1))
    ax.plot([0.1], [bowl_f_sky[idx_010]], 'bo', markersize=10)
    ax.plot([0.1], [cone_f_sky[idx_010]], 'ro', markersize=10)
    ax.text(0.12, bowl_f_sky[idx_010], f'{bowl_f_sky[idx_010]:.3f}', fontsize=10, color='blue')
    ax.text(0.12, cone_f_sky[idx_010], f'{cone_f_sky[idx_010]:.3f}', fontsize=10, color='red')

    # Panel B: F_walls comparison
    ax = axes[1]
    ax.plot(gamma_vals, bowl_f_walls, 'b-', linewidth=3, label='Bowl (approximate)')
    ax.plot(gamma_vals, cone_f_walls, 'r-', linewidth=3, label='Cone (exact)')
    ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5, label='γ = 0.1 (typical)')
    ax.axhline(y=0.5, color='blue', linestyle=':', alpha=0.5)
    ax.axhline(y=0.038, color='red', linestyle=':', alpha=0.5)

    ax.set_xlabel('Depth-to-diameter ratio (γ = d/D)', fontsize=12, fontweight='bold')
    ax.set_ylabel('View Factor to Walls ($F_{walls}$)', fontsize=12, fontweight='bold')
    ax.set_title('B. View Factor to Walls vs γ', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.05, 0.25])
    ax.set_ylim([0, 0.8])

    # Annotations at γ = 0.1
    ax.plot([0.1], [bowl_f_walls[idx_010]], 'bo', markersize=10)
    ax.plot([0.1], [cone_f_walls[idx_010]], 'ro', markersize=10)
    ax.text(0.12, bowl_f_walls[idx_010], f'{bowl_f_walls[idx_010]:.3f}', fontsize=10, color='blue')
    ax.text(0.12, cone_f_walls[idx_010], f'{cone_f_walls[idx_010]:.3f}', fontsize=10, color='red')

    plt.suptitle('Figure 3: View Factors vs Depth-to-Diameter Ratio\n' +
                 'Cone has exact analytical solution; Bowl uses approximation',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('/home/user/documents/fig3_view_factor_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fig3_view_factor_curves.png")
    plt.close()


def plot_shadow_geometry():
    """
    Figure 4: Shadow geometry at different solar elevations.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    gamma = 0.1
    D = 500
    d = gamma * D
    R = D / 2

    solar_elevations = [2, 5, 10]  # degrees

    from bowl_crater_thermal import crater_shadow_area_fraction
    from ingersol_cone_theory import cone_shadow_fraction

    for idx, e in enumerate(solar_elevations):
        # BOWL shadow (top row)
        ax = axes[0, idx]

        # Draw bowl profile
        R_sphere = (R**2 + d**2) / (2.0 * d)
        theta_max = np.arcsin(R / R_sphere)
        theta = np.linspace(-theta_max, theta_max, 100)
        x_bowl = R_sphere * np.sin(theta)
        y_bowl = -(R_sphere * np.cos(theta) - (R_sphere - d))

        ax.fill_between(x_bowl, y_bowl, -d-20, color='lightgray', alpha=0.3)
        ax.plot(x_bowl, y_bowl, 'b-', linewidth=2)
        ax.plot([-R-50, R+50], [0, 0], 'brown', linewidth=2)

        # Calculate shadow
        bowl_shadow = crater_shadow_area_fraction(gamma, -85, e)
        f_shadow = bowl_shadow['instantaneous_shadow_fraction']
        r_shadow = R * np.sqrt(f_shadow)

        # Draw shadow region
        if r_shadow > 0:
            shadow_mask = np.abs(x_bowl) <= r_shadow
            if np.any(shadow_mask):
                ax.fill_between(x_bowl[shadow_mask], y_bowl[shadow_mask], -d-20,
                               color='darkblue', alpha=0.6, label='Shadow')

        # Draw sun ray
        sun_angle_rad = e * np.pi / 180
        ray_length = 150
        ray_x = [-ray_length * np.cos(sun_angle_rad), 0]
        ray_y = [0, ray_length * np.sin(sun_angle_rad)]
        ax.plot(ray_x, ray_y, 'orange', linewidth=2, linestyle='--', label=f'Sun (e={e}°)')
        ax.plot([ray_x[0]], [ray_y[0]], 'o', color='orange', markersize=12)

        ax.set_xlim([-R-80, R+80])
        ax.set_ylim([-d-30, 80])
        ax.set_aspect('equal')
        ax.set_title(f'Bowl: e = {e}°, f_sh = {f_shadow:.3f}', fontsize=11, fontweight='bold')
        ax.set_ylabel('Depth (m)', fontsize=10)
        if idx == 0:
            ax.legend(loc='lower left', fontsize=8)
        ax.grid(True, alpha=0.3)

        # CONE shadow (bottom row)
        ax = axes[1, idx]

        # Draw cone profile
        x_cone = np.array([-R, 0, R])
        y_cone = np.array([0, -d, 0])

        ax.fill_between(x_cone, y_cone, -d-20, color='lightgray', alpha=0.3)
        ax.plot(x_cone, y_cone, 'r-', linewidth=2)
        ax.plot([-R-50, R+50], [0, 0], 'brown', linewidth=2)

        # Calculate shadow
        cone_sh = cone_shadow_fraction(gamma, e)
        f_shadow_cone = cone_sh['shadow_fraction']
        r_shadow_cone = R * cone_sh['shadow_radius_normalized']

        # Draw shadow region
        if r_shadow_cone > 0 and r_shadow_cone <= R:
            shadow_x = np.array([-r_shadow_cone, 0, r_shadow_cone])
            shadow_y = np.array([0, -d * (1 - r_shadow_cone/R), 0])
            ax.fill_between(shadow_x, shadow_y, -d-20, color='darkred', alpha=0.6, label='Shadow')
        elif f_shadow_cone >= 1.0:
            # Fully shadowed
            ax.fill_between(x_cone, y_cone, -d-20, color='darkred', alpha=0.6, label='Shadow (full)')

        # Draw sun ray
        ax.plot(ray_x, ray_y, 'orange', linewidth=2, linestyle='--', label=f'Sun (e={e}°)')
        ax.plot([ray_x[0]], [ray_y[0]], 'o', color='orange', markersize=12)

        # Critical angle
        theta_crit = np.arctan(2*gamma) * 180 / np.pi
        if e <= theta_crit:
            ax.text(0, -d-15, f'FULLY SHADOWED\n(e < θ_crit = {theta_crit:.1f}°)',
                   ha='center', fontsize=9, color='red', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        ax.set_xlim([-R-80, R+80])
        ax.set_ylim([-d-30, 80])
        ax.set_aspect('equal')
        ax.set_title(f'Cone: e = {e}°, f_sh = {f_shadow_cone:.3f}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Radial Distance (m)', fontsize=10)
        ax.set_ylabel('Depth (m)', fontsize=10)
        if idx == 0:
            ax.legend(loc='lower left', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Figure 4: Shadow Geometry vs Solar Elevation (γ = 0.1, Latitude 85°S)\n' +
                 'Top: Bowl | Bottom: Cone',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig('/home/user/documents/fig4_shadow_geometry.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fig4_shadow_geometry.png")
    plt.close()


def plot_radiation_balance():
    """
    Figure 5: Radiation balance energy flow diagrams.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    gamma = 0.1

    # BOWL radiation balance (left)
    ax = axes[0]
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.axis('off')

    # Crater floor
    floor = FancyBboxPatch((3, 0.5), 4, 0.5, boxstyle="round,pad=0.05",
                           edgecolor='black', facecolor='gray', linewidth=2)
    ax.add_patch(floor)
    ax.text(5, 0.75, 'CRATER FLOOR\n(Shadow)', ha='center', va='center',
           fontsize=10, fontweight='bold')

    # Incoming radiation arrows
    # Q_sky (from above)
    f_sky_bowl = 0.5
    Q_sky = f_sky_bowl * SIGMA_SB * 3**4
    ax.annotate('', xy=(5, 1.0), xytext=(5, 3.5),
                arrowprops=dict(arrowstyle='->', color='cyan', lw=4))
    ax.text(5.5, 2.2, f'$Q_{{sky}}$\n{Q_sky:.2e} W/m²\n$F_{{sky}}={f_sky_bowl}$',
           fontsize=9, color='cyan', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Q_thermal (from walls)
    f_walls_bowl = 0.5
    T_wall = 60  # K (example)
    Q_thermal = f_walls_bowl * 0.95 * SIGMA_SB * T_wall**4
    ax.annotate('', xy=(4.0, 0.7), xytext=(1.5, 2.5),
                arrowprops=dict(arrowstyle='->', color='orange', lw=4))
    ax.annotate('', xy=(6.0, 0.7), xytext=(8.5, 2.5),
                arrowprops=dict(arrowstyle='->', color='orange', lw=4))
    ax.text(1.5, 3.2, f'$Q_{{thermal}}$\n{Q_thermal:.2f} W/m²\n$F_{{walls}}={f_walls_bowl}$\n$T_{{wall}}={T_wall}$ K',
           fontsize=9, color='orange', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Q_scattered (scattered sunlight)
    Q_scattered = 0.3  # W/m² (small)
    ax.annotate('', xy=(3.5, 0.9), xytext=(2.0, 4.0),
                arrowprops=dict(arrowstyle='->', color='yellow', lw=2))
    ax.text(2.0, 4.5, f'$Q_{{scattered}}$\n{Q_scattered:.2f} W/m²',
           fontsize=9, color='orange', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Total Q in
    Q_total_bowl = Q_sky + Q_thermal + Q_scattered

    # Emitted radiation (out)
    T_shadow_bowl = (Q_total_bowl / (0.95 * SIGMA_SB))**0.25
    ax.annotate('', xy=(5, 6.5), xytext=(5, 1.0),
                arrowprops=dict(arrowstyle='->', color='red', lw=4))
    ax.text(5, 7.5, f'Emitted: εσT⁴\nT = {T_shadow_bowl:.1f} K\nQ = {Q_total_bowl:.2f} W/m²',
           ha='center', fontsize=10, color='red', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', lw=2))

    # Energy balance equation
    ax.text(5, 9, 'Bowl: Large $F_{walls}$ → Large $Q_{thermal}$ → WARMER',
           ha='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    ax.set_title('A. BOWL CRATER RADIATION BALANCE', fontsize=13, fontweight='bold')

    # CONE radiation balance (right)
    ax = axes[1]
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.axis('off')

    # Crater floor
    floor = FancyBboxPatch((3, 0.5), 4, 0.5, boxstyle="round,pad=0.05",
                           edgecolor='black', facecolor='gray', linewidth=2)
    ax.add_patch(floor)
    ax.text(5, 0.75, 'CRATER FLOOR\n(Shadow)', ha='center', va='center',
           fontsize=10, fontweight='bold')

    # Incoming radiation arrows
    # Q_sky (from above - LARGE)
    f_sky_cone = 0.962
    Q_sky_cone = f_sky_cone * SIGMA_SB * 3**4
    ax.annotate('', xy=(5, 1.0), xytext=(5, 3.5),
                arrowprops=dict(arrowstyle='->', color='cyan', lw=6))
    ax.text(5.5, 2.2, f'$Q_{{sky}}$\n{Q_sky_cone:.2e} W/m²\n$F_{{sky}}={f_sky_cone:.3f}$',
           fontsize=9, color='cyan', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Q_thermal (from walls - SMALL)
    f_walls_cone = 0.038
    Q_thermal_cone = f_walls_cone * 0.95 * SIGMA_SB * T_wall**4
    ax.annotate('', xy=(4.5, 0.8), xytext=(2.5, 2.0),
                arrowprops=dict(arrowstyle='->', color='orange', lw=1))
    ax.text(2.0, 2.5, f'$Q_{{thermal}}$\n{Q_thermal_cone:.2f} W/m²\n$F_{{walls}}={f_walls_cone:.3f}$\n$T_{{wall}}={T_wall}$ K',
           fontsize=9, color='orange', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Q_scattered (scattered sunlight)
    Q_scattered_cone = 0.15  # W/m² (small)
    ax.annotate('', xy=(3.5, 0.9), xytext=(2.0, 4.0),
                arrowprops=dict(arrowstyle='->', color='yellow', lw=1))
    ax.text(2.0, 4.5, f'$Q_{{scattered}}$\n{Q_scattered_cone:.2f} W/m²',
           fontsize=9, color='orange', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Total Q in
    Q_total_cone = Q_sky_cone + Q_thermal_cone + Q_scattered_cone

    # Emitted radiation (out)
    T_shadow_cone = (Q_total_cone / (0.95 * SIGMA_SB))**0.25
    ax.annotate('', xy=(5, 6.5), xytext=(5, 1.0),
                arrowprops=dict(arrowstyle='->', color='red', lw=4))
    ax.text(5, 7.5, f'Emitted: εσT⁴\nT = {T_shadow_cone:.1f} K\nQ = {Q_total_cone:.2f} W/m²',
           ha='center', fontsize=10, color='red', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', lw=2))

    # Energy balance equation
    ax.text(5, 9, 'Cone: Small $F_{walls}$ → Small $Q_{thermal}$ → COLDER',
           ha='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    ax.set_title('B. CONE CRATER RADIATION BALANCE', fontsize=13, fontweight='bold')

    plt.suptitle('Figure 5: Radiation Balance Comparison (γ = 0.1)\n' +
                 f'ΔT = {T_shadow_cone:.1f} - {T_shadow_bowl:.1f} = {T_shadow_cone - T_shadow_bowl:.1f} K',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('/home/user/documents/fig5_radiation_balance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fig5_radiation_balance.png")
    plt.close()


def plot_temperature_comparison():
    """
    Figure 6: Shadow temperature as function of latitude and gamma.
    """
    from bowl_crater_thermal import CraterGeometry, ingersoll_crater_temperature
    from ingersol_cone_theory import InvConeGeometry, ingersol_cone_temperature

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Temperature vs Latitude
    ax = axes[0]

    latitudes = np.linspace(-70, -90, 30)
    gamma = 0.1
    D = 1000
    d = gamma * D
    T_sunlit = 200
    solar_e = 5.0

    T_bowl = []
    T_cone = []

    for lat in latitudes:
        bowl = CraterGeometry(D, d, lat)
        cone = InvConeGeometry(D, d, lat)

        bowl_temps = ingersoll_crater_temperature(bowl, T_sunlit, solar_e)
        cone_temps = ingersol_cone_temperature(cone, T_sunlit, solar_e)

        T_bowl.append(bowl_temps['T_shadow'])
        T_cone.append(cone_temps['T_shadow'])

    ax.plot(latitudes, T_bowl, 'b-', linewidth=3, marker='o', markersize=5, label='Bowl')
    ax.plot(latitudes, T_cone, 'r-', linewidth=3, marker='s', markersize=5, label='Cone')
    ax.axhline(y=110, color='orange', linestyle=':', linewidth=2, label='H₂O stability (110 K)')
    ax.axhline(y=80, color='green', linestyle=':', linewidth=2, label='CO₂ stability (80 K)')

    ax.fill_between(latitudes, 0, 80, color='green', alpha=0.1, label='CO₂ stable zone')
    ax.fill_between(latitudes, 80, 110, color='yellow', alpha=0.1, label='H₂O stable zone')

    ax.set_xlabel('Latitude (degrees)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Shadow Temperature (K)', fontsize=12, fontweight='bold')
    ax.set_title('A. Shadow Temperature vs Latitude (γ = 0.1)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([30, 150])

    # Panel B: Temperature vs gamma
    ax = axes[1]

    gamma_vals = np.linspace(0.05, 0.20, 30)
    lat = -85

    T_bowl_g = []
    T_cone_g = []

    for g in gamma_vals:
        d_g = g * D
        bowl = CraterGeometry(D, d_g, lat)
        cone = InvConeGeometry(D, d_g, lat)

        bowl_temps = ingersoll_crater_temperature(bowl, T_sunlit, solar_e)
        cone_temps = ingersol_cone_temperature(cone, T_sunlit, solar_e)

        T_bowl_g.append(bowl_temps['T_shadow'])
        T_cone_g.append(cone_temps['T_shadow'])

    ax.plot(gamma_vals, T_bowl_g, 'b-', linewidth=3, marker='o', markersize=5, label='Bowl')
    ax.plot(gamma_vals, T_cone_g, 'r-', linewidth=3, marker='s', markersize=5, label='Cone')
    ax.axhline(y=110, color='orange', linestyle=':', linewidth=2, label='H₂O stability (110 K)')
    ax.axhline(y=80, color='green', linestyle=':', linewidth=2, label='CO₂ stability (80 K)')

    ax.set_xlabel('Depth-to-diameter ratio (γ = d/D)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Shadow Temperature (K)', fontsize=12, fontweight='bold')
    ax.set_title('B. Shadow Temperature vs γ (Latitude 85°S)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([30, 150])

    plt.suptitle('Figure 6: Shadow Temperature Predictions\n' +
                 'Cone consistently colder than Bowl across all parameters',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('/home/user/documents/fig6_temperature_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fig6_temperature_comparison.png")
    plt.close()


def main():
    """Generate all theoretical comparison figures."""
    print("=" * 80)
    print("GENERATING THEORETICAL COMPARISON FIGURES")
    print("Bowl vs Cone Crater Framework")
    print("=" * 80)

    print("\nFigure 1: Crater geometry cross-sections...")
    plot_crater_geometry_comparison()

    print("\nFigure 2: View factor diagrams...")
    plot_view_factor_diagrams()

    print("\nFigure 3: View factor curves...")
    plot_view_factor_curves()

    print("\nFigure 4: Shadow geometry evolution...")
    plot_shadow_geometry()

    print("\nFigure 5: Radiation balance diagrams...")
    plot_radiation_balance()

    print("\nFigure 6: Temperature comparisons...")
    plot_temperature_comparison()

    print("\n" + "=" * 80)
    print("ALL FIGURES GENERATED")
    print("=" * 80)
    print("\nFiles created:")
    print("  - fig1_crater_geometry.png")
    print("  - fig2_view_factors.png")
    print("  - fig3_view_factor_curves.png")
    print("  - fig4_shadow_geometry.png")
    print("  - fig5_radiation_balance.png")
    print("  - fig6_temperature_comparison.png")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
