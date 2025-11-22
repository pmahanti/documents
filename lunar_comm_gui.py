"""
Lunar Communication Simulator - Web GUI

Comprehensive web-based interface for lunar surface communication analysis.
Deployable locally or on web servers.

Run with: streamlit run lunar_comm_gui.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import os
import sys
from datetime import datetime, timedelta

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation_engine import (
    LunarCommSimulationEngine,
    SimulationConfig,
    create_example_config
)
from output_manager import OutputManager
from propagation_models import list_available_models, get_model_description


# Page configuration
st.set_page_config(
    page_title="Lunar Communication Simulator",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application entry point."""

    # Header
    st.markdown('<h1 class="main-header">🌙 Lunar Communication Simulator 📡</h1>', unsafe_allow_html=True)
    st.markdown("**Comprehensive RF Coverage Analysis for Lunar Surface Communications**")

    # Sidebar - Scenario Selection
    st.sidebar.title("Configuration")

    scenario = st.sidebar.selectbox(
        "Select Scenario (ConOps)",
        [
            "Surface TX → Surface RX",
            "Surface TX → Earth RX (DTE)",
            "Crater TX → Earth RX (DTE)",
            "Rover Path → Earth RX (DTE)"
        ],
        help="Choose the communication scenario to analyze"
    )

    # Map scenario names
    scenario_map = {
        "Surface TX → Surface RX": "surface_to_surface",
        "Surface TX → Earth RX (DTE)": "surface_to_earth",
        "Crater TX → Earth RX (DTE)": "crater_to_earth",
        "Rover Path → Earth RX (DTE)": "rover_path_dte"
    }

    scenario_key = scenario_map[scenario]

    # Initialize session state
    if 'simulation_run' not in st.session_state:
        st.session_state.simulation_run = False
    if 'results' not in st.session_state:
        st.session_state.results = None

    # Main content tabs
    tabs = st.tabs(["📋 Setup", "▶️ Run Simulation", "📊 Results", "📖 Help"])

    # ========== SETUP TAB ==========
    with tabs[0]:
        st.markdown('<h2 class="sub-header">Simulation Setup</h2>', unsafe_allow_html=True)

        config = setup_simulation_parameters(scenario_key)

        # Store config in session state
        st.session_state.config = config

        # Show configuration summary
        st.markdown("### Configuration Summary")
        col1, col2 = st.columns(2)

        with col1:
            st.info(f"**Scenario:** {scenario}")
            st.info(f"**TX Location:** {config.tx_lat:.4f}°, {config.tx_lon:.4f}°")
            st.info(f"**TX Height:** {config.tx_height_m} m")

        with col2:
            st.info(f"**Frequency:** {config.frequency_mhz} MHz")
            st.info(f"**TX Power:** {config.tx_power_dbm} dBm")
            st.info(f"**Propagation Model:** {config.propagation_model}")

    # ========== RUN SIMULATION TAB ==========
    with tabs[1]:
        st.markdown('<h2 class="sub-header">Run Simulation</h2>', unsafe_allow_html=True)

        if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
            run_simulation(st.session_state.config)

        if st.session_state.simulation_run:
            st.success("✅ Simulation completed successfully!")

    # ========== RESULTS TAB ==========
    with tabs[2]:
        st.markdown('<h2 class="sub-header">Simulation Results</h2>', unsafe_allow_html=True)

        if st.session_state.results is not None:
            display_results(st.session_state.results, scenario_key)
        else:
            st.warning("⚠️ No results available. Please run a simulation first.")

    # ========== HELP TAB ==========
    with tabs[3]:
        display_help(scenario_key)


def setup_simulation_parameters(scenario: str) -> SimulationConfig:
    """
    Setup simulation parameters based on scenario.

    Args:
        scenario: Scenario identifier

    Returns:
        SimulationConfig object
    """
    # Start with example config
    config = create_example_config(scenario)

    st.markdown("### Transmitter Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        tx_lat = st.number_input(
            "TX Latitude (°)",
            min_value=-90.0,
            max_value=90.0,
            value=float(config.tx_lat),
            step=0.1,
            help="Transmitter latitude in degrees"
        )

        tx_lon = st.number_input(
            "TX Longitude (°)",
            min_value=-180.0,
            max_value=180.0,
            value=float(config.tx_lon),
            step=0.1,
            help="Transmitter longitude in degrees"
        )

    with col2:
        tx_height = st.number_input(
            "TX Height (m)",
            min_value=0.1,
            max_value=100.0,
            value=float(config.tx_height_m),
            step=0.5,
            help="Transmitter antenna height above ground"
        )

        frequency = st.number_input(
            "Frequency (MHz)",
            min_value=100.0,
            max_value=30000.0,
            value=float(config.frequency_mhz),
            step=100.0,
            help="Operating frequency in MHz"
        )

    with col3:
        tx_power = st.number_input(
            "TX Power (dBm)",
            min_value=0.0,
            max_value=60.0,
            value=float(config.tx_power_dbm),
            step=1.0,
            help="Transmitter power in dBm"
        )

        tx_gain = st.number_input(
            "TX Antenna Gain (dBi)",
            min_value=0.0,
            max_value=50.0,
            value=float(config.tx_gain_dbi),
            step=1.0,
            help="Transmitter antenna gain in dBi"
        )

    # Propagation Model Selection
    st.markdown("### Propagation Model")

    models = list_available_models()
    model_options = {m: get_model_description(m) for m in models}

    selected_model = st.selectbox(
        "Select Propagation Model",
        options=list(model_options.keys()),
        format_func=lambda x: f"{x}: {model_options[x]}",
        index=list(model_options.keys()).index(config.propagation_model) if config.propagation_model in model_options else 0
    )

    col1, col2 = st.columns(2)

    with col1:
        include_multipath = st.checkbox(
            "Include Multipath Effects",
            value=config.include_multipath,
            help="Include ground reflection multipath (two-ray model)"
        )

    with col2:
        include_diffraction = st.checkbox(
            "Include Diffraction",
            value=config.include_diffraction,
            help="Include knife-edge diffraction over obstacles"
        )

    surface_roughness = st.slider(
        "Surface Roughness (m)",
        min_value=0.0,
        max_value=1.0,
        value=float(config.surface_roughness_m),
        step=0.01,
        help="RMS surface roughness for scattering calculations"
    )

    # Scenario-specific parameters
    if scenario == 'surface_to_surface':
        st.markdown("### Surface-to-Surface Parameters")

        col1, col2 = st.columns(2)

        with col1:
            analysis_range = st.number_input(
                "Analysis Range (km)",
                min_value=1.0,
                max_value=100.0,
                value=float(config.analysis_range_km),
                step=1.0
            )

            grid_resolution = st.number_input(
                "Grid Resolution (m)",
                min_value=10.0,
                max_value=500.0,
                value=float(config.grid_resolution_m),
                step=10.0
            )

        with col2:
            rx_sensitivity = st.number_input(
                "RX Sensitivity (dBm)",
                min_value=-150.0,
                max_value=-50.0,
                value=float(config.rx_sensitivity_dbm),
                step=1.0
            )

        # Surface assets
        st.markdown("#### Surface Assets")

        num_assets = st.number_input("Number of Surface Assets", min_value=0, max_value=10, value=len(config.surface_assets))

        surface_assets = []
        for i in range(num_assets):
            with st.expander(f"Asset {i+1}"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    asset_name = st.text_input(f"Name", value=f"Asset-{i+1}", key=f"asset_name_{i}")
                    asset_lat = st.number_input(f"Latitude (°)", value=-89.0, step=0.1, key=f"asset_lat_{i}")

                with col2:
                    asset_lon = st.number_input(f"Longitude (°)", value=0.0, step=0.1, key=f"asset_lon_{i}")
                    asset_alt = st.number_input(f"Altitude (m)", value=2.0, step=0.1, key=f"asset_alt_{i}")

                with col3:
                    asset_rx_sens = st.number_input(f"RX Sens (dBm)", value=-110.0, step=1.0, key=f"asset_rx_{i}")
                    asset_gain = st.number_input(f"Antenna Gain (dBi)", value=8.0, step=0.5, key=f"asset_gain_{i}")

                surface_assets.append({
                    'name': asset_name,
                    'lat': asset_lat,
                    'lon': asset_lon,
                    'altitude': asset_alt,
                    'rx_sensitivity': asset_rx_sens,
                    'antenna_gain': asset_gain
                })

        config.surface_assets = surface_assets
        config.analysis_range_km = analysis_range
        config.grid_resolution_m = grid_resolution
        config.rx_sensitivity_dbm = rx_sensitivity

    elif scenario in ['surface_to_earth', 'crater_to_earth']:
        st.markdown("### Direct-to-Earth Parameters")

        col1, col2 = st.columns(2)

        with col1:
            dte_start = st.date_input("Start Date", value=datetime(2026, 1, 1))
            dte_start_time = st.time_input("Start Time (UTC)", value=datetime.strptime("00:00", "%H:%M").time())

            dte_datetime = datetime.combine(dte_start, dte_start_time)
            config.dte_start_time = dte_datetime.isoformat()

            dte_duration = st.number_input(
                "Duration (hours)",
                min_value=1.0,
                max_value=720.0,
                value=float(config.dte_duration_hours),
                step=12.0
            )

        with col2:
            dte_freq = st.number_input(
                "DTE Frequency (MHz)",
                min_value=1000.0,
                max_value=40000.0,
                value=float(config.dte_frequency_mhz),
                step=100.0,
                help="Typical: 8450 MHz for X-band"
            )

            dte_power = st.number_input(
                "DTE TX Power (dBm)",
                min_value=30.0,
                max_value=60.0,
                value=float(config.dte_tx_power_dbm),
                step=1.0
            )

            dte_gain = st.number_input(
                "DTE TX Gain (dBi)",
                min_value=10.0,
                max_value=50.0,
                value=float(config.dte_tx_gain_dbi),
                step=1.0
            )

        config.dte_duration_hours = dte_duration
        config.dte_frequency_mhz = dte_freq
        config.dte_tx_power_dbm = dte_power
        config.dte_tx_gain_dbi = dte_gain

        if scenario == 'crater_to_earth':
            st.markdown("#### Crater Parameters")

            col1, col2 = st.columns(2)

            with col1:
                crater_depth = st.number_input(
                    "Crater Depth (m)",
                    min_value=10.0,
                    max_value=5000.0,
                    value=float(config.crater_depth_m),
                    step=10.0
                )

                crater_radius = st.number_input(
                    "Crater Radius (m)",
                    min_value=50.0,
                    max_value=10000.0,
                    value=float(config.crater_radius_m),
                    step=50.0
                )

            with col2:
                tx_inside = st.checkbox(
                    "TX Inside Crater",
                    value=config.tx_inside_crater
                )

            config.crater_depth_m = crater_depth
            config.crater_radius_m = crater_radius
            config.tx_inside_crater = tx_inside

    elif scenario == 'rover_path_dte':
        st.markdown("### Rover Path Parameters")

        col1, col2 = st.columns(2)

        with col1:
            rover_speed = st.number_input(
                "Rover Speed (km/h)",
                min_value=0.1,
                max_value=10.0,
                value=float(config.rover_speed_kmh),
                step=0.1
            )

            mission_hours = st.number_input(
                "Mission Duration (hours)",
                min_value=1.0,
                max_value=720.0,
                value=float(config.rover_mission_hours),
                step=1.0
            )

        with col2:
            dte_freq = st.number_input(
                "DTE Frequency (MHz)",
                min_value=1000.0,
                max_value=40000.0,
                value=8450.0,
                step=100.0
            )

            dte_power = st.number_input(
                "DTE TX Power (dBm)",
                min_value=30.0,
                max_value=60.0,
                value=43.0,
                step=1.0
            )

        # Waypoints
        st.markdown("#### Rover Waypoints")

        num_waypoints = st.number_input("Number of Waypoints", min_value=2, max_value=20, value=max(2, len(config.rover_waypoints)))

        waypoints = []
        for i in range(num_waypoints):
            col1, col2 = st.columns(2)
            with col1:
                wp_lat = st.number_input(f"WP{i+1} Lat", value=-89.5 if i == 0 else -89.5 - i*0.05, step=0.01, key=f"wp_lat_{i}")
            with col2:
                wp_lon = st.number_input(f"WP{i+1} Lon", value=0.0 if i == 0 else i*5.0, step=0.1, key=f"wp_lon_{i}")

            waypoints.append((wp_lat, wp_lon))

        config.rover_waypoints = waypoints
        config.rover_speed_kmh = rover_speed
        config.rover_mission_hours = mission_hours
        config.dte_frequency_mhz = dte_freq
        config.dte_tx_power_dbm = dte_power

    # Update config
    config.tx_lat = tx_lat
    config.tx_lon = tx_lon
    config.tx_height_m = tx_height
    config.frequency_mhz = frequency
    config.tx_power_dbm = tx_power
    config.tx_gain_dbi = tx_gain
    config.propagation_model = selected_model
    config.include_multipath = include_multipath
    config.include_diffraction = include_diffraction
    config.surface_roughness_m = surface_roughness

    return config


def run_simulation(config: SimulationConfig):
    """
    Run the simulation.

    Args:
        config: Simulation configuration
    """
    with st.spinner("Running simulation... This may take a few moments."):
        try:
            # Create engine
            engine = LunarCommSimulationEngine(config)

            # Run simulation
            results = engine.run_simulation()

            # Store results
            st.session_state.results = results
            st.session_state.simulation_run = True

            st.success(f"✅ Simulation completed! Status: {engine.status}")

        except Exception as e:
            st.error(f"❌ Simulation failed: {str(e)}")
            st.session_state.simulation_run = False


def display_results(results: Dict, scenario: str):
    """
    Display simulation results.

    Args:
        results: Simulation results dictionary
        scenario: Scenario identifier
    """
    # Statistics
    st.markdown("### Summary Statistics")

    if scenario == 'surface_to_surface':
        stats = results.get('statistics', {})

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Coverage", f"{stats.get('coverage_percentage', 0):.1f}%")

        with col2:
            st.metric("Max Range", f"{stats.get('max_covered_range_km', 0):.2f} km")

        with col3:
            st.metric("Covered Area", f"{stats.get('coverage_area_km2', 0):.2f} km²")

        # Coverage map
        st.markdown("### Coverage Map")

        coverage_data = results.get('coverage_data', {})
        if coverage_data:
            X = coverage_data.get('X', np.array([[0]]))
            Y = coverage_data.get('Y', np.array([[0]]))
            rx_power = coverage_data.get('received_power_dbm', np.array([[0]]))

            # Create plotly figure
            rx_power_plot = np.copy(rx_power)
            rx_power_plot[np.isinf(rx_power_plot)] = np.nan

            fig = go.Figure(data=go.Heatmap(
                x=X[0]/1000,
                y=Y[:,0]/1000,
                z=rx_power_plot,
                colorscale='RdYlGn',
                colorbar=dict(title="Power (dBm)")
            ))

            fig.update_layout(
                title="Received Signal Strength",
                xaxis_title="X (km)",
                yaxis_title="Y (km)",
                height=600
            )

            st.plotly_chart(fig, use_container_width=True)

        # Asset links
        asset_links = results.get('asset_links', [])
        if asset_links:
            st.markdown("### Surface Asset Links")

            df = pd.DataFrame(asset_links)
            st.dataframe(df, use_container_width=True)

    elif scenario in ['surface_to_earth', 'crater_to_earth']:
        visibility = results.get('visibility', {})

        if visibility:
            windows = visibility.get('windows', [])

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Visibility Windows", len(windows))

            with col2:
                if windows:
                    total_hours = sum([w['duration_hours'] for w in windows])
                    st.metric("Total Visible Time", f"{total_hours:.1f} hrs")

            with col3:
                if windows:
                    avg_duration = np.mean([w['duration_hours'] for w in windows])
                    st.metric("Avg Window", f"{avg_duration:.1f} hrs")

            # Plot visibility
            st.markdown("### Earth Visibility Timeline")

            et_array = np.array(visibility.get('et_array', []))
            vis_array = np.array(visibility.get('visibility', []))

            if len(et_array) > 0:
                hours = (et_array - et_array[0]) / 3600

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hours,
                    y=vis_array,
                    fill='tozeroy',
                    name='Visibility'
                ))

                fig.update_layout(
                    title="Earth Visibility",
                    xaxis_title="Time (hours)",
                    yaxis_title="Visible",
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

        # DSN links
        dsn_links = results.get('dsn_links', {})
        if dsn_links:
            st.markdown("### DSN Station Links")

            dsn_df = pd.DataFrame([
                {
                    'Station': name,
                    'Distance (km)': f"{link['distance_km']:,.0f}",
                    'Path Loss (dB)': f"{link['fspl_db']:.1f}",
                    'RX Power (dBm)': f"{link['rx_power_dbm']:.1f}",
                    'Link Margin (dB)': f"{link['link_margin_db']:+.1f}",
                    'Available': '✅' if link['link_available'] else '❌'
                }
                for name, link in dsn_links.items()
            ])

            st.dataframe(dsn_df, use_container_width=True)

    elif scenario == 'rover_path_dte':
        summary = results.get('summary', {})

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Mission Duration", f"{summary.get('total_minutes', 0)} min")

        with col2:
            st.metric("Earth Visible", f"{summary.get('earth_visible_percent', 0):.1f}%")

        with col3:
            st.metric("DSN Available", f"{summary.get('any_dsn_percent', 0):.1f}%")

        with col4:
            st.metric("Best Margin", f"{summary.get('best_margin_overall_db', 0):.1f} dB")

        # Coverage timeline
        st.markdown("### Coverage Timeline")

        coverage_records = results.get('coverage_data', [])
        if coverage_records:
            df = pd.DataFrame(coverage_records)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                y=df['earth_visible'],
                mode='lines',
                fill='tozeroy',
                name='Earth Visible'
            ))

            fig.update_layout(
                title="Earth Visibility During Mission",
                xaxis_title="Time (minutes)",
                yaxis_title="Visible",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

    # Download outputs
    st.markdown("### Download Outputs")

    output_mgr = OutputManager()

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Generate PNG", use_container_width=True):
            with st.spinner("Generating PNG..."):
                try:
                    files = output_mgr.generate_all_outputs(results, formats=['png'])
                    if 'png' in files:
                        st.success(f"✅ PNG saved: {files['png']}")
                        with open(files['png'], 'rb') as f:
                            st.download_button(
                                "📥 Download PNG",
                                f,
                                file_name=os.path.basename(files['png']),
                                mime="image/png"
                            )
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        if st.button("Generate GeoTIFF", use_container_width=True):
            if scenario == 'surface_to_surface':
                with st.spinner("Generating GeoTIFF..."):
                    try:
                        files = output_mgr.generate_all_outputs(results, formats=['geotiff'])
                        if 'geotiff_power' in files:
                            st.success(f"✅ GeoTIFF saved: {files['geotiff_power']}")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.info("GeoTIFF only available for surface-to-surface scenario")

    with col3:
        if st.button("Generate CSV", use_container_width=True):
            with st.spinner("Generating CSV..."):
                try:
                    files = output_mgr.generate_all_outputs(results, formats=['csv'])
                    if 'csv' in files:
                        st.success(f"✅ CSV saved: {files['csv']}")
                        with open(files['csv'], 'rb') as f:
                            st.download_button(
                                "📥 Download CSV",
                                f,
                                file_name=os.path.basename(files['csv']),
                                mime="text/csv"
                            )
                except Exception as e:
                    st.error(f"Error: {e}")


def display_help(scenario: str):
    """Display help information."""

    st.markdown("## User Guide")

    st.markdown("""
    ### Quick Start

    1. **Select Scenario**: Choose your communication scenario from the sidebar
    2. **Configure Parameters**: Adjust transmitter and propagation settings in the Setup tab
    3. **Run Simulation**: Click the "Run Simulation" button
    4. **View Results**: Examine coverage maps, statistics, and link budgets
    5. **Download**: Generate and download PNG, GeoTIFF, or CSV outputs

    ### Scenarios

    #### Surface TX → Surface RX
    Analyzes surface-to-surface LTE communication with terrain effects.
    - Useful for: Rover-to-base, rover-to-rover communications
    - Includes: Multipath, diffraction, line-of-sight analysis
    - Outputs: Coverage maps, asset link budgets

    #### Surface TX → Earth RX (DTE)
    Analyzes Direct-to-Earth communication via DSN.
    - Useful for: Telemetry downlink, science data transmission
    - Includes: Earth visibility windows, DSN link budgets
    - Outputs: Visibility timeline, station availability

    #### Crater TX → Earth RX (DTE)
    Same as Surface→Earth but accounts for crater diffraction.
    - Useful for: Communication from permanently shadowed regions
    - Includes: Crater rim diffraction effects
    - Outputs: Adjusted link margins with crater effects

    #### Rover Path → Earth RX (DTE)
    Analyzes DTE coverage as rover traverses planned path.
    - Useful for: Mission planning, data transfer scheduling
    - Includes: Minute-by-minute DSN availability
    - Outputs: CSV with coverage timeline, path visualization

    ### Propagation Models

    - **Free Space**: Ideal vacuum propagation (baseline)
    - **Two-Ray**: Includes ground reflection multipath (recommended)
    - **Knife-Edge**: Single obstacle diffraction
    - **Crater**: Crater rim diffraction
    - **Plane Earth**: Far-field approximation
    - **Egli**: Modified for lunar terrain
    - **Longley-Rice**: Irregular terrain model
    - **COST-231**: Empirical model adapted for Moon

    ### Parameter Ranges

    - **Frequency**: 100-30,000 MHz (VHF to Ka-band)
    - **TX Power**: 0-60 dBm (1 mW to 1 kW)
    - **Antenna Gain**: 0-50 dBi
    - **Range**: 1-100 km (surface-to-surface)
    - **Mission Duration**: 1-720 hours (DTE)

    ### Tips

    - **S-band (2-4 GHz)**: Good for surface mobility
    - **X-band (8-12 GHz)**: Standard for DTE
    - **Multipath**: Enable for realistic surface scenarios
    - **Resolution**: Lower for faster preview, higher for final analysis
    - **Assets**: Add rovers/landers to check specific links

    ### Outputs

    - **PNG**: Visual coverage maps and timelines
    - **GeoTIFF**: Georeferenced raster (surface scenarios)
    - **CSV**: Tabular data for further analysis
    """)

    st.markdown("---")
    st.markdown("**Version:** 1.0.0 | **Contact:** See repository for support")


if __name__ == "__main__":
    main()
