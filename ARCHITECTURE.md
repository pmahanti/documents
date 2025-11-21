# Small Lunar Crater (SLC) Analysis Pipeline - Architecture Block Diagram

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        USER ENTRY POINTS (CLI Tools)                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    ▼                  ▼                  ▼
        ┌───────────────────┐ ┌──────────────────┐ ┌────────────────────┐
        │ STANDALONE TOOLS  │ │ STANDALONE TOOLS │ │  COMPLETE PIPELINE │
        └───────────────────┘ └──────────────────┘ └────────────────────┘


┌═══════════════════════════════════════════════════════════════════════════════┐
║                      HIERARCHICAL BLOCK DIAGRAM                                ║
║                                                                                ║
║  Legend:                                                                       ║
║  ┌─────┐                                                                       ║
║  │ CLI │  = Command-line interface (user-facing)                              ║
║  └─────┘                                                                       ║
║  ┌─────┐                                                                       ║
║  │ MOD │  = Module (library code)                                             ║
║  └─────┘                                                                       ║
║  [file] = Data file (input/output)                                            ║
║  ───────> = Function call / import dependency                                 ║
║  =======> = Data flow                                                         ║
╚═══════════════════════════════════════════════════════════════════════════════╝


┌─────────────────────────────────────────────────────────────────────────────┐
│                          COMPLETE PIPELINE (Main Entry)                      │
└─────────────────────────────────────────────────────────────────────────────┘

                    analyze_small_lunar_craters.py (CLI)
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
                    ▼              ▼              ▼
    ┌───────────────────┐  ┌──────────────┐  ┌──────────────────┐
    │ run_step0_input_  │  │ run_step2_   │  │ run_step3_       │
    │   processing()    │  │ rim_refine() │  │ morphometry()    │
    └───────────────────┘  └──────────────┘  └──────────────────┘
            │                      │                   │
            │ imports              │ imports           │ imports
            ▼                      ▼                   ▼
    input_module.py        refine_crater_rim.py  analyze_morphometry.py
    (from src/)            (from src/)            (from src/)
                                   │
                                   │ calls
                                   ▼
                    ┌────────────────────────────┐
                    │ generate_comprehensive_    │
                    │        report()            │
                    └────────────────────────────┘
                                   │
                                   │ imports
                                   ▼
                            report_generator.py
                            (from src/)


┌─────────────────────────────────────────────────────────────────────────────┐
│                     STEP 0: INPUT PROCESSING (Standalone)                    │
└─────────────────────────────────────────────────────────────────────────────┘

  [image.tif]  [dtm.tif]  [craters.csv/.diam]
       │            │              │
       └────────────┼──────────────┘
                    │
                    ▼
        process_crater_inputs.py (CLI)
                    │
                    │ imports & calls
                    ▼
        ┌───────────────────────────┐
        │ input_module.py           │ (src/crater_analysis/)
        │                           │
        │ Functions:                │
        │ • read_crater_file()      │────> CoordinateConverter class
        │ • read_isis_cube()        │        │
        │ • process_crater_inputs() │        ├─> latlon_to_xy()
        │ • create_crater_shapefile()│       └─> xy_to_latlon()
        │ • plot_crater_locations() │
        │ • plot_csfd()             │
        └───────────────────────────┘
                    │
                    │ imports
                    ▼
        ┌───────────────────────────┐
        │ config.py                 │ (src/crater_analysis/)
        │                           │
        │ • Config class            │────> reads config/regions.json
        │ • get_dem_path()          │
        │ • get_orthophoto_path()   │
        └───────────────────────────┘
                    │
                    │ outputs
                    ▼
        ══════════════════════════════
        [craters_initial.shp]  (shapefile)
        [crater_locations.png]  (figure)
        [csfd_initial.png]      (figure)
        ══════════════════════════════


┌─────────────────────────────────────────────────────────────────────────────┐
│                     STEP 2: RIM REFINEMENT (Standalone)                      │
└─────────────────────────────────────────────────────────────────────────────┘

  [craters_initial.shp]  [image.tif]  [dtm.tif]
            │                  │            │
            └──────────────────┼────────────┘
                               │
                               ▼
              refine_crater_rims.py (CLI)
                               │
                               │ imports & calls
                               ▼
        ┌─────────────────────────────────────┐
        │ refine_crater_rim.py                │ (src/crater_analysis/)
        │                                     │
        │ Functions:                          │
        │ • compute_edge_strength()           │───> cv2.Canny()
        │ • compute_topographic_quality()     │        (edge detection)
        │ • compute_rim_probability()         │
        │ • refine_single_crater()            │
        │ • refine_crater_rims()              │────┐
        │ • plot_refined_positions()          │    │
        │ • plot_csfd_refined()               │    │
        │ • plot_rim_differences()            │    │
        └─────────────────────────────────────┘    │
                               │                    │
                               │ imports            │
                               ▼                    │
        ┌─────────────────────────────────────┐    │
        │ cratools.py                         │<───┘
        │                                     │ (src/crater_analysis/)
        │ Core Functions:                    │
        │ • fit_crater_rim()                 │───> Topographic rim detection
        │ • remove_external_topography()     │       (72 azimuth sampling)
        │ • compute_E_matrix()               │
        │ • compute_depth_diameter_ratio()   │
        └─────────────────────────────────────┘
                               │
                               │ outputs
                               ▼
        ════════════════════════════════════════
        [craters_refined.shp]           (shapefile with rim_probability)
        [craters_refined_positions.png] (figure)
        [craters_refined_csfd.png]      (figure)
        [craters_rim_differences.png]   (figure)
        ════════════════════════════════════════


┌─────────────────────────────────────────────────────────────────────────────┐
│                   STEP 3: MORPHOMETRY ANALYSIS (Standalone)                  │
└─────────────────────────────────────────────────────────────────────────────┘

  [craters_refined.shp]  [dtm.tif]  [image.tif]
            │                  │          │
            └──────────────────┼──────────┘
                               │
                               ▼
          analyze_crater_morphometry.py (CLI)
                               │
                               │ imports & calls
                               ▼
        ┌──────────────────────────────────────────┐
        │ analyze_morphometry.py                   │ (src/crater_analysis/)
        │                                          │
        │ Functions:                               │
        │ • gaussian_2d()                          │───> 2D Gaussian model
        │ • fit_gaussian_floor()                   │       (7 parameters)
        │ • compute_morphometry_dual_method()      │            │
        │ • analyze_crater_morphometry()           │            │
        │ • extract_morphometry_fields()           │            │
        │ • plot_morphometry_scatter()             │            │
        │ • plot_probability_distributions()       │            │
        │ • compute_conditional_probabilities()    │            │
        │ • compute_summary_statistics()           │            │
        └──────────────────────────────────────────┘            │
                               │                                │
                               │ imports                        │
                               ▼                                │
        ┌──────────────────────────────────────────┐            │
        │ cratools.py                              │<───────────┘
        │                                          │
        │ Called for Method 1:                    │
        │ • compute_depth_diameter_ratio()        │───> Rim perimeter method
        │   - Detects rim as perimeter pixels    │       (existing proven algorithm)
        │   - Computes mean rim elevation        │
        │   - Finds floor minimum                │
        │   - Returns depth with uncertainty     │
        └──────────────────────────────────────────┘
                               │
                               │ outputs
                               ▼
        ════════════════════════════════════════════
        [craters_morphometry.shp]           (shapefile with 20+ fields)
        [morphometry_scatter_plots.png]     (figure - 2 panels)
        [probability_distributions.png]     (figure - 2 panels)
        [morphometry_data.csv]              (all measurements)
        [conditional_probability.csv]       (P(d|D) and P(D|d))
        ════════════════════════════════════════════


┌─────────────────────────────────────────────────────────────────────────────┐
│                         PDF REPORT GENERATION                                │
└─────────────────────────────────────────────────────────────────────────────┘

  [All outputs from Steps 0, 2, 3]
            │
            │ read by
            ▼
  ┌─────────────────────────────────────────┐
  │ report_generator.py                     │ (src/crater_analysis/)
  │                                         │
  │ Functions:                              │
  │ • generate_pdf_report()                 │───> Main orchestration
  │ • create_first_page_image()             │       │
  │ • create_title_summary_page()           │       ├─> Page 1: Image + craters
  │ • create_results_page()                 │       ├─> Page 2: Title/summary
  │ • create_single_figure_page()           │       ├─> Pages 3-9: Results
  │ • create_error_propagation_page()       │       ├─> Page 10: Error theory
  │ • create_error_application_page()       │       ├─> Page 11: Error application
  │ • create_conditional_probability_page() │       ├─> Page 12: Cond. prob.
  │ • create_summary_page()                 │       └─> Page 13: Summary
  └─────────────────────────────────────────┘
            │
            │ uses
            ▼
  ┌─────────────────────────────────────────┐
  │ matplotlib.backends.backend_pdf         │
  │                                         │
  │ • PdfPages                              │───> Multi-page PDF
  │ • Figure layout (8.5" x 11")           │       0.5" margins
  │ • Text rendering (12pt)                │       Professional formatting
  └─────────────────────────────────────────┘
            │
            │ outputs
            ▼
  ════════════════════════════════════════════
  [SLC_Analysis_Report.pdf]  (10-13 pages)
  ════════════════════════════════════════════


┌─────────────────────────────────────────────────────────────────────────────┐
│                        SUPPORTING MODULES                                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────┐
│ config.py                      │ (src/crater_analysis/)
│                                │
│ • Config class                 │───> Reads config/regions.json
│ • get_dem_path()               │       │
│ • get_orthophoto_path()        │       ├─> DEM paths
│ • get_shapefile_path()         │       ├─> Image paths
│ • get_min_diameter()           │       └─> Region configs
└────────────────────────────────┘

┌────────────────────────────────┐
│ cratools.py                    │ (src/crater_analysis/)
│                                │
│ Core crater analysis tools     │
│ (shared by all steps):         │
│                                │
│ • fit_crater_rim()             │───> Topographic rim fitting
│   - 72 azimuth sampling        │       (used by Step 2)
│   - Peak detection             │
│   - Circle fitting             │
│                                │
│ • remove_external_topography() │───> Regional slope removal
│   - Plane fitting              │       (used by Steps 2 & 3)
│   - Detrending                 │
│                                │
│ • compute_depth_diameter_ratio()│──> Depth measurement
│   - Rim perimeter method       │       (used by Step 3)
│   - Floor minimum detection    │
│   - Uncertainty propagation    │
│                                │
│ • compute_E_matrix()           │───> Covariance matrix
│                                │       (error estimation)
└────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW DIAGRAM                                  │
└─────────────────────────────────────────────────────────────────────────────┘

USER INPUT FILES:
  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐
  │  image.tif   │  │   dtm.tif    │  │ craters.csv/.diam   │
  └──────┬───────┘  └──────┬───────┘  └──────────┬──────────┘
         │                 │                      │
         └─────────────────┼──────────────────────┘
                           │
                           ▼
    ╔═══════════════════════════════════════════════════════╗
    ║              STEP 0: INPUT PROCESSING                 ║
    ║           process_crater_inputs.py (CLI)              ║
    ║              input_module.py (MOD)                    ║
    ╚═══════════════════════════════════════════════════════╝
                           │
                           │ produces
                           ▼
    ┌──────────────────────────────────────────────────────┐
    │  craters_initial.shp  (N craters, lat/lon, diameter) │
    │  crater_locations.png                                 │
    │  csfd_initial.png                                     │
    └──────────────┬───────────────────────────────────────┘
                   │
                   │ input to
                   ▼
    ╔═══════════════════════════════════════════════════════╗
    ║              STEP 2: RIM REFINEMENT                   ║
    ║             refine_crater_rims.py (CLI)               ║
    ║            refine_crater_rim.py (MOD)                 ║
    ║              + cratools.py (MOD)                      ║
    ╚═══════════════════════════════════════════════════════╝
                   │
                   │ produces
                   ▼
    ┌──────────────────────────────────────────────────────┐
    │  craters_refined.shp  (+ rim_probability, err_r,     │
    │                          topo_quality, edge_strength) │
    │  craters_refined_positions.png                        │
    │  craters_refined_csfd.png                             │
    │  craters_rim_differences.png                          │
    └──────────────┬───────────────────────────────────────┘
                   │
                   │ input to
                   ▼
    ╔═══════════════════════════════════════════════════════╗
    ║           STEP 3: MORPHOMETRY ANALYSIS                ║
    ║         analyze_crater_morphometry.py (CLI)           ║
    ║           analyze_morphometry.py (MOD)                ║
    ║              + cratools.py (MOD)                      ║
    ╚═══════════════════════════════════════════════════════╝
                   │
                   │ produces
                   ▼
    ┌──────────────────────────────────────────────────────┐
    │  craters_morphometry.shp  (+ depth_m1, depth_m2,     │
    │                              d_D_m1, d_D_m2,          │
    │                              total_error_m1/m2,       │
    │                              fit_quality_m2)          │
    │  morphometry_scatter_plots.png                        │
    │  probability_distributions.png                        │
    │  morphometry_data.csv                                 │
    │  conditional_probability.csv                          │
    └──────────────┬───────────────────────────────────────┘
                   │
                   │ all inputs to
                   ▼
    ╔═══════════════════════════════════════════════════════╗
    ║              PDF REPORT GENERATION                    ║
    ║            report_generator.py (MOD)                  ║
    ╚═══════════════════════════════════════════════════════╝
                   │
                   │ produces
                   ▼
    ┌──────────────────────────────────────────────────────┐
    │  SLC_Analysis_Report.pdf  (10-13 pages)              │
    │    - Page 1: Contrast image with craters             │
    │    - Pages 2-9: All results and figures              │
    │    - Page 10: Error propagation theory               │
    │    - Page 11: Error propagation application          │
    │    - Page 12: Conditional probability table          │
    │    - Page 13: Summary and conclusions                │
    └──────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                         MODULE DEPENDENCY GRAPH                              │
└─────────────────────────────────────────────────────────────────────────────┘

CLI Tools (Root):
┌─────────────────────────────────────┐
│ analyze_small_lunar_craters.py      │────┐
└─────────────────────────────────────┘    │
                                           │ orchestrates
┌─────────────────────────────────────┐    │
│ process_crater_inputs.py            │<───┤
└─────────────────────────────────────┘    │
         │                                 │
         │ imports                         │
         ▼                                 │
┌─────────────────────────────────────┐    │
│ input_module.py                     │    │
└─────────────────────────────────────┘    │
         │                                 │
         │ imports                         │
         ▼                                 │
┌─────────────────────────────────────┐    │
│ config.py                           │    │
└─────────────────────────────────────┘    │
                                           │
┌─────────────────────────────────────┐    │
│ refine_crater_rims.py               │<───┤
└─────────────────────────────────────┘    │
         │                                 │
         │ imports                         │
         ▼                                 │
┌─────────────────────────────────────┐    │
│ refine_crater_rim.py                │    │
└─────────────────────────────────────┘    │
         │                                 │
         │ imports                         │
         ▼                                 │
┌─────────────────────────────────────┐    │
│ cratools.py                         │<───┼───┐
└─────────────────────────────────────┘    │   │ (shared core)
                                           │   │
┌─────────────────────────────────────┐    │   │
│ analyze_crater_morphometry.py       │<───┤   │
└─────────────────────────────────────┘    │   │
         │                                 │   │
         │ imports                         │   │
         ▼                                 │   │
┌─────────────────────────────────────┐    │   │
│ analyze_morphometry.py              │────┘   │
└─────────────────────────────────────┘        │
         │                                     │
         │ imports                             │
         └─────────────────────────────────────┘

Report Generation:
┌─────────────────────────────────────┐
│ report_generator.py                 │
└─────────────────────────────────────┘
         │
         │ reads outputs from all steps
         │ (no code dependencies, data only)
         ▼
    [All PNG figures]
    [All shapefiles]
    [All CSV files]


┌─────────────────────────────────────────────────────────────────────────────┐
│                      EXTERNAL DEPENDENCIES                                   │
└─────────────────────────────────────────────────────────────────────────────┘

requirements.txt specifies:
┌─────────────────────────────────┐
│ numpy                           │───> Array operations, mathematics
│ pandas                          │───> Data frames, CSV handling
│ geopandas                       │───> Shapefile I/O, spatial operations
│ rasterio                        │───> GeoTIFF reading, raster operations
│ matplotlib                      │───> Plotting, PDF generation
│ scipy                           │───> Gaussian fitting (curve_fit), KDE
│ opencv-python (cv2)             │───> Edge detection (Canny)
│ uncertainties                   │───> Error propagation
│ shapely                         │───> Geometric operations
│ craterstats (optional)          │───> CSFD plotting
└─────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                            TESTING FRAMEWORK                                 │
└─────────────────────────────────────────────────────────────────────────────┘

tests/
  │
  ├─ test_input_module.py
  │    │
  │    ├─> test_coordinate_conversion()
  │    ├─> test_read_crater_file()
  │    ├─> test_latlon_xy_roundtrip()
  │    └─> test_projection_accuracy()
  │
  ├─ test_refine_rim.py
  │    │
  │    ├─> test_compute_edge_strength()
  │    ├─> test_compute_rim_probability()
  │    ├─> test_topographic_quality()
  │    └─> test_probability_scoring()
  │
  ├─ test_morphometry.py
  │    │
  │    ├─> test_gaussian_2d()
  │    ├─> test_fit_gaussian_floor()
  │    ├─> test_error_propagation()
  │    └─> test_morphometry_field_extraction()
  │
  └─ test_syntax.py
       │
       └─> Validates all .py files compile


┌─────────────────────────────────────────────────────────────────────────────┐
│                        CONFIGURATION FILES                                   │
└─────────────────────────────────────────────────────────────────────────────┘

config/regions.json
  │
  │ defines:
  ├─> Region names
  ├─> DEM file paths
  ├─> Orthophoto paths
  ├─> Shapefile paths
  ├─> Output directories
  └─> Processing parameters (min_diameter, etc.)
       │
       │ read by
       ▼
  config.py (Config class)
       │
       │ provides paths to
       ▼
  All processing modules


┌─────────────────────────────────────────────────────────────────────────────┐
│                    EXECUTION MODES SUMMARY                                   │
└─────────────────────────────────────────────────────────────────────────────┘

MODE 1: Complete Pipeline (Recommended)
  python analyze_small_lunar_craters.py \
      --image image.tif --dtm dtm.tif --craters craters.csv

  Executes: Step 0 → Step 2 → Step 3 → PDF Report
  Output: SLC_morphometry_results/ with complete analysis

MODE 2: Individual Steps (Advanced)
  # Step 0 only
  python process_crater_inputs.py --image ... --dtm ... --craters ...

  # Step 2 only
  python refine_crater_rims.py --shapefile initial.shp --image ... --dtm ...

  # Step 3 only
  python analyze_crater_morphometry.py --shapefile refined.shp --dtm ... --image ...

MODE 3: Python API (Programmatic)
  from crater_analysis.input_module import process_crater_inputs
  from crater_analysis.refine_crater_rim import refine_crater_rims
  from crater_analysis.analyze_morphometry import analyze_crater_morphometry

  # Call functions directly with parameters


═══════════════════════════════════════════════════════════════════════════════
                              END OF BLOCK DIAGRAM
═══════════════════════════════════════════════════════════════════════════════
