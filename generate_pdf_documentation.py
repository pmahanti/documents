#!/usr/bin/env python3
"""
Generate PDF documentation for Crater Detection Training Data Generator.
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, Image, ListFlowable, ListItem
)
from reportlab.lib import colors
from datetime import datetime


def generate_pdf(output_file='Crater_Detection_Documentation.pdf'):
    """Generate comprehensive PDF documentation."""

    # Create the PDF document
    doc = SimpleDocTemplate(
        output_file,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18,
    )

    # Container for the 'Flowable' objects
    elements = []

    # Define styles
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )

    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )

    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=10,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    )

    code_style = ParagraphStyle(
        'Code',
        parent=styles['Code'],
        fontSize=9,
        leftIndent=20,
        rightIndent=20,
        textColor=colors.HexColor('#2c3e50'),
        backColor=colors.HexColor('#f4f4f4'),
        borderWidth=1,
        borderColor=colors.HexColor('#cccccc'),
        borderPadding=10,
        fontName='Courier'
    )

    # Title Page
    elements.append(Spacer(1, 2*inch))
    elements.append(Paragraph("Crater Detection<br/>Training Data Generator", title_style))
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph(
        "A Comprehensive Python Toolkit for Lunar Crater Detection",
        ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=14, alignment=TA_CENTER)
    ))
    elements.append(Spacer(1, 0.5*inch))
    elements.append(Paragraph(
        f"Documentation Generated: {datetime.now().strftime('%B %d, %Y')}",
        ParagraphStyle('Date', parent=styles['Normal'], fontSize=10, alignment=TA_CENTER, textColor=colors.grey)
    ))
    elements.append(PageBreak())

    # Table of Contents
    elements.append(Paragraph("Table of Contents", heading1_style))
    elements.append(Spacer(1, 0.2*inch))

    toc_items = [
        "1. Overview",
        "2. Features",
        "3. Installation",
        "4. Quick Start",
        "5. Command Line Usage",
        "6. Input Requirements",
        "7. Output Structure",
        "8. Label Formats",
        "9. Dataset Utilities",
        "10. Advanced Examples",
        "11. Troubleshooting",
        "12. Technical Details"
    ]

    for item in toc_items:
        elements.append(Paragraph(item, styles['Normal']))
        elements.append(Spacer(1, 0.1*inch))

    elements.append(PageBreak())

    # 1. Overview
    elements.append(Paragraph("1. Overview", heading1_style))
    elements.append(Paragraph(
        """The Crater Detection Training Data Generator is a professional-grade Python toolkit designed
        to facilitate the creation of labeled training datasets for crater detection in lunar images.
        This tool bridges the gap between planetary science data and modern deep learning frameworks,
        enabling researchers to quickly prepare datasets for training YOLO, CNN, and DETR models.""",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph(
        """The toolkit handles the complexities of geospatial data processing, including coordinate
        reference system transformations, multiple map projections, and format conversions, allowing
        researchers to focus on model development rather than data preprocessing.""",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 0.3*inch))

    # 2. Features
    elements.append(Paragraph("2. Features", heading1_style))

    features = [
        "<b>Multi-Format Input Support:</b> Handles GeoTiff and ISIS .cub files seamlessly",
        "<b>Projection Flexibility:</b> Supports equidistant cylindrical, stereographic, and orthographic projections",
        "<b>Shapefile Integration:</b> Reads crater annotations from ESRI shapefiles with automatic CRS reprojection",
        "<b>Dual Output Formats:</b> Generates both YOLO and COCO format labels for different model architectures",
        "<b>Topography Processing:</b> Optional support for topography/DEM rasters for multi-modal learning",
        "<b>Intelligent Tiling:</b> Automatically splits large images into manageable tiles with configurable overlap",
        "<b>Visualization:</b> Creates annotated PNG images showing all labeled craters for quality verification",
        "<b>Dataset Management:</b> Built-in utilities for splitting, validation, statistics, and format conversion",
        "<b>Production Ready:</b> Robust error handling, progress reporting, and comprehensive logging"
    ]

    for feature in features:
        elements.append(Paragraph(f"• {feature}", styles['BodyText']))
        elements.append(Spacer(1, 0.1*inch))

    elements.append(PageBreak())

    # 3. Installation
    elements.append(Paragraph("3. Installation", heading1_style))

    elements.append(Paragraph("3.1 System Dependencies", heading2_style))
    elements.append(Paragraph(
        "For ISIS .cub file support, install GDAL with ISIS drivers:",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 0.1*inch))

    elements.append(Paragraph(
        "<font name='Courier' size='9'>Ubuntu/Debian:<br/>"
        "sudo apt-get install gdal-bin libgdal-dev python3-gdal<br/><br/>"
        "macOS (using Homebrew):<br/>"
        "brew install gdal</font>",
        code_style
    ))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("3.2 Python Dependencies", heading2_style))
    elements.append(Paragraph(
        "<font name='Courier' size='9'>pip install -r requirements.txt</font>",
        code_style
    ))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("3.3 Required Python Packages", heading2_style))

    packages_data = [
        ['Package', 'Version', 'Purpose'],
        ['rasterio', '≥1.3.0', 'Geospatial raster I/O'],
        ['gdal', '≥3.0.0', 'ISIS .cub file support'],
        ['geopandas', '≥0.12.0', 'Shapefile processing'],
        ['shapely', '≥2.0.0', 'Geometric operations'],
        ['numpy', '≥1.23.0', 'Array processing'],
        ['pillow', '≥9.0.0', 'Image handling'],
        ['matplotlib', '≥3.5.0', 'Visualization'],
        ['scikit-learn', '≥1.0.0', 'Dataset splitting']
    ]

    package_table = Table(packages_data, colWidths=[1.5*inch, 1.2*inch, 2.5*inch])
    package_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
    ]))
    elements.append(package_table)

    elements.append(PageBreak())

    # 4. Quick Start
    elements.append(Paragraph("4. Quick Start", heading1_style))

    elements.append(Paragraph("4.1 Basic Usage", heading2_style))
    elements.append(Paragraph(
        "Generate a YOLO format dataset from a single image:",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 0.1*inch))

    elements.append(Paragraph(
        "<font name='Courier' size='9'>python crater_training_data_generator.py \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--image lunar_image.tif \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--craters craters.shp \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--output ./crater_dataset \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--format yolo</font>",
        code_style
    ))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("4.2 Advanced Usage with Tiling", heading2_style))
    elements.append(Paragraph(
        "Generate tiled dataset with topography in both formats:",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 0.1*inch))

    elements.append(Paragraph(
        "<font name='Courier' size='9'>python crater_training_data_generator.py \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--image lunar_mosaic.cub \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--craters crater_database.shp \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--output ./crater_dataset_tiled \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--format both \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--tile-size 512 \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--overlap 64 \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--topography lunar_dem.tif</font>",
        code_style
    ))

    elements.append(PageBreak())

    # 5. Command Line Arguments
    elements.append(Paragraph("5. Command Line Usage", heading1_style))

    args_data = [
        ['Argument', 'Required', 'Description'],
        ['--image', 'Yes', 'Path to input image (GeoTiff or .cub)'],
        ['--craters', 'Yes', 'Path to crater shapefile'],
        ['--output', 'Yes', 'Output directory for dataset'],
        ['--format', 'No', 'Output format: yolo, coco, or both (default: yolo)'],
        ['--topography', 'No', 'Optional topography raster file path'],
        ['--tile-size', 'No', 'Tile size for splitting large images (e.g., 512)'],
        ['--overlap', 'No', 'Overlap between tiles in pixels (default: 0)'],
    ]

    args_table = Table(args_data, colWidths=[1.3*inch, 0.9*inch, 3.0*inch])
    args_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    elements.append(args_table)

    elements.append(PageBreak())

    # 6. Input Requirements
    elements.append(Paragraph("6. Input Requirements", heading1_style))

    elements.append(Paragraph("6.1 Image Files", heading2_style))
    elements.append(Paragraph("<b>Supported Formats:</b>", styles['BodyText']))
    elements.append(Paragraph("• GeoTiff (.tif, .tiff) with embedded georeferencing", styles['BodyText']))
    elements.append(Paragraph("• ISIS cube files (.cub) with label information", styles['BodyText']))
    elements.append(Spacer(1, 0.1*inch))

    elements.append(Paragraph("<b>Requirements:</b>", styles['BodyText']))
    elements.append(Paragraph("• Valid coordinate reference system (CRS) information", styles['BodyText']))
    elements.append(Paragraph("• Single-band (grayscale) or multi-band (RGB) supported", styles['BodyText']))
    elements.append(Paragraph("• Any bit depth supported by GDAL/rasterio", styles['BodyText']))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("6.2 Crater Shapefile", heading2_style))
    elements.append(Paragraph("<b>Format:</b> ESRI Shapefile (.shp)", styles['BodyText']))
    elements.append(Spacer(1, 0.1*inch))

    elements.append(Paragraph("<b>Supported Geometry Types:</b>", styles['BodyText']))
    elements.append(Paragraph("• <b>Point:</b> Center point of crater (generates circular bounding box)", styles['BodyText']))
    elements.append(Paragraph("• <b>Polygon:</b> Crater outline (generates bounding box from bounds)", styles['BodyText']))
    elements.append(Paragraph("• <b>Circle/Ellipse:</b> Crater perimeter", styles['BodyText']))
    elements.append(Spacer(1, 0.1*inch))

    elements.append(Paragraph("<b>Common Attributes (optional but recommended):</b>", styles['BodyText']))
    elements.append(Paragraph("• diameter or radius: Crater size in map units", styles['BodyText']))
    elements.append(Paragraph("• name or id: Crater identifier", styles['BodyText']))
    elements.append(Paragraph("• confidence: Annotation confidence/quality", styles['BodyText']))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("6.3 Projection Support", heading2_style))
    elements.append(Paragraph(
        """The tool automatically handles different map projections commonly used in planetary science.
        The projection information should be embedded in the GeoTiff metadata or ISIS label.
        Supported projections include:""",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph("• <b>Equidistant Cylindrical (Equirectangular):</b> Most common for global mosaics", styles['BodyText']))
    elements.append(Paragraph("• <b>Stereographic:</b> Often used for polar regions", styles['BodyText']))
    elements.append(Paragraph("• <b>Orthographic:</b> Used for hemisphere views", styles['BodyText']))

    elements.append(PageBreak())

    # 7. Output Structure
    elements.append(Paragraph("7. Output Structure", heading1_style))

    elements.append(Paragraph(
        "<font name='Courier' size='8'>"
        "output_directory/<br/>"
        "├── images/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Training images<br/>"
        "│&nbsp;&nbsp;&nbsp;├── image_000.png<br/>"
        "│&nbsp;&nbsp;&nbsp;├── image_001.png<br/>"
        "│&nbsp;&nbsp;&nbsp;└── ...<br/>"
        "├── labels/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# YOLO format labels<br/>"
        "│&nbsp;&nbsp;&nbsp;├── image_000.txt<br/>"
        "│&nbsp;&nbsp;&nbsp;├── image_001.txt<br/>"
        "│&nbsp;&nbsp;&nbsp;└── ...<br/>"
        "├── visualizations/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Annotated images<br/>"
        "│&nbsp;&nbsp;&nbsp;├── image_000_labeled.png<br/>"
        "│&nbsp;&nbsp;&nbsp;└── ...<br/>"
        "├── topography_images/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Topography images (optional)<br/>"
        "│&nbsp;&nbsp;&nbsp;└── ...<br/>"
        "├── topography_labels/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Topography labels (optional)<br/>"
        "│&nbsp;&nbsp;&nbsp;└── ...<br/>"
        "├── annotations_coco.json&nbsp;&nbsp;&nbsp;&nbsp;# COCO format labels<br/>"
        "└── dataset.yaml&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# YOLO dataset config"
        "</font>",
        code_style
    ))

    elements.append(PageBreak())

    # 8. Label Formats
    elements.append(Paragraph("8. Label Formats", heading1_style))

    elements.append(Paragraph("8.1 YOLO Format", heading2_style))
    elements.append(Paragraph(
        "Each .txt file contains one line per crater with normalized coordinates:",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph(
        "<font name='Courier' size='9'>class_id x_center y_center width height</font>",
        code_style
    ))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph("Example:", styles['BodyText']))
    elements.append(Paragraph(
        "<font name='Courier' size='9'>0 0.5234 0.7123 0.0456 0.0389<br/>"
        "0 0.3421 0.2156 0.0892 0.0734</font>",
        code_style
    ))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph(
        "All values except class_id are normalized to [0, 1] range.",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("8.2 COCO Format", heading2_style))
    elements.append(Paragraph(
        "JSON file with three main sections:",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph(
        '<font name="Courier" size="8">{<br/>'
        '&nbsp;&nbsp;"images": [<br/>'
        '&nbsp;&nbsp;&nbsp;&nbsp;{"id": 0, "file_name": "image_000.png", "width": 512, "height": 512}<br/>'
        '&nbsp;&nbsp;],<br/>'
        '&nbsp;&nbsp;"annotations": [<br/>'
        '&nbsp;&nbsp;&nbsp;&nbsp;{"id": 0, "image_id": 0, "category_id": 0, "bbox": [x, y, w, h]}<br/>'
        '&nbsp;&nbsp;],<br/>'
        '&nbsp;&nbsp;"categories": [<br/>'
        '&nbsp;&nbsp;&nbsp;&nbsp;{"id": 0, "name": "crater"}<br/>'
        '&nbsp;&nbsp;]<br/>'
        '}</font>',
        code_style
    ))

    elements.append(PageBreak())

    # 9. Dataset Utilities
    elements.append(Paragraph("9. Dataset Utilities", heading1_style))

    elements.append(Paragraph(
        "The crater_utils.py module provides essential dataset management functions:",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("9.1 Split Dataset", heading2_style))
    elements.append(Paragraph(
        "<font name='Courier' size='9'>python crater_utils.py split --dataset ./crater_dataset \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--train-ratio 0.8 \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--val-ratio 0.1 \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--test-ratio 0.1</font>",
        code_style
    ))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("9.2 View Statistics", heading2_style))
    elements.append(Paragraph(
        "<font name='Courier' size='9'>python crater_utils.py stats --dataset ./crater_dataset</font>",
        code_style
    ))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph("Output includes:", styles['BodyText']))
    elements.append(Paragraph("• Total images and crater counts", styles['BodyText']))
    elements.append(Paragraph("• Average craters per image", styles['BodyText']))
    elements.append(Paragraph("• Crater size statistics (min, max, mean, median)", styles['BodyText']))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("9.3 Verify Dataset", heading2_style))
    elements.append(Paragraph(
        "<font name='Courier' size='9'>python crater_utils.py verify --dataset ./crater_dataset</font>",
        code_style
    ))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph("Checks for:", styles['BodyText']))
    elements.append(Paragraph("• Missing label files", styles['BodyText']))
    elements.append(Paragraph("• Missing image files", styles['BodyText']))
    elements.append(Paragraph("• Dataset integrity", styles['BodyText']))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("9.4 Convert Formats", heading2_style))
    elements.append(Paragraph(
        "<font name='Courier' size='9'>python crater_utils.py convert --dataset ./crater_dataset \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--output ./annotations_coco.json</font>",
        code_style
    ))

    elements.append(PageBreak())

    # 10. Advanced Examples
    elements.append(Paragraph("10. Advanced Examples", heading1_style))

    elements.append(Paragraph("10.1 Large Mosaic Processing", heading2_style))
    elements.append(Paragraph(
        "For large images (>10,000 pixels), use tiling to manage memory:",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph(
        "<font name='Courier' size='8'>python crater_training_data_generator.py \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--image large_mosaic_20000x20000.tif \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--craters crater_catalog.shp \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--output ./large_dataset \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--tile-size 640 \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--overlap 128 \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--format both</font>",
        code_style
    ))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("10.2 Multi-Modal Learning", heading2_style))
    elements.append(Paragraph(
        "Create datasets with both optical and topographic data:",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph(
        "<font name='Courier' size='8'>python crater_training_data_generator.py \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--image lunar_optical.tif \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--craters craters.shp \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--topography lunar_slope_map.tif \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--output ./multimodal_dataset \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--tile-size 512</font>",
        code_style
    ))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph(
        "This creates separate image and topography datasets with matching labels.",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("10.3 Complete Workflow", heading2_style))
    elements.append(Paragraph(
        "End-to-end workflow from generation to training:",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph(
        "<font name='Courier' size='8'># Step 1: Generate dataset<br/>"
        "python crater_training_data_generator.py \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--image data.tif --craters craters.shp \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--output ./dataset --tile-size 512<br/><br/>"
        "# Step 2: Verify<br/>"
        "python crater_utils.py verify --dataset ./dataset<br/><br/>"
        "# Step 3: View statistics<br/>"
        "python crater_utils.py stats --dataset ./dataset<br/><br/>"
        "# Step 4: Split into train/val/test<br/>"
        "python crater_utils.py split --dataset ./dataset</font>",
        code_style
    ))

    elements.append(PageBreak())

    # 11. Troubleshooting
    elements.append(Paragraph("11. Troubleshooting", heading1_style))

    issues = [
        {
            'issue': 'CRS mismatch warning',
            'solution': 'The tool automatically reprojects shapefiles. Verify both files have valid CRS: gdalinfo image.tif | grep "Coordinate System" and ogrinfo -al -so craters.shp'
        },
        {
            'issue': 'Craters not in correct locations',
            'solution': 'Check: (1) Projection mismatch, (2) Different coordinate systems, (3) Missing georeferencing. Ensure both image and shapefile have proper CRS defined.'
        },
        {
            'issue': 'ISIS .cub files not opening',
            'solution': 'Install GDAL with ISIS drivers or convert to GeoTiff: gdal_translate input.cub output.tif'
        },
        {
            'issue': 'Out of memory errors',
            'solution': 'Use tiling: --tile-size 512 --overlap 64'
        },
        {
            'issue': 'Empty visualizations or missing craters',
            'solution': 'Check that crater shapefile overlaps with image extent. Use ogrinfo and gdalinfo to verify spatial bounds match.'
        }
    ]

    for i, item in enumerate(issues, 1):
        elements.append(Paragraph(f"<b>Issue {i}: {item['issue']}</b>", styles['BodyText']))
        elements.append(Paragraph(f"<i>Solution:</i> {item['solution']}", styles['BodyText']))
        elements.append(Spacer(1, 0.15*inch))

    elements.append(PageBreak())

    # 12. Technical Details
    elements.append(Paragraph("12. Technical Details", heading1_style))

    elements.append(Paragraph("12.1 Coordinate Transformation", heading2_style))
    elements.append(Paragraph(
        """The tool uses rasterio's affine transformation to convert between geographic coordinates
        (from shapefiles) and pixel coordinates (for bounding boxes). The transformation automatically
        accounts for the map projection, resolution, and extent of the input image.""",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("12.2 Tiling Strategy", heading2_style))
    elements.append(Paragraph(
        """When tiling is enabled, the tool uses a sliding window approach with configurable overlap.
        This ensures craters near tile boundaries are captured completely. The overlap should be set
        to at least twice the maximum expected crater diameter to avoid splitting craters across tiles.""",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("12.3 Image Normalization", heading2_style))
    elements.append(Paragraph(
        """Images are normalized to 0-255 range using percentile-based scaling (2nd and 98th percentiles)
        to handle outliers and varying dynamic ranges. This approach is more robust than min-max scaling
        for planetary science data which often contains extreme values.""",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("12.4 Performance Considerations", heading2_style))

    performance_data = [
        ['Image Size', 'Recommended Approach', 'Memory Usage'],
        ['< 5000x5000', 'Full image processing', 'Low (< 2GB)'],
        ['5000-10000', 'Full or tile (512-1024)', 'Medium (2-4GB)'],
        ['> 10000x10000', 'Tiling required (512-640)', 'High (4-8GB)'],
        ['Very large mosaics', 'Tiling with overlap 64-128', 'Moderate per tile'],
    ]

    perf_table = Table(performance_data, colWidths=[1.5*inch, 2.0*inch, 1.7*inch])
    perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    elements.append(perf_table)

    elements.append(PageBreak())

    # File Information Section
    elements.append(Paragraph("13. Code Structure", heading1_style))

    elements.append(Paragraph("13.1 Main Components", heading2_style))

    components = [
        ('<b>crater_training_data_generator.py</b>',
         'Main script containing the CraterDatasetGenerator class. Handles image loading, coordinate transformation, tiling, and label generation.'),
        ('<b>crater_utils.py</b>',
         'Utility functions for dataset management including splitting, statistics, verification, and format conversion.'),
        ('<b>example_usage.py</b>',
         'Example scripts demonstrating various use cases from simple generation to complete training workflows.'),
        ('<b>requirements.txt</b>',
         'Python package dependencies with version specifications.')
    ]

    for name, desc in components:
        elements.append(Paragraph(name, styles['BodyText']))
        elements.append(Paragraph(desc, styles['BodyText']))
        elements.append(Spacer(1, 0.15*inch))

    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("13.2 Key Classes and Methods", heading2_style))

    elements.append(Paragraph("<b>CraterDatasetGenerator</b>", styles['BodyText']))
    elements.append(Paragraph("• load_data(): Loads image and shapefile with CRS handling", styles['BodyText']))
    elements.append(Paragraph("• geometry_to_bbox(): Converts shapefile geometry to pixel bounding boxes", styles['BodyText']))
    elements.append(Paragraph("• bbox_to_yolo(): Converts pixel bbox to YOLO normalized format", styles['BodyText']))
    elements.append(Paragraph("• generate_full_image_dataset(): Processes entire image", styles['BodyText']))
    elements.append(Paragraph("• generate_tiled_dataset(): Splits image into tiles with overlap", styles['BodyText']))
    elements.append(Spacer(1, 0.2*inch))

    # Best Practices
    elements.append(Paragraph("14. Best Practices", heading1_style))

    practices = [
        ('<b>Data Preparation:</b>', 'Always verify CRS information in both image and shapefile before processing. Use gdalinfo and ogrinfo to check.'),
        ('<b>Tile Size Selection:</b>', 'Choose tile sizes that match your target model input size (512, 640, or 1024 are common for YOLO).'),
        ('<b>Overlap Strategy:</b>', 'Use 10-20% overlap (64-128 pixels for 512-640 px tiles) to avoid missing craters on boundaries.'),
        ('<b>Quality Control:</b>', 'Always review visualizations in the visualizations/ folder to verify correct labeling.'),
        ('<b>Dataset Balance:</b>', 'Check statistics to ensure good distribution of crater sizes and reasonable craters per image.'),
        ('<b>Training Split:</b>', 'Use 70-80% training, 10-20% validation, and 10% test for typical datasets. Adjust based on dataset size.'),
        ('<b>Multi-scale Training:</b>', 'For datasets with large size variation, consider creating separate datasets for different scales.'),
        ('<b>Memory Management:</b>', 'For large datasets, process in batches or use tiling to avoid memory issues.')
    ]

    for title, desc in practices:
        elements.append(Paragraph(f"{title} {desc}", styles['BodyText']))
        elements.append(Spacer(1, 0.1*inch))

    elements.append(PageBreak())

    # Training Section
    elements.append(Paragraph("15. Training Models", heading1_style))

    elements.append(Paragraph("15.1 YOLOv8 Training", heading2_style))
    elements.append(Paragraph(
        "<font name='Courier' size='8'>from ultralytics import YOLO<br/><br/>"
        "# Load pretrained model<br/>"
        "model = YOLO('yolov8n.pt')<br/><br/>"
        "# Train on crater dataset<br/>"
        "results = model.train(<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;data='./crater_dataset/dataset.yaml',<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;epochs=100,<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;imgsz=512,<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;batch=16,<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;patience=20,<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;name='crater_detector'<br/>"
        ")<br/><br/>"
        "# Validate<br/>"
        "metrics = model.val()<br/>"
        "print(f'mAP50: {metrics.box.map50:.3f}')</font>",
        code_style
    ))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("15.2 Model Selection", heading2_style))

    model_data = [
        ['Model', 'Speed', 'Accuracy', 'Use Case'],
        ['YOLOv8n', 'Fastest', 'Good', 'Real-time detection, embedded systems'],
        ['YOLOv8s', 'Fast', 'Better', 'Balanced speed/accuracy'],
        ['YOLOv8m', 'Medium', 'High', 'Research applications'],
        ['YOLOv8l/x', 'Slow', 'Highest', 'Maximum accuracy required'],
        ['DETR', 'Slow', 'High', 'Transformer-based, research'],
    ]

    model_table = Table(model_data, colWidths=[1.2*inch, 1.0*inch, 1.0*inch, 2.0*inch])
    model_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
    ]))
    elements.append(model_table)

    elements.append(PageBreak())

    # Appendix
    elements.append(Paragraph("Appendix A: Example Workflows", heading1_style))

    elements.append(Paragraph("A.1 Workflow for Small Dataset (< 100 images)", heading2_style))
    elements.append(Paragraph(
        "<font name='Courier' size='8'># Generate without tiling<br/>"
        "python crater_training_data_generator.py \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--image small_region.tif \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--craters craters.shp \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--output ./small_dataset \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--format yolo<br/><br/>"
        "# Split 70/20/10<br/>"
        "python crater_utils.py split --dataset ./small_dataset \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1<br/><br/>"
        "# Train lightweight model<br/>"
        "yolo train data=./small_dataset/dataset.yaml \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;model=yolov8n.pt epochs=50</font>",
        code_style
    ))
    elements.append(Spacer(1, 0.3*inch))

    elements.append(Paragraph("A.2 Workflow for Large Dataset (> 1000 images)", heading2_style))
    elements.append(Paragraph(
        "<font name='Courier' size='8'># Generate with tiling<br/>"
        "python crater_training_data_generator.py \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--image large_mosaic.cub \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--craters comprehensive_catalog.shp \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--output ./large_dataset \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--format both \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--tile-size 640 --overlap 96<br/><br/>"
        "# Verify and get stats<br/>"
        "python crater_utils.py verify --dataset ./large_dataset<br/>"
        "python crater_utils.py stats --dataset ./large_dataset<br/><br/>"
        "# Split 80/15/5<br/>"
        "python crater_utils.py split --dataset ./large_dataset \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;--train-ratio 0.8 --val-ratio 0.15 --test-ratio 0.05<br/><br/>"
        "# Train robust model<br/>"
        "yolo train data=./large_dataset/dataset.yaml \\<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;model=yolov8m.pt epochs=150 imgsz=640</font>",
        code_style
    ))

    elements.append(PageBreak())

    # Summary and Conclusion
    elements.append(Paragraph("Summary", heading1_style))
    elements.append(Paragraph(
        """The Crater Detection Training Data Generator provides a complete solution for preparing
        lunar crater detection datasets. By automating the complex processes of geospatial data handling,
        coordinate transformations, and format conversions, it enables researchers to focus on model
        development and analysis rather than data preprocessing.""",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("<b>Key Takeaways:</b>", styles['BodyText']))
    elements.append(Spacer(1, 0.1*inch))

    takeaways = [
        "Supports multiple input formats (GeoTiff, ISIS .cub) and projections",
        "Generates both YOLO and COCO format labels for flexibility",
        "Handles large images efficiently through intelligent tiling",
        "Includes comprehensive utilities for dataset management",
        "Provides visualization for quality assurance",
        "Production-ready with robust error handling",
        "Well-documented with extensive examples"
    ]

    for takeaway in takeaways:
        elements.append(Paragraph(f"• {takeaway}", styles['BodyText']))
        elements.append(Spacer(1, 0.08*inch))

    elements.append(Spacer(1, 0.3*inch))

    elements.append(Paragraph(
        """Whether you're working with small annotated regions or large planetary mosaics, this toolkit
        provides the flexibility and robustness needed for professional crater detection research.
        The combination of automation, quality control features, and comprehensive documentation makes
        it an ideal choice for planetary scientists and machine learning researchers alike.""",
        styles['BodyText']
    ))

    # Build PDF
    doc.build(elements)
    print(f"PDF documentation generated: {output_file}")


if __name__ == '__main__':
    generate_pdf()
