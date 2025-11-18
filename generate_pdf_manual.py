#!/usr/bin/env python3
"""
Generate PDF manual for Crater Marker Tool
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, ListFlowable, ListItem
)
from reportlab.lib.colors import HexColor
from datetime import datetime


def create_pdf_manual(filename="Crater_Marker_Tool_Manual.pdf"):
    """Generate a comprehensive PDF manual for the Crater Marker Tool."""

    # Create the PDF document
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=1*inch,
        bottomMargin=0.75*inch,
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
        textColor=HexColor('#1a237e'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )

    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=14,
        textColor=HexColor('#424242'),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica'
    )

    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=HexColor('#1976d2'),
        spaceAfter=12,
        spaceBefore=20,
        fontName='Helvetica-Bold'
    )

    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=HexColor('#1976d2'),
        spaceAfter=10,
        spaceBefore=15,
        fontName='Helvetica-Bold'
    )

    heading3_style = ParagraphStyle(
        'CustomHeading3',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=HexColor('#424242'),
        spaceAfter=8,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )

    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceAfter=10,
        leading=14
    )

    code_style = ParagraphStyle(
        'Code',
        parent=styles['Code'],
        fontSize=9,
        fontName='Courier',
        textColor=HexColor('#c7254e'),
        backColor=HexColor('#f9f2f4'),
        leftIndent=20,
        rightIndent=20,
        spaceAfter=10,
        spaceBefore=5
    )

    # Title Page
    elements.append(Spacer(1, 1.5*inch))
    elements.append(Paragraph("Crater Marker Tool", title_style))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph("Installation and User Guide", subtitle_style))
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph(
        "A Python Application for Marking and Measuring Impact Craters<br/>on GeoTIFF and ISIS Cube Files",
        subtitle_style
    ))
    elements.append(Spacer(1, 1*inch))

    # Version info
    info_data = [
        ['Version:', '1.0'],
        ['Date:', datetime.now().strftime('%B %Y')],
        ['License:', 'MIT License'],
    ]
    info_table = Table(info_data, colWidths=[1.5*inch, 3*inch])
    info_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(info_table)

    elements.append(PageBreak())

    # Table of Contents
    elements.append(Paragraph("Table of Contents", heading1_style))
    toc_items = [
        "1. Overview",
        "2. System Requirements",
        "3. Installation Instructions",
        "   3.1 Windows Installation",
        "   3.2 macOS Installation",
        "   3.3 Linux Installation",
        "4. Getting Started",
        "5. Working Instructions",
        "   5.1 Opening an Image",
        "   5.2 Marking Craters",
        "   5.3 Deleting Craters",
        "   5.4 Exporting Data",
        "6. Output Formats",
        "7. Tips and Best Practices",
        "8. Troubleshooting",
        "9. Technical Details",
        "10. References",
    ]
    for item in toc_items:
        elements.append(Paragraph(item, body_style))

    elements.append(PageBreak())

    # 1. Overview
    elements.append(Paragraph("1. Overview", heading1_style))
    elements.append(Paragraph(
        "The Crater Marker Tool is a specialized Python application designed for planetary scientists "
        "and researchers to identify, mark, and measure impact craters on planetary surface images. "
        "The tool provides an intuitive graphical interface for crater identification using a "
        "three-point selection method on the crater rim.",
        body_style
    ))

    elements.append(Paragraph("Key Features:", heading3_style))
    features = [
        "Load GeoTIFF and ISIS cube (.cub) files",
        "Mark craters by selecting 3 points on the rim",
        "Automatic circle fitting algorithm",
        "Visual overlay of identified craters",
        "Delete most recent crater marking",
        "Automatic saving after each crater identification",
        "Export data in .diam and shapefile formats",
        "Projection-independent coordinate handling",
        "Session recovery from auto-save"
    ]
    feature_list = ListFlowable(
        [ListItem(Paragraph(f, body_style), leftIndent=20) for f in features],
        bulletType='bullet',
        start='circle',
    )
    elements.append(feature_list)
    elements.append(Spacer(1, 0.2*inch))

    # 2. System Requirements
    elements.append(Paragraph("2. System Requirements", heading1_style))

    req_data = [
        ['Component', 'Requirement'],
        ['Operating System', 'Windows 10+, macOS 10.14+, or Linux'],
        ['Python', 'Version 3.7 or higher'],
        ['RAM', 'Minimum 4 GB (8 GB recommended)'],
        ['Disk Space', '500 MB for software + space for data'],
        ['Display', '1024x768 or higher resolution'],
    ]
    req_table = Table(req_data, colWidths=[2*inch, 4*inch])
    req_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1976d2')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f5f5f5')),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ]))
    elements.append(req_table)
    elements.append(Spacer(1, 0.2*inch))

    elements.append(PageBreak())

    # 3. Installation Instructions
    elements.append(Paragraph("3. Installation Instructions", heading1_style))

    # 3.1 Windows
    elements.append(Paragraph("3.1 Windows Installation", heading2_style))
    elements.append(Paragraph("Step 1: Install Python", heading3_style))
    elements.append(Paragraph(
        "Download and install Python from <a href='https://www.python.org/downloads/'>python.org</a>. "
        "During installation, make sure to check 'Add Python to PATH'.",
        body_style
    ))

    elements.append(Paragraph("Step 2: Install GDAL", heading3_style))
    elements.append(Paragraph(
        "GDAL is required for reading GeoTIFF and ISIS cube files. Download the appropriate GDAL wheel "
        "for your Python version from:",
        body_style
    ))
    elements.append(Paragraph(
        "https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal",
        code_style
    ))
    elements.append(Paragraph("Then install it using:", body_style))
    elements.append(Paragraph("pip install path/to/GDAL-xxx.whl", code_style))

    elements.append(Paragraph("Step 3: Download Crater Marker Tool", heading3_style))
    elements.append(Paragraph(
        "Download the crater marker tool files to a directory of your choice.",
        body_style
    ))

    elements.append(Paragraph("Step 4: Install Dependencies", heading3_style))
    elements.append(Paragraph("Open Command Prompt in the tool directory and run:", body_style))
    elements.append(Paragraph("pip install -r requirements.txt", code_style))

    elements.append(Paragraph("Step 5: Run the Application", heading3_style))
    elements.append(Paragraph("python crater_marker.py", code_style))

    elements.append(Spacer(1, 0.2*inch))

    # 3.2 macOS
    elements.append(Paragraph("3.2 macOS Installation", heading2_style))
    elements.append(Paragraph("Step 1: Install Homebrew (if not installed)", heading3_style))
    elements.append(Paragraph("/bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"", code_style))

    elements.append(Paragraph("Step 2: Install Python and GDAL", heading3_style))
    elements.append(Paragraph("brew install python gdal", code_style))

    elements.append(Paragraph("Step 3: Download and Install Tool", heading3_style))
    elements.append(Paragraph("cd /path/to/crater-marker-tool<br/>pip install -r requirements.txt", code_style))

    elements.append(Paragraph("Step 4: Run the Application", heading3_style))
    elements.append(Paragraph("python crater_marker.py", code_style))

    elements.append(Spacer(1, 0.2*inch))

    # 3.3 Linux
    elements.append(Paragraph("3.3 Linux Installation (Ubuntu/Debian)", heading2_style))
    elements.append(Paragraph("Step 1: Install System Dependencies", heading3_style))
    elements.append(Paragraph("sudo apt-get update<br/>sudo apt-get install python3 python3-pip gdal-bin libgdal-dev", code_style))

    elements.append(Paragraph("Step 2: Install Python GDAL", heading3_style))
    elements.append(Paragraph("pip3 install GDAL==$(gdal-config --version)", code_style))

    elements.append(Paragraph("Step 3: Install Crater Marker Tool", heading3_style))
    elements.append(Paragraph("cd /path/to/crater-marker-tool<br/>pip3 install -r requirements.txt", code_style))

    elements.append(Paragraph("Step 4: Run the Application", heading3_style))
    elements.append(Paragraph("python3 crater_marker.py", code_style))

    elements.append(PageBreak())

    # 4. Getting Started
    elements.append(Paragraph("4. Getting Started", heading1_style))
    elements.append(Paragraph("Quick Start - Generate Test Data", heading2_style))
    elements.append(Paragraph(
        "To familiarize yourself with the tool, generate synthetic crater data:",
        body_style
    ))
    elements.append(Paragraph("python generate_test_data.py", code_style))
    elements.append(Paragraph(
        "This creates 'test_crater_image.tif' with synthetic craters that you can use to "
        "practice marking craters before working with real data.",
        body_style
    ))
    elements.append(Spacer(1, 0.2*inch))

    # 5. Working Instructions
    elements.append(Paragraph("5. Working Instructions", heading1_style))

    elements.append(Paragraph("5.1 Opening an Image", heading2_style))
    steps_open = [
        "Launch the application by running: <font name='Courier'>python crater_marker.py</font>",
        "Click the 'Open Image' button in the top toolbar",
        "Navigate to your GeoTIFF (.tif, .tiff) or ISIS cube (.cub) file",
        "Select the file and click 'Open'",
        "The image will be displayed in the main window",
        "The status bar will show the loaded filename"
    ]
    open_list = ListFlowable(
        [ListItem(Paragraph(s, body_style), leftIndent=20) for s in steps_open],
        bulletType='1',
    )
    elements.append(open_list)
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("5.2 Marking Craters", heading2_style))
    elements.append(Paragraph(
        "The crater marking process uses a three-point selection method:",
        body_style
    ))
    steps_mark = [
        "Identify a crater you wish to mark in the displayed image",
        "Click on a point on the crater rim - you will see a red dot appear",
        "Click on a second point on the rim, roughly 120° around the crater",
        "Click on a third point on the rim, completing the circle",
        "After the third click, a green circle will automatically appear, fitted to your three points",
        "The crater center and diameter are calculated using geometric algorithms",
        "The crater is automatically saved to 'craters_autosave.json'",
        "The status bar updates showing the crater has been marked",
        "Repeat for additional craters"
    ]
    mark_list = ListFlowable(
        [ListItem(Paragraph(s, body_style), leftIndent=20) for s in steps_mark],
        bulletType='1',
    )
    elements.append(mark_list)

    elements.append(Paragraph("Important Notes:", heading3_style))
    notes_mark = [
        "Try to select points that are evenly spaced around the crater rim for best results",
        "Avoid selecting three points that lie in a straight line (collinear) - the circle fitting will fail",
        "The more circular the crater, the better the fit will be",
        "You can zoom in using your system's zoom features for precise point selection"
    ]
    notes_list = ListFlowable(
        [ListItem(Paragraph(n, body_style), leftIndent=20) for n in notes_mark],
        bulletType='bullet',
        start='circle',
    )
    elements.append(notes_list)
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("5.3 Deleting Craters", heading2_style))
    elements.append(Paragraph(
        "If you make a mistake or want to remove the most recently marked crater:",
        body_style
    ))
    steps_delete = [
        "Click the 'Delete Last Crater' button in the toolbar",
        "The most recently marked crater will be removed from the display",
        "The auto-save file is immediately updated",
        "You can only delete one crater at a time, working backwards from most recent"
    ]
    delete_list = ListFlowable(
        [ListItem(Paragraph(s, body_style), leftIndent=20) for s in steps_delete],
        bulletType='1',
    )
    elements.append(delete_list)
    elements.append(Spacer(1, 0.2*inch))

    elements.append(PageBreak())

    elements.append(Paragraph("5.4 Exporting Data", heading2_style))
    elements.append(Paragraph(
        "When you have finished marking all craters, export your data:",
        body_style
    ))
    steps_export = [
        "Click the 'Export' button in the toolbar",
        "Select a directory where you want to save the output files",
        "Click 'Select Folder' or 'OK'",
        "Two files will be created: a .diam file and a shapefile set",
        "A confirmation dialog will show the export was successful"
    ]
    export_list = ListFlowable(
        [ListItem(Paragraph(s, body_style), leftIndent=20) for s in steps_export],
        bulletType='1',
    )
    elements.append(export_list)
    elements.append(Spacer(1, 0.2*inch))

    # 6. Output Formats
    elements.append(Paragraph("6. Output Formats", heading1_style))

    elements.append(Paragraph("6.1 .diam File Format", heading2_style))
    elements.append(Paragraph(
        "The .diam file is a tab-delimited text file that can be opened in any text editor "
        "or spreadsheet application. It contains:",
        body_style
    ))
    elements.append(Paragraph(
        "# Crater Marker Tool Export<br/>"
        "# Image: /path/to/your/image.tif<br/>"
        "# CRS: +proj=longlat +datum=WGS84 ...<br/>"
        "# Columns: ID    Center_X    Center_Y    Diameter<br/>"
        "1    250.5    750.3    160.2<br/>"
        "2    600.1    400.8    240.5",
        code_style
    ))

    diam_cols = [
        "ID: Unique crater identifier (integer)",
        "Center_X: X coordinate of crater center in map units",
        "Center_Y: Y coordinate of crater center in map units",
        "Diameter: Crater diameter in map units (2 × radius)"
    ]
    diam_list = ListFlowable(
        [ListItem(Paragraph(c, body_style), leftIndent=20) for c in diam_cols],
        bulletType='bullet',
        start='circle',
    )
    elements.append(diam_list)
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("6.2 Shapefile Format", heading2_style))
    elements.append(Paragraph(
        "The shapefile output consists of multiple files (.shp, .shx, .dbf, .prj) that together "
        "form a standard ESRI shapefile. This can be opened in GIS software like QGIS, ArcGIS, or "
        "used in spatial analysis libraries.",
        body_style
    ))
    elements.append(Paragraph("Each crater is represented as a circular polygon with attributes:", body_style))
    shp_attrs = [
        "crater_id: Unique identifier",
        "center_x: X coordinate of center",
        "center_y: Y coordinate of center",
        "radius: Crater radius in map units",
        "diameter: Crater diameter in map units"
    ]
    shp_list = ListFlowable(
        [ListItem(Paragraph(a, body_style), leftIndent=20) for a in shp_attrs],
        bulletType='bullet',
        start='circle',
    )
    elements.append(shp_list)
    elements.append(Spacer(1, 0.2*inch))

    elements.append(PageBreak())

    # 7. Tips and Best Practices
    elements.append(Paragraph("7. Tips and Best Practices", heading1_style))

    tips = [
        "<b>Point Selection:</b> Select points that are evenly distributed around the crater rim "
        "(approximately 120° apart) for the most accurate circle fit.",

        "<b>Avoid Degraded Craters:</b> Heavily eroded or partially buried craters may not fit well "
        "to a perfect circle. Use your judgment on whether to include these.",

        "<b>Consistent Rim Selection:</b> Try to be consistent in selecting the same part of the rim "
        "(e.g., always the outer edge) across all craters for comparable measurements.",

        "<b>Regular Exports:</b> Although the tool auto-saves, export your data periodically as a backup.",

        "<b>Lighting Conditions:</b> If working with oblique lighting images, be aware that shadows "
        "may affect your perception of the true crater rim.",

        "<b>Scale Awareness:</b> Be aware of your image resolution and only mark craters that are "
        "well-resolved (typically >10 pixels in diameter).",

        "<b>Session Management:</b> For large datasets, work in sessions. The auto-save feature allows "
        "you to resume work later.",

        "<b>Coordinate Systems:</b> The tool preserves the coordinate reference system of the input "
        "image. Verify your image has proper CRS information for accurate spatial data."
    ]

    for tip in tips:
        elements.append(Paragraph(f"• {tip}", body_style))
        elements.append(Spacer(1, 0.1*inch))

    elements.append(Spacer(1, 0.2*inch))

    # 8. Troubleshooting
    elements.append(Paragraph("8. Troubleshooting", heading1_style))

    trouble_data = [
        ['Problem', 'Solution'],
        ['Failed to open image',
         'Ensure GDAL is properly installed. Check that the file format is supported (GeoTIFF or ISIS cube).'],
        ['Could not fit circle to points',
         'The three selected points are collinear (in a straight line). Select points that better define a circle.'],
        ['Points appear in wrong location',
         'The image may lack proper geospatial metadata. Check the GeoTIFF geotransform.'],
        ['Application is slow',
         'Very large images may be slow to load. Consider creating overviews or working with smaller tiles.'],
        ['Export fails',
         'Check that you have write permissions in the export directory. Ensure GeoPandas is installed.'],
        ['Cannot see marked craters',
         'Green circles may not be visible against certain backgrounds. This is a display issue only - data is still saved.'],
        ['Session not recovered',
         'Check if craters_autosave.json exists. It may have been deleted or corrupted.'],
    ]

    trouble_table = Table(trouble_data, colWidths=[2*inch, 4*inch])
    trouble_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1976d2')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f5f5f5')),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ('LEFTPADDING', (0, 1), (-1, -1), 8),
        ('RIGHTPADDING', (0, 1), (-1, -1), 8),
    ]))
    elements.append(trouble_table)
    elements.append(Spacer(1, 0.2*inch))

    elements.append(PageBreak())

    # 9. Technical Details
    elements.append(Paragraph("9. Technical Details", heading1_style))

    elements.append(Paragraph("9.1 Circle Fitting Algorithm", heading2_style))
    elements.append(Paragraph(
        "The tool uses a geometric algorithm to fit a circle through three non-collinear points. "
        "The algorithm calculates the perpendicular bisectors of the line segments connecting the "
        "points, and finds their intersection point, which is the circle center. The radius is "
        "then calculated as the distance from the center to any of the three points.",
        body_style
    ))

    elements.append(Paragraph("9.2 Coordinate Transformations", heading2_style))
    elements.append(Paragraph(
        "The tool uses GDAL's geotransform matrix to convert between pixel coordinates (used for "
        "display and clicking) and geographic/projected coordinates (used for measurements and export). "
        "This ensures that measurements are accurate regardless of the image's spatial reference system.",
        body_style
    ))

    elements.append(Paragraph("9.3 Supported File Formats", heading2_style))
    supported = [
        "<b>GeoTIFF:</b> Industry-standard format for georeferenced raster images. Must contain "
        "valid geospatial metadata.",
        "<b>ISIS Cube:</b> Planetary image format used by USGS Integrated Software for Imagers and "
        "Spectrometers (ISIS). Requires GDAL compiled with ISIS support."
    ]
    for item in supported:
        elements.append(Paragraph(f"• {item}", body_style))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("9.4 Dependencies", heading2_style))
    elements.append(Paragraph(
        "The tool relies on several open-source Python libraries:",
        body_style
    ))
    deps = [
        "<b>PyQt5:</b> GUI framework for the application interface",
        "<b>GDAL:</b> Geospatial Data Abstraction Library for reading image formats",
        "<b>NumPy:</b> Numerical computing for array operations",
        "<b>GeoPandas:</b> Geospatial data handling for shapefile export",
        "<b>Shapely:</b> Geometric operations for circle polygon creation",
        "<b>PyProj:</b> Cartographic projections and coordinate transformations"
    ]
    for dep in deps:
        elements.append(Paragraph(f"• {dep}", body_style))
    elements.append(Spacer(1, 0.2*inch))

    # 10. References
    elements.append(Paragraph("10. References", heading1_style))
    elements.append(Paragraph(
        "This tool was inspired by OpenCraterTool, an open-source QGIS plugin for crater analysis:",
        body_style
    ))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph(
        "Heyer, T., Kuhn, K., Wöhler, C., Klemm, S., &amp; Fawdon, P. (2023). OpenCraterTool: "
        "An open source QGIS plugin for crater size-frequency measurements. <i>Planetary and Space Science</i>, "
        "227, 105687. https://doi.org/10.1016/j.pss.2023.105687",
        body_style
    ))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph(
        "GitHub Repository: https://github.com/thomasheyer/OpenCraterTool",
        body_style
    ))
    elements.append(Spacer(1, 0.3*inch))

    # Footer
    elements.append(Spacer(1, 1*inch))
    elements.append(Paragraph(
        "For issues, questions, or contributions, please refer to the project repository.",
        ParagraphStyle('Footer', parent=styles['Normal'], fontSize=9, textColor=colors.grey, alignment=TA_CENTER)
    ))

    # Build PDF
    doc.build(elements)
    print(f"\n{'='*60}")
    print(f"PDF manual generated successfully: {filename}")
    print(f"{'='*60}\n")
    return filename


if __name__ == "__main__":
    create_pdf_manual()
