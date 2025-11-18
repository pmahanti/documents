#!/bin/bash
# Setup script for Crater Marker Tool

echo "Crater Marker Tool - Setup"
echo "=========================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
echo "Note: GDAL installation may take some time..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================="
    echo "Setup completed successfully!"
    echo ""
    echo "To run the application:"
    echo "  1. Activate the virtual environment: source venv/bin/activate"
    echo "  2. Run the application: python crater_marker.py"
    echo ""
    echo "To generate test data:"
    echo "  python generate_test_data.py"
    echo ""
    echo "See USER_GUIDE.md for detailed instructions."
else
    echo ""
    echo "=========================="
    echo "Installation encountered errors."
    echo "You may need to install GDAL system packages first."
    echo "See USER_GUIDE.md for platform-specific instructions."
fi
