#!/bin/bash
# Quick start script for lunar volatile sublimation calculator

echo "========================================"
echo "Lunar Volatile Sublimation Calculator"
echo "Quick Start Examples"
echo "========================================"
echo ""

echo "1. Water ice at 110K (typical PSR temperature):"
python vaporp_temp.py -t 110 -s H2O
echo ""
echo "Press Enter to continue..."
read

echo "2. All volatile species at 100K:"
python vaporp_temp.py -t 100 --all
echo ""
echo "Press Enter to continue..."
read

echo "3. Temperature range for water ice (40K to 120K):"
python vaporp_temp.py -t 40 60 80 100 120 -s H2O
echo ""
echo "Press Enter to continue..."
read

echo "4. Running comprehensive examples (generates CSV):"
python examples.py
echo ""

echo "========================================"
echo "Quick start completed!"
echo "See README.md for more information"
echo "========================================"
