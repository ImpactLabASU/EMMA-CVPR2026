#!/bin/bash

# Rover Architecture Ablation Study Runner
# This script runs video processing and then the architecture ablation study

echo "======================================"
echo "ROVER ARCHITECTURE ABLATION STUDY"
echo "======================================"
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Check if video processing is needed
if [ ! -f "data/xData.txt" ]; then
    echo "Video data not found. Running video processing first..."
    echo ""
    python3 rover-ablation.py --skip-video=false
    echo ""
    echo "Video processing complete. Starting ablation study..."
    echo ""
else
    echo "Using existing video data. Starting ablation study..."
    echo ""
fi

# Run architecture ablation study
python3 rover-ablation.py --ablation --skip-video

echo ""
echo "======================================"
echo "ABLATION STUDY COMPLETE!"
echo "Results saved to: rover_architecture_ablation_results.csv"
echo "======================================"

