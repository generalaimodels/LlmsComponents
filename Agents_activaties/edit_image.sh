#!/usr/bin/env bash

git clone https://github.com/sail-sg/EditAnything.git
cd EditAnything
set -euo pipefail

# Define color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to log messages
log() {
    local level=$1
    shift
    echo -e "${!level}[${level}]${NC} $*"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for required commands
for cmd in conda wget pip; do
    if ! command_exists "$cmd"; then
        log RED "Error: $cmd is not installed. Please install it and try again."
        exit 1
    fi
done

# Create and activate conda environment
create_conda_env() {
    log YELLOW "Creating conda environment..."
    conda env create -f environment.yaml
    conda activate control || {
        log RED "Failed to activate conda environment. Exiting."
        exit 1
    }
    log GREEN "Conda environment created and activated successfully."
}

# Install required packages
install_packages() {
    log YELLOW "Installing required packages..."
    pip install --no-cache-dir \
        git+https://github.com/huggingface/transformers.git \
        git+https://github.com/facebookresearch/segment-anything.git \
        git+https://github.com/openai/CLIP.git \
        git+https://github.com/facebookresearch/detectron2.git \
        git+https://github.com/IDEA-Research/GroundingDINO.git
    log GREEN "Packages installed successfully."
}

# Download pretrained models
download_models() {
    local models_dir="models"
    log YELLOW "Downloading pretrained models..."
    mkdir -p "$models_dir"
    wget -P "$models_dir" \
        https://github.com/Cheems-Seminar/segment-anything-and-name-it/releases/download/v1.0/swinbase_part_0a0000.pth \
        https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
    log GREEN "Models downloaded successfully."
}

# Main function to run the demo
run_demo() {
    local script=$1
    log YELLOW "Running demo: $script"
    python "$script"
    log GREEN "Demo completed."
}

# Main execution
main() {
    create_conda_env
    install_packages
    download_models

    # List of available demo scripts
    local demos=("app.py" "editany.py" "sam2image.py" "sam2vlpart_edit.py" "sam2groundingdino_edit.py")

    # Prompt user to select a demo
    echo "Available demos:"
    for i in "${!demos[@]}"; do
        echo "$((i+1)). ${demos[i]}"
    done

    read -p "Enter the number of the demo you want to run (1-${#demos[@]}): " choice

    if [[ "$choice" =~ ^[1-${#demos[@]}]$ ]]; then
        run_demo "${demos[$((choice-1))]}"
    else
        log RED "Invalid choice. Exiting."
        exit 1
    fi
}

# Run the main function
main