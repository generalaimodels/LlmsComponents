#!/bin/bash

# Clone the Segment Anything repository
git clone https://github.com/facebookresearch/segment-anything-2.git

# Change directory to the cloned repository and install the package
cd segment-anything-2 && pip install -e .

# Navigate to the checkpoints directory and download model checkpoints
cd checkpoints && ./download_ckpts.sh && cd ..

# Optionally: Confirm completion
echo "Model checkpoints have been downloaded successfully."