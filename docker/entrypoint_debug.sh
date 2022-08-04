#!/bin/bash

# run tests in live debug
source ../../opt/conda/etc/profile.d/conda.sh
conda activate ASIM-DEV
#python -m debugpy --listen 0.0.0.0:5678 --wait-for-client ../../opt/conda/envs/ASIM-DEV/bin/pytest test/auto_ownership/test_auto_ownership.py
python -m debugpy --listen 0.0.0.0:5678 --wait-for-client ../../opt/conda/envs/ASIM-DEV/bin/pytest test/parking_location/test_parking_location.py