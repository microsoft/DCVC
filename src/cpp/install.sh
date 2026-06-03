#!/bin/bash

start=$(date +%s)
pip install --no-build-isolation .
end=$(date +%s)
runtime=$((end - start))
echo "Execution time: $runtime seconds"
