#!/bin/bash


nvidia-smi --query-compute-apps="vertical_split" --format=csv,noheader | wc -l




echo "${process_count[0]}"
echo "${process_count[1]}"
echo "${process_count[2]}"
echo "${process_count[3]}"





