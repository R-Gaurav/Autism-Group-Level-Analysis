#!/bin/bash

echo "Script to run fsl_motion_outliers on the output data of MCFLIRT"
input_dir=$1
output_dir=$2
metric=$3

num_files=0

echo "Running fsl_motion_outliers for all the 4D functional files in: $input_dir"

for file in $input_dir/*.nii.gz; do
  file_name=$(echo $file | cut -d'/' -f 10)
  file_name=$(echo $file_name | cut -d'.' -f 1)

  confounds_file="fsl_mo_$file_name""_""$metric""_conf.txt"
  mval_file_name="fsl_mo_$file_name""_""$metric""_mval.txt"
  confounds_file_path="$output_dir""/""$confounds_file"
  mval_file_name_path="$output_dir""/""$mval_file_name"

  cmd="fsl_motion_outliers -i $file -o $confounds_file_path --$metric"
  echo $confounds_file
  eval $cmd

  num_files=$((num_files+1))
  if (( $num_files == 10 )); then
    break
  fi
  done

echo "Number of files on which fsl_motion_outliers ran: $num_files"
