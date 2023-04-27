#!/bin/bash

# 1. Install
# Create a new conda environment, source environment, install mfa and run:
# conda create -n aligner python=3
# conda activate aligner
# conda install -c conda-forge montreal-forced-aligner

# 2. Download acoustic-model/dictionary
# mfa model download acoustic english_us_arpa
# mfa model download dictionary english_us_arpa

# 3. validate corpus and align
# corpus="data/dev/audio"
# alignpath="data/dev/alignment"

root=$1

# Check if the first argument ends with a slash
if [[ $root == */ ]]; then
    root=${root::-1}
fi

corpus=$root/audio
alignpath=$root/alignment

# Abort if $corpus is not a directory
if [ ! -d "$corpus" ]; then
    echo "Error: $corpus is not a directory"
    exit 1
fi


# echo "Validating $corpus"
# mfa validate $corpus english_us_arpa english_us_arpa

# -j n_jobs
echo "Alignment $corpus -> $alignpath"
conda run -n aligner mfa align $corpus english_us_arpa english_us_arpa $alignpath --clean
