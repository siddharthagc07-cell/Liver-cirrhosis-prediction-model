#!/bin/bash

INPUT_DIR=~/liver_cirrhosis/data/PRJNA1019460
cd $INPUT_DIR

echo "Splitting interleaved files..."

for f in *.fastq.gz; do
    SAMPLE="${f%.fastq.gz}"
    echo "Splitting $SAMPLE..."
    seqtk seq -1 "$f" | gzip > "${SAMPLE}_1.fastq.gz"
    seqtk seq -2 "$f" | gzip > "${SAMPLE}_2.fastq.gz"
    rm "$f"
    echo "$SAMPLE done."
done

echo "All files split successfully."
