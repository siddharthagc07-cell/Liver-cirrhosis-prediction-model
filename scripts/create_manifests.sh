#!/bin/bash

# Dataset 1 - PRJNA1019460
MANIFEST1=~/liver_cirrhosis/docs/manifest_PRJNA1019460.tsv
echo -e "sample-id\tforward-absolute-filepath\treverse-absolute-filepath" > $MANIFEST1

for f in ~/liver_cirrhosis/data/PRJNA1019460/*_1.fastq.gz; do
    SAMPLE=$(basename "$f" _1.fastq.gz)
    R1="$f"
    R2="${f/_1.fastq.gz/_2.fastq.gz}"
    echo -e "${SAMPLE}\t${R1}\t${R2}" >> $MANIFEST1
done

# Dataset 2 - PRJNA471972
MANIFEST2=~/liver_cirrhosis/docs/manifest_PRJNA471972.tsv
echo -e "sample-id\tforward-absolute-filepath\treverse-absolute-filepath" > $MANIFEST2

for f in ~/liver_cirrhosis/data/PRJNA471972/*_1.fastq.gz; do
    SAMPLE=$(basename "$f" _1.fastq.gz)
    R1="$f"
    R2="${f/_1.fastq.gz/_2.fastq.gz}"
    echo -e "${SAMPLE}\t${R1}\t${R2}" >> $MANIFEST2
done

# Dataset 3 - PRJNA1259947
MANIFEST3=~/liver_cirrhosis/docs/manifest_PRJNA1259947.tsv
echo -e "sample-id\tforward-absolute-filepath\treverse-absolute-filepath" > $MANIFEST3

for f in ~/liver_cirrhosis/data/PRJNA1259947/*_1.fastq.gz; do
    SAMPLE=$(basename "$f" _1.fastq.gz)
    R1="$f"
    R2="${f/_1.fastq.gz/_2.fastq.gz}"
    echo -e "${SAMPLE}\t${R1}\t${R2}" >> $MANIFEST3
done

echo "All manifests created."
