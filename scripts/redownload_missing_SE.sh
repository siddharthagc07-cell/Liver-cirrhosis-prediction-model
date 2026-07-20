#!/bin/bash
OUTPUT_DIR=~/liver_cirrhosis/data/PRJNA1019460
cd $OUTPUT_DIR

echo "Re-downloading 26 missing single-end files..."

for SRR in SRR26159182 SRR26159196 SRR26159197 SRR26159199 SRR26159200 \
            SRR26159201 SRR26159202 SRR26159203 SRR26159204 SRR26159205 \
            SRR26159206 SRR26159207 SRR26159208 SRR26159209 SRR26159210 \
            SRR26159211 SRR26159212 SRR26159213 SRR26159214 SRR26159215 \
            SRR26159216 SRR26159217 SRR26159218 SRR26159219 SRR26159220 \
            SRR26159221
do
    echo "Fetching link for $SRR..."
    wget -q "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=${SRR}&result=read_run&fields=fastq_ftp&format=tsv" -O ${SRR}_links.txt

    LINKS=$(tail -1 ${SRR}_links.txt | tr ';' '\n' | grep fastq)

    for LINK in $LINKS; do
        wget -c \
            --timeout=60 \
            --tries=10 \
            --waitretry=30 \
            --retry-connrefused \
            "https://${LINK}"
    done

    rm -f ${SRR}_links.txt
    echo "$SRR done."
done

echo "All 26 missing samples re-downloaded."
