#!/bin/bash

OUTPUT_DIR=~/liver_cirrhosis/data/PRJNA1019460
cd $OUTPUT_DIR

echo "Re-downloading missing paired-end files..."

for SRR in SRR26159183 SRR26159184 SRR26159185 SRR26159186 SRR26159187 \
            SRR26159188 SRR26159189 SRR26159190 SRR26159191 SRR26159192 \
            SRR26159193 SRR26159194 SRR26159195 SRR26159198
do
    echo "Downloading $SRR R1 and R2..."
    wget -c "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=${SRR}&result=read_run&fields=fastq_ftp&format=tsv" -O ${SRR}_links.txt
    
    # Extract the FTP links
    LINKS=$(tail -1 ${SRR}_links.txt | tr ';' '\n')
    
    for LINK in $LINKS; do
        wget -c "https://${LINK}"
    done
    
    rm ${SRR}_links.txt
    echo "$SRR done."
done

echo "All missing samples re-downloaded."
