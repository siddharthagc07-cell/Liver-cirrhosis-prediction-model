#!/bin/bash

# Dataset: PRJNA1019460
# Paper: Chen et al. 2023 - Compensated Liver Cirrhosis vs Healthy
# Samples: 40 total
# Source: ENA FTP (exact paths from API)

OUTPUT_DIR=~/liver_cirrhosis/data/PRJNA1019460
cd $OUTPUT_DIR

echo "Starting download of PRJNA1019460..."

wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/082/SRR26159182/SRR26159182.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/085/SRR26159185/SRR26159185.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/086/SRR26159186/SRR26159186.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/087/SRR26159187/SRR26159187.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/088/SRR26159188/SRR26159188.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/090/SRR26159190/SRR26159190.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/095/SRR26159195/SRR26159195.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/099/SRR26159199/SRR26159199.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/003/SRR26159203/SRR26159203.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/004/SRR26159204/SRR26159204.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/007/SRR26159207/SRR26159207.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/008/SRR26159208/SRR26159208.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/009/SRR26159209/SRR26159209.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/011/SRR26159211/SRR26159211.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/013/SRR26159213/SRR26159213.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/015/SRR26159215/SRR26159215.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/083/SRR26159183/SRR26159183.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/091/SRR26159191/SRR26159191.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/092/SRR26159192/SRR26159192.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/094/SRR26159194/SRR26159194.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/096/SRR26159196/SRR26159196.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/097/SRR26159197/SRR26159197.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/000/SRR26159200/SRR26159200.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/001/SRR26159201/SRR26159201.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/005/SRR26159205/SRR26159205.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/006/SRR26159206/SRR26159206.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/012/SRR26159212/SRR26159212.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/016/SRR26159216/SRR26159216.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/019/SRR26159219/SRR26159219.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/084/SRR26159184/SRR26159184.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/089/SRR26159189/SRR26159189.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/093/SRR26159193/SRR26159193.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/098/SRR26159198/SRR26159198.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/002/SRR26159202/SRR26159202.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/010/SRR26159210/SRR26159210.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/014/SRR26159214/SRR26159214.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/017/SRR26159217/SRR26159217.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/018/SRR26159218/SRR26159218.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/020/SRR26159220/SRR26159220.fastq.gz
wget -c https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR261/021/SRR26159221/SRR26159221.fastq.gz

echo "All 40 samples downloaded successfully."
