import gzip
import os
import glob

data_dir = os.path.expanduser("~/liver_cirrhosis/data/PRJNA1019460")
files = glob.glob(os.path.join(data_dir, "*.fastq.gz"))

for filepath in sorted(files):
    if "_1" in filepath or "_2" in filepath:
        continue
    
    sample = os.path.basename(filepath).replace(".fastq.gz", "")
    out_r1 = os.path.join(data_dir, f"{sample}_1.fastq.gz")
    out_r2 = os.path.join(data_dir, f"{sample}_2.fastq.gz")
    
    print(f"Splitting {sample}...")
    
    with gzip.open(filepath, 'rt') as infile, \
         gzip.open(out_r1, 'wt') as r1, \
         gzip.open(out_r2, 'wt') as r2:
        
        read_num = 0
        lines = []
        
        for line in infile:
            lines.append(line)
            if len(lines) == 4:
                read_num += 1
                if read_num % 2 == 1:
                    r1.writelines(lines)
                else:
                    r2.writelines(lines)
                lines = []
    
    os.remove(filepath)
    print(f"{sample} done.")

print("All files split successfully.")
