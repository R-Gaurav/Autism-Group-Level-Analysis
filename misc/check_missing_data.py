import os
import sys
import pandas as pd

drc = sys.argv[1]
pheno_file = sys.argv[2]

df = pd.read_csv(pheno_file, encoding="ISO-8859-1")
print("DF Shape of file: %s is: %s" % (pheno_file, df.shape))
sub_ids = df["SUB_ID"].tolist()
print("Number of subjects = %s" % len(sub_ids))

files = os.listdir(drc)
print("Number of nii.gz files: %s" % len(files))
file_sub_ids = [int(fl.split("_")[0].split("-")[1]) for fl in files]
for sub_id in sub_ids:
  if sub_id not in file_sub_ids:
    print(sub_id)
