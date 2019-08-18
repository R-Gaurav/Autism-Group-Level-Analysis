#
# Author: Ramashish Gaurav
#
# This file conducts the experiment for ABIDE I and ABIDE II dataset. It does
# the group level GLM analysis and statistical analysis.
# Note: Accordingly change the value of `phenotype_csv` variable when using ABIDE
# I and ABIDE II Phenotype CSV files.
#
# The order of columns in regressors_matrix is: "ASD", "TDH", "LEFT_H", "RIGHT_H",
# "EYE_OPEN", "EYE_CLOSED", "FIQ", "AGE_AT_SCAN".
#
import numpy.linalg as npl
import numpy as np
import numpy.linalg as npl
import pandas as pd
import pickle

from utility.dcp_utilities import (
    get_no_missing_vals_df, get_phenotypes_regressors_matrix,
    get_row_indices_without_discarded_value_list)

phenotype_csv = "/mnt/project1/home1/varunk/ramashish/data/ABIDE1_Phenotypic.csv"

pheno_df = pd.read_csv(phenotype_csv, encoding="ISO-8859-1")
# Remove the NA values from phenotype_df with respect to the considered regressor
# columns (Note: Column names are mentioned are in get_no_missing_vals_df func).
pheno_no_na_df = get_no_missing_vals_df(pheno_df)

# Discard the "-9999" values in "HANDEDNESS_CATEGORY", -9999 values in "FIQ"
# column from pheno_no_na_df. Note that "HANDEDNESS_CATEGORY" in ABIDE II is a
# float, and is a string in ABIDE I. Therefore accordingly change the type of
# the value to be discarded.
row_indices_no_discarded_list = get_row_indices_without_discarded_value_list(
    pheno_no_na_df, [("HANDEDNESS_CATEGORY", "-9999"), ("FIQ", -9999),
    ("EYE_STATUS_AT_SCAN", 0)])
pheno_no_na_no_dval_df = pheno_no_na_df.loc[row_indices_no_discarded_list]

print("Dataframe shape after discarding the values:",
      pheno_no_na_no_dval_df.shape)

# Get the ASD and TDH subjects indices list and regressor matrix.
# DX_GROUP = 1 => ASD and DX_GROUP = 2 => TDH. (in ABIDE1 and ABIDE2)
asd_subjects_list = pheno_no_na_no_dval_df.loc[
    pheno_no_na_no_dval_df["DX_GROUP"] == 1].index.tolist()
tdh_subjects_list = pheno_no_na_no_dval_df.loc[
    pheno_no_na_no_dval_df["DX_GROUP"] == 2].index.tolist()

# The regressors matris has first the ASD subjects data followed by TDH subjects.
regressors_matrix = get_phenotypes_regressors_matrix(
    pheno_no_na_no_dval_df, asd_subjects_list, tdh_subjects_list)
print("Shape of Design Matrix: ", regressors_matrix.shape)
print("RANK: ", npl.matrix_rank(regressors_matrix))
print("Condition number of X.T.dot(X): ", npl.cond(
    regressors_matrix.T.dot(regressors_matrix)), npl.cond(regressors_matrix))
pd.DataFrame(regressors_matrix, columns=["ASD_SUBS","TDH_SUBS", "LEFT_H",
             "RIGHT_H",	"EYE_OPEN", "EYE_CLOSED",	"FIQ","AGE_AT_SCAN"]).to_csv(
             "ABIDE_1_design_matrix.csv")
pd.DataFrame(
    pheno_no_na_no_dval_df.loc[asd_subjects_list+tdh_subjects_list]["SUB_ID"]
    ).to_csv("ABIDE_1_sub_ids.csv")
#for sub_index, sub_pheno in zip(
#    asd_subjects_list + tdh_subjects_list, regressors_matrix):
#  print pheno_no_na_no_dval_df.loc[sub_index, "SUB_ID"], sub_pheno
