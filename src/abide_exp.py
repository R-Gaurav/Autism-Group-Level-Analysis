#
# Author: Ramashish Gaurav
#
# This file conducts the experiment for ABIDE I and ABIDE II dataset. It does
# the group level GLM analysis and statistical analysis.
# Note: Accordingly change the value of `phenotype_csv` variable when using ABIDE
# I and ABIDE II Phenotype CSV files.
#
# The order of columns in regressors_matrix is: "ASD", "TDH", "LEFT_H", "RIGHT_H",
# "EYE_OPEN", "EYE_CLOSED", "FIQ".
#

import numpy as np
import pandas as pd

from utility.dcp_utilities import (
    get_no_missing_vals_df, get_regressors_matrix,
    get_row_indices_without_discarded_value_list)

#phenotype_csv = "../data/Phenotypic_V1_0b.csv"
phenotype_csv = "../data/ABIDEII_Composite_Phenotypic.csv"

pheno_df = pd.read_csv(phenotype_csv)
# Remove the NA values from phenotype_df with respect to the considered regressor
# columns (Note: Column names are mentioned are in get_no_missing_vals_df func).
pheno_no_na_df = get_no_missing_vals_df(pheno_df)

# Discard the "-9999" values in "HANDEDNESS_CATEGORY", -9999 values in "FIQ"
# column from pheno_no_na_df.
row_indices_no_discarded_list = get_row_indices_without_discarded_value_list(
    pheno_no_na_df, [("HANDEDNESS_CATEGORY", "-9999"), ("FIQ", -9999)])
pheno_no_na_no_dval_df = pheno_no_na_df.loc[row_indices_no_discarded_list]

# Get the ASD and TDH subjects indices list and regressor matrix.
# DX_GROUP = 1 => ASD and DX_GROUP = 2 => TDH.
asd_subjects_list = pheno_no_na_no_dval_df.loc[
    pheno_no_na_no_dval_df["DX_GROUP"] == 1].index.tolist()
tdh_subjects_list = pheno_no_na_no_dval_df.loc[
    pheno_no_na_no_dval_df["DX_GROUP"] == 2].index.tolist()
reg_clm_list = ["FIQ"]
regressors_matrix = get_regressors_matrix(
    pheno_no_na_no_dval_df, asd_subjects_list, tdh_subjects_list, reg_clm_list)
for sub_index, sub_pheno in zip(
    asd_subjects_list + tdh_subjects_list, regressors_matrix):
  print pheno_no_na_no_dval_df.loc[sub_index, "SUB_ID"], sub_pheno
