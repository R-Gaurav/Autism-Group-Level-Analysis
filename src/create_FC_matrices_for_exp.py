"""
This file creates the 5D FC matrices on which Group Level GLM is supposed to run.

To execute this file: python create_FC_matrice_for_exp.py

Note: Change the value of `is_abide1` variable to toggle between ABIDE1 & ABIDE2.
"""

import pickle
import pandas as pd
import nibabel as nib
import numpy as np
import numpy.linalg as npl
import time

from utility.dcp_utilities import (get_valid_subject_ids_with_complete_phenotype,
    get_asd_and_tdh_subs_row_indices, get_processed_design_matrix)
from utility.exp_utilities import (get_valid_subject_ids_wrt_fc_data,
    construct_subs_batchwise_rois_fc_5D_mats, get_regressors_mask_list,
    ROI_WITH_ZERO_FC_MAT)
from do_group_level_glm import do_group_level_glm
from do_statistical_analysis import do_statistical_analysis

pheno_data_path_1 = "/mnt/project1/home1/varunk/ramashish/data/phenotype_csvs/ABIDE1_Phenotypic.csv"
pheno_data_path_2 = "/mnt/project1/home1/varunk/ramashish/data/phenotype_csvs/ABIDE2_Phenotypic.csv"
fc_files_path1 = "/mnt/project2/home/varunk/fMRI/results/resultsABIDE1_4/functional_connectivity/calc_residual0smoothing1filt1calc_residual_options/"
fc_files_path2 = "/mnt/project2/home/varunk/fMRI/results/resultsABIDE2_1/functional_connectivity/calc_residual0smoothing1filt1calc_residual_options/"

is_abide1 = True # If abide1 is True, it prepares FC files for ABIDE1 else ABIDE2.

if is_abide1:
  pheno_data_path = pheno_data_path_1
  fc_files_path = fc_files_path1
  abide = "ABIDE-1"
  buggy_subs_file = "abide_1_bug_sub_ids_from_varun.p"
else:
  pheno_data_path = pheno_data_path_2
  fc_files_path = fc_files_path2
  abide = "ABIDE-2"
  buggy_subs_file = "abide_2_bug_sub_ids_from_varun.p"

print "FC files creation started for %s dataset" % abide

################################################################################
############ Get the row IDs of all valid subjects ############
################################################################################

asd_subs, tdh_subs = get_valid_subject_ids_with_complete_phenotype(
    pheno_data_path, is_abide1)
all_subs = asd_subs + tdh_subs

print "Number of subjects in %s dataset with complete phenotypic data: %s" % (
    abide, len(all_subs))

buggy_subs = pickle.load(
    open("/mnt/project1/home1/varunk/ramashish/data/buggy_subjects/%s"
    % buggy_subs_file, "rb"))

print "Removing buggy subs in file: %s" % buggy_subs_file
all_subs = list(set(all_subs) - set(buggy_subs))

print "After removal of buggy subs, number of subs: %s" % len(all_subs)

print "Now getting the valid subs who have FC data available..."
valid_subs, no_fc_subs, zero_fc_subs = get_valid_subject_ids_wrt_fc_data(
    fc_files_path, all_subs, is_abide1)

print ("Number of valid subs in %s dataset for whom FC matrices would be "
       "created: %s, number of subs with NO FC data: %s, number of subs with "
       "zero FC data for ROIs other than %s ROI: %s" % (abide, len(valid_subs),
       len(no_fc_subs), ROI_WITH_ZERO_FC_MAT, len(zero_fc_subs)))

df = pd.read_csv(pheno_data_path)
asd_row_ids, tdh_row_ids = get_asd_and_tdh_subs_row_indices(df, valid_subs)

print "Row IDs of ASD and TDH subs obtained!"

################################################################################
############# Creation of batchwise subs ROIs FC matrices #############
################################################################################

start = time.time()
construct_subs_batchwise_rois_fc_5D_mats(
    fc_files_path, df, asd_row_ids[:5], tdh_row_ids[:5], 32, is_abide1)
print "Hurray! FC matrices for %s dataset created! Time taken: %s" % (
    abide, time.time() - start)

################################################################################
################################################################################
