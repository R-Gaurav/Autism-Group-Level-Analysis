#
# Autism Group Level Analysis
#
# Author: Ramashish Gaurav
#
# This file contains the constants used throughout the code.
#

ROI_WITH_ZERO_FC_MAT = 254 # 0 based indexing.
TOTAL_NUMBER_OF_ROIS = 274
BATCH_SIZE = 32
ABIDE_1 = "ABIDE-1"
ABIDE_2 = "ABIDE-2"

# Experiment Data Paths.
OUTPUT_FILE_PATH = "/mnt/project1/home1/varunk/ramashish/second_level_analysis_data/"
LOG_FILE_PATH = "/mnt/project1/home1/varunk/ramashish/misc/logs/"
PHENO_DATA_PATH_1 = "/mnt/project1/home1/varunk/ramashish/data/phenotype_csvs/ABIDE1_Phenotypic.csv"
PHENO_DATA_PATH_2 = "/mnt/project1/home1/varunk/ramashish/data/phenotype_csvs/ABIDE2_Phenotypic.csv"
FC_FILES_PATH_1 = "/mnt/project2/home/varunk/fMRI/results/resultsABIDE1_4/functional_connectivity/calc_residual0smoothing1filt1calc_residual_options/"
FC_FILES_PATH_2 = "/mnt/project2/home/varunk/fMRI/results/resultsABIDE2_1/functional_connectivity/calc_residual0smoothing1filt1calc_residual_options/"

# Experiment Data Dirs.
ABIDE_1_SUBS_MD = "/all_subs_metadata_ABIDE1/"
ABIDE_2_SUBS_MD = "/all_subs_metadata_ABIDE2/"
ABIDE_1_BW_BE_TPL_DATA = "/all_valid_subs_batch_wise_rois_be_tuple_data_ABIDE1/"
ABIDE_2_BW_BE_TPL_DATA = "/all_valid_subs_batch_wise_rois_be_tuple_data_ABIDE2/"
ABIDE_1_BW_ROI_FC_DATA = "/all_valid_subs_batch_wise_rois_fc_data_ABIDE1/"
ABIDE_2_BW_ROI_FC_DATA = "/all_valid_subs_batch_wise_rois_fc_data_ABIDE2/"
ABIDE_1_BW_TP_TPL_DATA = "/all_valid_subs_batch_wise_rois_tp_tuple_data_ABIDE1/"
ABIDE_2_BW_TP_TPL_DATA = "/all_valid_subs_batch_wise_rois_tp_tuple_data_ABIDE2/"
