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
EXP_DIR = "/mnt/scratch1/ramashish/"
FIRST_LEVEL_DATA = "/calc_residual1smoothing1filt1calc_residual_optionsconstcsfwmmotionglobal/"
BINARIZED_ATLAS_MP = "/mnt/project2/home/varunk/fMRI/atlas/Full_brain_atlas_thr0-2mm/fullbrain_atlas_thr0-3mm_binarized.nii.gz"
BRAIN_HEADER_AND_AFFINE = "/mnt/project2/home/varunk/fMRI/results/resultsABIDE2_1/fc_datasink/" + FIRST_LEVEL_DATA + "fc_map_brain_file_list.npy"
BRAIN_X = 61 # x dimension of the brain.
BRAIN_Y = 73 # y dimension of the brain.
BRAIN_Z = 61 # z dimension of the brain.

# Experiment Data Paths.
OUTPUT_FILE_PATH = EXP_DIR + "/second_level_analysis_data/"
LOG_FILE_PATH = EXP_DIR + "/misc/logs/"
PHENO_DATA_PATH_1 = EXP_DIR + "/data/phenotype_csvs/ABIDE1_Phenotypic.csv"
PHENO_DATA_PATH_2 = EXP_DIR + "/data/phenotype_csvs/ABIDE2_Phenotypic.csv"
FC_FILES_PATH_1 = "/mnt/project2/home/varunk/fMRI/results/resultsABIDE1_4/functional_connectivity/" + FIRST_LEVEL_DATA
FC_FILES_PATH_2 = "/mnt/project2/home/varunk/fMRI/results/resultsABIDE2_1/functional_connectivity/" + FIRST_LEVEL_DATA

# Experiment Data Dirs.
ABIDE_1_SUBS_MD = "/all_subs_metadata_ABIDE1/"
ABIDE_2_SUBS_MD = "/all_subs_metadata_ABIDE2/"
ABIDE_1_BW_BE_TPL_DATA = "/all_valid_subs_batch_wise_rois_be_tuple_data_ABIDE1/"
ABIDE_2_BW_BE_TPL_DATA = "/all_valid_subs_batch_wise_rois_be_tuple_data_ABIDE2/"
ABIDE_1_BW_ROI_FC_DATA = "/all_valid_subs_batch_wise_rois_fc_data_ABIDE1/"
ABIDE_2_BW_ROI_FC_DATA = "/all_valid_subs_batch_wise_rois_fc_data_ABIDE2/"
ABIDE_1_BW_TP_TPL_DATA = "/all_valid_subs_batch_wise_rois_tp_tuple_data_ABIDE1/"
ABIDE_2_BW_TP_TPL_DATA = "/all_valid_subs_batch_wise_rois_tp_tuple_data_ABIDE2/"
ABIDE_1_ALL_TSTAT_DATA = "/all_valid_subs_all_rois_t_stat_3d_mat_data_ABIDE1/"
ABIDE_2_ALL_TSTAT_DATA = "/all_valid_subs_all_rois_t_stat_3d_mat_data_ABIDE2/"
ABIDE_1_ALL_PSTAT_DATA = "/all_valid_subs_all_rois_p_stat_3d_mat_data_ABIDE1/"
ABIDE_2_ALL_PSTAT_DATA = "/all_valid_subs_all_rois_p_stat_3d_mat_data_ABIDE2/"
ABIDE_1_ALL_ROIS_CSV_REP = "/all_valid_subs_all_rois_csv_report_ABIDE1/"
ABIDE_2_ALL_ROIS_CSV_REP = "/all_valid_subs_all_rois_csv_report_ABIDE2/"
ABIDE_1_FDR_CRCTN_OUTPUT = "fdr_correction_output_ABIDE1"
ABIDE_2_FDR_CRCTN_OUTPUT = "fdr_correction_output_ABIDE2"
ABIDE_1_ALL_ROIS_LOGQ_MATS = "/all_valid_subs_all_rois_logq_mats_ABIDE1/"
ABIDE_2_ALL_ROIS_LOGQ_MATS = "/all_valid_subs_all_rois_logq_mats_ABIDE2/"
