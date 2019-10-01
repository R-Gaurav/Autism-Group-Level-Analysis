#
# Autism Group Level Analysis
#
# Author: Ramashish Gaurav
#
# This file does the FDR (multiple comparison correction) correction over the
# p values to obtain the q values and subsequently the q' values  (defined in
# Varun's MSR thesis). This file also creates the CSV reports with respect to
# metrics relating to p and q values.
#
# NOTE: Only p values are used to calculate q and q' values.

import datetime
import nibabel as nib
import numpy as np

from utility import log
from utility.constants import (ABIDE_1_ALL_PSTAT_DATA, ABIDE_1_ALL_TSTAT_DATA,
    ABIDE_2_ALL_PSTAT_DATA, ABIDE_2_ALL_TSTAT_DATA, OUTPUT_FILE_PATH,
    LOG_FILE_PATH, ABIDE_1, ABIDE_2, TOTAL_NUMBER_OF_ROIS, ROI_WITH_ZERO_FC_MAT,
    FIRST_LEVEL_DATA, ABIDE_1_SUBS_MD, ABIDE_2_SUBS_MD, BINARIZED_ATLAS_MP,
    ABIDE_1_FDR_CRCTN_OUTPUT, ABIDE_2_FDR_CRCTN_OUTPUT, BRAIN_HEADER_AND_AFFINE)
from utility.dcp_utilities import (
    get_4d_stat_matrix_of_all_ROIs, get_4D_subjects_mean_correlation_vals_mat)
from utility.fdrBrainResultsModular import fdr_correction_and_viz

from create_design_matrix_and_contrast_for_exp import (
    get_design_matrix_for_the_exp, get_contrast_vector_for_exp)

is_abide1 = False # If is_abide1 = True, get 3D stat mat for ABIDE1 else ABIDE2.

if is_abide1:
  abide = ABIDE_1
  all_rois_p_stat_data = ABIDE_1_ALL_PSTAT_DATA
  all_rois_t_stat_data = ABIDE_1_ALL_TSTAT_DATA
  all_subs_metadata = ABIDE_1_SUBS_MD
  fdr_correction_output = ABIDE_1_FDR_CRCTN_OUTPUT
else:
  abide = ABIDE_2
  all_rois_p_stat_data = ABIDE_2_ALL_PSTAT_DATA
  all_rois_t_stat_data = ABIDE_2_ALL_TSTAT_DATA
  all_subs_metadata = ABIDE_2_SUBS_MD
  fdr_correction_output = ABIDE_2_FDR_CRCTN_OUTPUT

log.configure_log_handler(
    "%s_%s.log" % (LOG_FILE_PATH + abide + FIRST_LEVEL_DATA + __file__,
    datetime.datetime.now()))

log.INFO("Creating and saving q value 3D matrices for %s" % abide)

################################################################################
################ Get the design matrix ###################
################################################################################

log.INFO("Obtaining %s design matrix" % abide)
design_matrix, regressors_strv = get_design_matrix_for_the_exp(is_abide1)
log.INFO("Regressors String Vec: %s" % regressors_strv)
################################################################################
################ Get the contrast vector #################
################################################################################

contrast, contrast_str = get_contrast_vector_for_exp()
log.INFO("Contrast vector: %s" % contrast)
# Assert the size of contrast vector is equal to the number of cols in dsgn mat.
assert contrast.shape[0] == design_matrix.shape[1]
################################################################################
################ Start the FDR correction on p vals ###################
################################################################################

log.INFO("Obtaining the 4D p value matrix of all ROIs")
# Dimension of 4D matrix: bx x by x bz x TOTAL_NUMBER_OF_ROIS
all_rois_p_val_4d_mat = get_4d_stat_matrix_of_all_ROIs(
    "p", OUTPUT_FILE_PATH + abide + FIRST_LEVEL_DATA + all_rois_p_stat_data,
    regressors_strv, contrast_str)

log.INFO("Obtaining the 4D t value matrix of all ROIs")
# Dimension of 4D matrix: bx x by x bz x TOTAL_NUMBER_OF_ROIS
all_rois_t_val_4d_mat = get_4d_stat_matrix_of_all_ROIs(
    "t", OUTPUT_FILE_PATH + abide + FIRST_LEVEL_DATA + all_rois_t_stat_data,
    regressors_strv, contrast_str)

log.INFO("Obtaining the 4D matrix of ASD subs mean FC for each ROI")
# Dimension of 4D matrix: bx x by x bz x TOTAL_NUMBER_OF_ROIS
all_asd_subs_mean_fc_4d_mat = np.load(
    OUTPUT_FILE_PATH + abide + FIRST_LEVEL_DATA + all_subs_metadata +
    "asd_mean_fc_all_rois_4d_mat.npy")

log.INFO("Obtaining the 4D matrix of TDH subs mean FC for each ROI")
# Dimension of 4D matrix: bx x by x bz x TOTAL_NUMBER_OF_ROIS
all_tdh_subs_mean_fc_4d_mat = np.load(
    OUTPUT_FILE_PATH + abide + FIRST_LEVEL_DATA + all_subs_metadata +
    "tdh_mean_fc_all_rois_4d_mat.npy")

log.INFO("Obtaining header and affine information")
brain_data = nib.load(np.load(BRAIN_HEADER_AND_AFFINE)[0])
affine = brain_data.affine
header = brain_data.header

log.INFO("Starting FDR correction...")
fdr_correction_and_viz(
    all_rois_p_val_4d_mat, all_rois_t_val_4d_mat, all_asd_subs_mean_fc_4d_mat,
    all_tdh_subs_mean_fc_4d_mat, BINARIZED_ATLAS_MP,
    OUTPUT_FILE_PATH + abide + FIRST_LEVEL_DATA, affine, header,
    fdr_correction_output)
log.INFO("Hip Hip Hurray! FDR correction done!!!")
