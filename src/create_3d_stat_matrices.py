#
# Autism Group Level Analysis
#
# Author: Ramashish Gaurav
#
# This file creates 3D stat matrices, i.e. each and every cell of the 3D brain
# matrix contains either a p-value or t-value obtained from tp-tuple matrices
# after the execution of `do_statistical_analysis.py`.
#
# Usage: python2 create_3d_stat_matrices.py
#
# Note: It saves those 3D matrices in *.nii.gz format. Change the values of
# `is_abide1` and `is_t_val` to toggle between ABIDE1/ABIDE2 and t-val/p-val stat.
#

import datetime
import nibabel as nib
import numpy as np

from multiprocessing import Pool

from utility.constants import (ABIDE_1_BW_TP_TPL_DATA, ABIDE_2_BW_TP_TPL_DATA,
    ABIDE_1_ALL_TSTAT_DATA, ABIDE_2_ALL_TSTAT_DATA, ABIDE_1_ALL_PSTAT_DATA,
    ABIDE_2_ALL_PSTAT_DATA, OUTPUT_FILE_PATH, LOG_FILE_PATH, ABIDE_1, ABIDE_2,
    TOTAL_NUMBER_OF_ROIS, BATCH_SIZE, ROI_WITH_ZERO_FC_MAT, FIRST_LEVEL_DATA)
from utility.exp_utilities import (
    get_3d_stat_mat_to_nii_gz_format, get_interval_list)
from utility import log

from create_design_matrix_and_contrast_for_exp import (
    get_design_matrix_for_the_exp, get_contrast_vector_for_exp)

num_cores = 32

is_abide1 = False # If is_abide1 = True, get 3D stat mat for ABIDE1 else ABIDE2.
is_t_val = True # If is_t_val = True, generate t-value 3D stat mat else p-value.

if is_abide1:
  abide = ABIDE_1
  batch_wise_tp_tpl_data = ABIDE_1_BW_TP_TPL_DATA
  if is_t_val:
    all_rois_stat_data = ABIDE_1_ALL_TSTAT_DATA
  else:
    all_rois_stat_data = ABIDE_1_ALL_PSTAT_DATA
else:
  abide = ABIDE_2
  batch_wise_tp_tpl_data = ABIDE_2_BW_TP_TPL_DATA
  if is_t_val:
    all_rois_stat_data = ABIDE_2_ALL_TSTAT_DATA
  else:
    all_rois_stat_data = ABIDE_2_ALL_PSTAT_DATA

stat = "t" if is_t_val else "p"

log.configure_log_handler(
    "%s_%s.log" % (LOG_FILE_PATH + abide + FIRST_LEVEL_DATA + __file__,
    datetime.datetime.now()))

log.INFO("Creating and saving 3D %s stat matrices for %s" % (stat, abide))

################################################################################
################ Get the design matrix ###################
################################################################################

log.INFO("Obtaining %s design matrix" % abide)
design_matrix, regressors_strv = get_design_matrix_for_the_exp(is_abide1)
log.INFO("Regressors String Vec: %s" % regressors_strv)
################################################################################
################## Set the contrast vector ####################
################################################################################

contrast, contrast_str = get_contrast_vector_for_exp()
log.INFO("Contrast vector: %s" % contrast)
# Assert the size of contrast vector is equal to the number of cols in dsgn mat.
assert contrast.shape[0] == design_matrix.shape[1]
################################################################################
################## Start the parallel stat matrix generation ###################
################################################################################

# Create a pool of Processes.
pool = Pool(num_cores)

for batch_start in xrange(0, TOTAL_NUMBER_OF_ROIS, BATCH_SIZE):
  batch_end = min(batch_start + BATCH_SIZE, TOTAL_NUMBER_OF_ROIS)
  log.INFO("Reading the batch: [%s %s)" % (batch_start, batch_end))
  all_subs_batch_wise_rois_tp_tuple_matrix = np.load(
      OUTPUT_FILE_PATH + abide + FIRST_LEVEL_DATA + batch_wise_tp_tpl_data +
      "batch_%s_%s_rois_tp_stat_tuple_4D_matrix_dsgn_mat_%s_contrast_%s.npy"
      % (batch_start, batch_end, regressors_strv, contrast_str))

  rois, bx, by, bz = all_subs_batch_wise_rois_tp_tuple_matrix.shape

  # Do data parallelization to divide all_subs_batch_wise_rois_tp_tuple_matrix
  # to each processor. Division is on ROIs.
  rois_range_list = get_interval_list(rois, num_cores)
  log.INFO("ROIS range list: %s" % rois_range_list)

  data_input_list = [(
      all_subs_batch_wise_rois_tp_tuple_matrix[roi_range[0]:roi_range[1]].copy(),
      is_t_val) for roi_range in rois_range_list]
  del all_subs_batch_wise_rois_tp_tuple_matrix

  log.INFO("Starting %s stat matrix generation for batch %s %s"
           % (stat, batch_start, batch_end))
  batch_wise_rois_stat_3d_mat_obj_list = pool.map(
      get_3d_stat_mat_to_nii_gz_format, data_input_list)
  # batch_wise_rois_stat_3d_mat_obj_list is a list of lists (e.g. [[3D_stat_mat1,
  # 3D_stat_mat2, 3D_stat_mat3, ..], [3D_stat_mat1, 3D_stat_mat2, ..]]), such
  # that each element of the inner list is a Nifti1Image() of the 3D stat matrix
  # corresponding to an ROI. The outer list has number of elements (i.e. inner
  # lists equal to the number of processors passed in arguments.
  del data_input_list

  stat_3d_mat_list = [
      mat for stat_mat_list in batch_wise_rois_stat_3d_mat_obj_list for mat in
      stat_mat_list]

  log.INFO("3D %s stat matrices for the batch %s %s obtained, now saving mats.."
           % (stat, batch_start, batch_end))
  roi_slice = range(batch_start, batch_end)
  # Remove the all zero FC matrix at 254th ROI.
  if ROI_WITH_ZERO_FC_MAT >= batch_start and ROI_WITH_ZERO_FC_MAT < batch_end:
    roi_slice.remove(ROI_WITH_ZERO_FC_MAT)

  # Assert that the number ROIs in `roi_slice` is equal to number of 3D stat
  # matrices in `stat_3d_mat_list`. The order of ROIs matches the order of 3D
  # stat mats in `stat_3d_mat_list`.
  assert len(roi_slice) == len(stat_3d_mat_list)

  log.INFO("ROI_SLICE for which 3D %s stat matrices are saved in *.nii.gz : %s"
           % (stat, roi_slice))

  for roi, stat_mat_nifti_obj in zip(roi_slice, stat_3d_mat_list):
    nib.save(stat_mat_nifti_obj,
             OUTPUT_FILE_PATH + abide + FIRST_LEVEL_DATA + all_rois_stat_data +
             "/%s_ROI_%s_stat_mat_dsgn_mat_%s_contrast_%s.nii.gz"
             % (roi, stat, regressors_strv, contrast_str))
    log.INFO("%s ROI %s stat matrix saved!" % (roi, stat))

log.INFO("Hip Hip Hurray! All ROIs %s stat matrices are saved now!" % stat)
