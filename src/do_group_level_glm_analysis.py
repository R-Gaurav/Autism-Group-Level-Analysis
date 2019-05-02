#
# Autism Group Level Analysis
#
# Author: Ramashish Gaurav
#
# This file implements group level GLM to regress out various regressors to find
# the significance of autistic and healthy individuals. It saves the obtained
# 3D matrix of beta values (w.r.t. regressors) and residual/error for each ROI.
#
# do_group_level_glm(...) is the main function to be called with appropriate
# arguments.
#
# Usage: pythno2 do_group_level_glm_analysis.py
#
# Make sure that the design_matrix used in file do_group_level_glm_analysis.py
# and in do_statistical_analysis.py are same, accordingly change the value of
# `is_abide1` to toggle between ABIDE-1 and ABIDE-2 dataset.
#
# Also, the BATCH_SIZE should remain same across create_fc_matrices_for_exp.py,
# do_group_level_glm_analysis.py, and do_statistical_analysis.py .
#

import datetime
import numpy as np
from multiprocessing import Pool

from utility.gla_utilities import do_group_level_glm
from utility.exp_utilities import get_interval_list
from utility.constants import (
    TOTAL_NUMBER_OF_ROIS, OUTPUT_FILE_PATH, BATCH_SIZE, ABIDE_1, ABIDE_2,
    LOG_FILE_PATH, ABIDE_1_BW_BE_TPL_DATA, ABIDE_2_BW_BE_TPL_DATA,
    ABIDE_1_BW_ROI_FC_DATA, ABIDE_2_BW_ROI_FC_DATA)
from utility import log

from create_design_matrix_for_exp import get_design_matrix_for_the_exp

num_cores = 32

log.configure_log_handler(
    "%s_%s.log" % (LOG_FILE_PATH+__file__, datetime.datetime.now()))

is_abide1 = True # If is_abide1 = True, do group level GLM for ABIDE1 else ABIDE2.

if is_abide1:
  abide = ABIDE_1
  batch_wise_rois_fc_data = ABIDE_1_BW_ROI_FC_DATA
  batch_wise_be_tpl_data = ABIDE_1_BW_BE_TPL_DATA
else:
  abide = ABIDE_2
  batch_wise_rois_fc_data = ABIDE_2_BW_ROI_FC_DATA
  batch_wise_be_tpl_data = ABIDE_2_BW_BE_TPL_DATA

log.INFO("Starting Group Level GLM for %s" % abide)

################################################################################
################ Get the design matrix ###################
################################################################################

log.INFO("Obtaining %s design matrix" % abide)
design_matrix, regressors_strv = get_design_matrix_for_the_exp(is_abide1)
log.INFO("Regressors String Vec: %s" % regressors_strv)
################################################################################
################# Start the parallel GLM analysis ###################
################################################################################
log.INFO("Starting Beta and Error calculation for %s" % abide)
# Create a pool of processes.
pool = Pool(num_cores)

for batch_start in xrange(0, TOTAL_NUMBER_OF_ROIS, BATCH_SIZE):
  batch_end = min(batch_start + BATCH_SIZE, TOTAL_NUMBER_OF_ROIS)
  log.INFO("Reading the batch: %s %s" % (batch_start, batch_end))
  all_subs_batch_wise_rois_fc_matrix = np.load(
      OUTPUT_FILE_PATH + batch_wise_rois_fc_data +
      "/all_subs_%s_start_%s_end_ROIs_fc_5D_matrix.npy"
      % (batch_start, batch_end))

  num_subs_fc_matrix, rois, bx, by, bz = all_subs_batch_wise_rois_fc_matrix.shape

  # Do data parallelization by dividing all_subs_batch_wise_rois_fc_matrix
  # equally into bins equal to num_cores. Division is on ROIs.
  rois_range_list = get_interval_list(rois, num_cores)
  log.INFO("ROIs range list: %s" % rois_range_list)

  data_input_list = [(
      all_subs_batch_wise_rois_fc_matrix[:, roi_range[0]:roi_range[1]].copy(),
      design_matrix) for roi_range in rois_range_list]
  del all_subs_batch_wise_rois_fc_matrix

  log.INFO(
      "Starting Group Level GLM for batch: %s %s" % (batch_start, batch_end))
  batch_wise_rois_be_mat_list = pool.map(do_group_level_glm, data_input_list)
  # batch_wise_rois_be_mat_list is a list of np.ndarrays (e.g. [np.ndarray(
  # [3D_mat1, 3D_mat2, ...]), ..., np.ndarray([3D_mat1, 3D_mat2, ...])]), such
  # that each element np array is an array of 3D matrices. The element arrays
  # have length equal to the number of ROIs they intook. Each ROI corresponds
  # to one 3D matrix. The extenal list has length equal to the length of the
  # data_input_list/rois_range_list.
  del data_input_list

  log.INFO("Group Level GLM results for the batch %s %s obtained"
           % (batch_start, batch_end))
  result = []
  for rois_be_mat_arr in batch_wise_rois_be_mat_list:
    result.extend(rois_be_mat_arr) # Extend each array having 3D beta error mats.
  result = np.array(result)
  assert result.shape == (rois, bx, by, bz)

  log.INFO("Saving the Beta Error tuple matrix obtained from group level GLM "
           "for batch %s %s ..." % (batch_start, batch_end))
  np.save(
      OUTPUT_FILE_PATH + batch_wise_be_tpl_data +
      "batch_%s_%s_rois_beta_error_tuple_4D_matrix_dsgn_mat_%s"
      % (batch_start, batch_end, regressors_strv), result)
  log.INFO("Beta Error tuple matrix obtained from group level GLM saved for "
           "batch %s %s" % (batch_start, batch_end))

log.INFO("Hip Hip Hurray! Group Level GLM Done!!!")
