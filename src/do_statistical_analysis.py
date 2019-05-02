#
# Autism Group Level Analysis
#
# Author: Ramashish Gaurav
#
# This file finds the t value and p value for each brain voxel for each ROI. A
# voxel's t value and p value correspond across all subjects (autistic +
# healthy).
#
# Note: It does a two tailed t test.
#
# Usage: python2 do_statistical_analysis.py
#
# Make sure that the design_matrix used in file do_group_level_glm_analysis.py
# and in do_statistical_analysis.py are same.
# Also, the BATCH_SIZE should remain same across create_fc_matrices_for_exp.py,
# do_group_level_glm_analysis.py, and do_statistical_analysis.py .
#

import datetime
import numpy as np

from multiprocessing import Pool

from utility.gla_utilities import do_statistical_analysis
from utility.exp_utilities import get_interval_list
from utility.constants import (
    TOTAL_NUMBER_OF_ROIS, OUTPUT_FILE_PATH, LOG_FILE_PATH, BATCH_SIZE,
    ABIDE_1, ABIDE_2, ABIDE_1_BW_BE_TPL_DATA, ABIDE_2_BW_BE_TPL_DATA,
    ABIDE_1_BW_TP_TPL_DATA, ABIDE_2_BW_TP_TPL_DATA)
from utility import log

from create_design_matrix_for_exp import get_design_matrix_for_the_exp

num_cores = 32

log.configure_log_handler(
    "%s_%s.log" % (LOG_FILE_PATH+__file__, datetime.datetime.now()))

is_abide1 = True # If is_abide1 = True, do group level GLM for ABIDE1 else ABIDE2.

if is_abide1:
  abide = ABIDE_1
  batch_wise_be_tpl_data = ABIDE_1_BW_BE_TPL_DATA
  batch_wise_tp_tpl_data = ABIDE_1_BW_TP_TPL_DATA
else:
  abide = ABIDE_2
  batch_wise_be_tpl_data = ABIDE_2_BW_BE_TPL_DATA
  batch_wise_tp_tpl_data = ABIDE_2_BW_TP_TPL_DATA

log.INFO("Starting statistical analysis for %s" % abide)

################################################################################
################ Get the design matrix ###################
################################################################################

log.INFO("Obtaining %s design matrix" % abide)
design_matrix, regressors_strv = get_design_matrix_for_the_exp(is_abide1)
log.INFO("Regressors String Vec: %s" % regressors_strv)
################################################################################
################## Set the contrast vector ####################
################################################################################
# The first entry in the design_matrix with intercept could be either ASD or
# TDH column, with the intercept being the last column.
contrast = np.matrix([
    1, # = ASD - TDH (Since in the design matrix, the first column is ASD and
    0, # last column is that of the intercept).
    0,
    0,
    0,
    0,
    0
  ]).T
log.INFO("Contrast vector: %s" % contrast)
# Assert the size of contrast vector is equal to the number of cols in dsgn mat.
assert contrast.shape[0] == design_matrix.shape[1]
################################################################################
################# Start the parallel statistical analysis process ##############
################################################################################
log.INFO("Starting T and P value calculation for %s" % abide)
# Create a pool of processes.
pool = Pool(num_cores)

for batch_start in xrange(0, TOTAL_NUMBER_OF_ROIS, BATCH_SIZE):
  batch_end = min(batch_start + BATCH_SIZE, TOTAL_NUMBER_OF_ROIS)
  log.INFO("Reading the batch: %s %s" % (batch_start, batch_end))
  all_subs_batch_wise_rois_be_tuple_matrix = np.load(
      OUTPUT_FILE_PATH + batch_wise_be_tpl_data +
      "batch_%s_%s_rois_beta_error_tuple_4D_matrix_dsgn_mat_%s"
      % (batch_start, batch_end, regressors_strv))

  rois, bx, by, bz = all_subs_batch_wise_rois_be_tuple_matrix.shape

  # Do data parallelization to divide all_subs_batch_wise_rois_be_tuple_matrix
  # to each processor. Division is on ROIs.
  rois_range_list = get_interval_list(rois, num_cores)
  log.INFO("ROIs range list: %s" % rois_range_list)

  data_input_list = [(
      all_subs_batch_wise_rois_be_tuple_matrix[roi_range[0]:roi_range[1]].copy(),
      contrast, design_matrix) for roi_range in rois_range_list]
  del all_subs_batch_wise_rois_be_tuple_matrix

  log.INFO("Starting statistical analysis for batch: %s %s"
           % (batch_start, batch_end))

  batch_wise_rois_tp_mat_list = pool.map(do_statistical_analysis, data_input_list)
  # batch_wise_rois_tp_mat_list is a list of lists (e.g. [[3D_mat1, 3D_mat2, ...]
  # , ..., [3D_mat1, 3D_mat2, ...]]), such that each element list is a list of 3D
  # matrices. The element lists have length equal to the number of ROIs they
  # intook. Each ROI corresponds to one 3D matrix. The external list has length
  # equal to the number of processors passed in arguments.
  del data_input_list

  log.INFO("Statistical results for the batch %s %s obtained"
           % (batch_start, batch_end))
  result = []
  for rois_tp_stat_mat_list in batch_wise_rois_tp_mat_list:
    result.extend(rois_tp_stat_mat_list)
  result = np.array(result)
  assert result.shape == (rois, bx, by, bz)

  log.INFO("Saving the t-p value stat matrix obtained from statistical analysis "
           "for batch %s %s ..." % (batch_start, batch_end))
  np.save(OUTPUT_FILE_PATH + batch_wise_tp_tpl_data +
          "batch_%s_%s_rois_tp_stat_tuple_4D_matrix_dsgn_mat_%s_contrast_%s"
          % (batch_start, batch_end, regressors_strv,
          "_".join(str(c[0,0]) for c in contrast)), result)
  log.INFO("T and P value stat matrix obtained from statistical analysis saved "
           "for batch: %s %s" % (batch_start, batch_end))

log.INFO("Hip Hip Hurray! Statistical Analysis Done!")
