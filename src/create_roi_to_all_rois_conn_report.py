#
# Autism Group Level Analysis
#
# Author: Ramashish Gaurav
#
# This file reads all the *.nii.gz files in the passed directory and constructs
# a CSV report file which stores the connectivity information an ROI to all
# other ROIs.
#
# To run: python2 create_roi_to_all_rois_conn_report.py -a <opt> -t <opt> -v <opt>
#
# Note: It uses Varun's `Cluster-Reporting-Tool` to create the report.
# Github: https://github.com/varun-invent/Cluster-Reporting-Tool
#

import argparse
import datetime
import nibabel as nib
import os
import sys
sys.path.insert(0, "/home/others/ramashish/Cluster-Reporting-Tool/")

from multiprocessing import Pool

from utility.constants import (OUTPUT_FILE_PATH, ABIDE_1_ALL_ROIS_CSV_REP,
    ABIDE_2_ALL_ROIS_CSV_REP, LOG_FILE_PATH, ABIDE_1, ABIDE_2,
    TOTAL_NUMBER_OF_ROIS, BATCH_SIZE, ROI_WITH_ZERO_FC_MAT, FIRST_LEVEL_DATA,
    ABIDE_1_FDR_CRCTN_OUTPUT, ABIDE_2_FDR_CRCTN_OUTPUT,
    ABIDE_1_ALL_ROIS_LOGQ_MATS, ABIDE_2_ALL_ROIS_LOGQ_MATS)
from utility.exp_utilities import (
    get_interval_list, create_3d_brain_nii_from_4d_mat)
from utility import log

from cluster_reporting_tool import cluster_reporting_tool
from create_design_matrix_and_contrast_for_exp import (
    get_design_matrix_for_the_exp, get_contrast_vector_for_exp)

num_cores = 1

is_abide1 = False # If is_abide1 = True, get 3D stat mat for ABIDE1 else ABIDE2.

if is_abide1:
  abide = ABIDE_1
  all_rois_csv_report = ABIDE_1_ALL_ROIS_CSV_REP
  fdr_correction_output = "/" + ABIDE_1_FDR_CRCTN_OUTPUT + "/"
  all_rois_logq_output = ABIDE_1_ALL_ROIS_LOGQ_MATS
else:
  abide = ABIDE_2
  all_rois_csv_report = ABIDE_2_ALL_ROIS_CSV_REP
  fdr_correction_output = "/" + ABIDE_2_FDR_CRCTN_OUTPUT + "/"
  all_rois_logq_output = ABIDE_2_ALL_ROIS_LOGQ_MATS

stat = "logq"

log.configure_log_handler(
    "%s_%s.log" % (LOG_FILE_PATH + abide + FIRST_LEVEL_DATA + __file__,
    datetime.datetime.now()))

def wrap_cluster_reporting_tool(params):
  """
  A wrapper over cluster_reporting_tool to parallelize it.

  Args:
    params: A 4 element tuple.
      roi_slice, atlas_dict, threshold, volume = (
          params[0], params[1], params[2], params[3])

    roi_slice ([int]): A list of ROIs.
    atlas_dict (dict): A dict of brain atlases.
    threshold (float): Threshold for the stat to be filtered.
    volume (int): Volumne number of the brain to be accounted for.
  """
  print("RG here1")
  roi_slice, atlas_dict, threshold, volume = (
      params[0], params[1], params[2], params[3])
  print("RG here2")
  for roi in roi_slice:
    print("RG here3 %s" % roi)
    stat_mat = (OUTPUT_FILE_PATH + abide + FIRST_LEVEL_DATA + all_rois_logq_output
                + "%s_ROI_%s_stat_dsgn_%s_contrast_%s_3d_brain_file.nii.gz"
                % (roi, stat, regressors_strv, contrast_str))
    print("RG here4 %s " % roi)
    crl_object = cluster_reporting_tool(stat_mat, atlas_dict, threshold, volume)
    print("RG here5 %s" % roi)
    crl_object.report(
        out_file=OUTPUT_FILE_PATH + abide + FIRST_LEVEL_DATA + all_rois_csv_report
        + "%s_ROI_%s_stat_dsgn_mat_%s_contrast_%s_report.csv"
        % (roi, stat, regressors_strv, contrast_str))
    print("RG here6 %s" % roi)
    log.INFO("%s stat CSV report saved for %s ROI" % (stat, roi))
    print("RG here7 %s" % roi)

log.INFO("Creating the %s stat report of ROI to all ROIs connectivity for %s"
         % (stat, abide))

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
################## Start the parallel CSV report generation ####################
################################################################################

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("-a", "--atlas", required=False,
                  help="Path to Atlas file")
  ap.add_argument("-t", "--thresh", required=False,
                  help="Threshold")
  ap.add_argument("-v", "--vol", required=False,
                  help="Volume Number (If a 4D contrast is used as input) \
                  [Starts from 0]")

  args = vars(ap.parse_args())

  base_path = os.path.abspath('../../Cluster-Reporting-Tool') + '/'

  if args["atlas"] != None:
      atlas = args["atlas"]
  else:
      atlas = 'AAL'
  log.INFO("Using atlas %s" % atlas)

  if args["thresh"] != None:
      threshold = float(args["thresh"])
  else:
      threshold = 0.2
  log.INFO("Using threshold of %s" % threshold)

  if args["vol"] != None:
      volume = int(args["vol"])
  else:
      volume = 0
  log.INFO("Using Volume_index %s" % str(volume))

  if atlas == 'AAL':
      atlas_path = [base_path + 'aalAtlas/AAL.nii.gz']
      atlas_labels_path = [base_path + 'aalAtlas/AAL.xml']
      atlas_xml_zero_start_index  =  False
  elif atlas == 'fb':
      atlas_path = [base_path +
      'Full_brain_atlas_thr0-2mm/fullbrain_atlas_thr0-2mm_resample.nii']
      atlas_labels_path = [base_path +
      'Full_brain_atlas_thr0-2mm/fullbrain_atlas.xml']
      atlas_xml_zero_start_index  =  True

  atlas_dict = {
  'atlas_path': atlas_path,
  'atlas_labels_path': atlas_labels_path,
  'atlas_xml_zero_start_index': atlas_xml_zero_start_index
  }
  log.INFO("Using atlas dict: %s" % atlas_dict)

  # Create a pool of Processes.
  pool = Pool(num_cores)

  # Load the map_logq.nii.gz, a 4D bx x by x bz x TOTAL_NUMBER_OF_ROIS matrix.
  # Clusters have to be found for each ROI's logq values (a 3D mat of brain).
  log.INFO("Obtaining the map_%s.nii.gz (a 4D file: bx, by, bz, #ROIs)" % stat)
  stat_4d_mat = nib.load(
      OUTPUT_FILE_PATH + abide + FIRST_LEVEL_DATA + fdr_correction_output +
      "map_%s.nii.gz" % stat).get_fdata()

  log.INFO("Creating and saving the individual ROI's %s brain file..." % stat)
  create_3d_brain_nii_from_4d_mat(
      stat_4d_mat, regressors_strv, contrast_str, stat, is_abide1)
  log.INFO("All ROI's %s brain files saved!" % stat)

  for batch_start in xrange(0, TOTAL_NUMBER_OF_ROIS, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, TOTAL_NUMBER_OF_ROIS)
    log.INFO(
        "Creating CSV reports for batch: [%s %s)" % (batch_start, batch_end))

    # Do data parallelization to divide the ROI batch to each processor.
    rois_slice = range(batch_start, batch_end)
    if ROI_WITH_ZERO_FC_MAT >= batch_start and ROI_WITH_ZERO_FC_MAT < batch_end:
      rois_slice.remove(ROI_WITH_ZERO_FC_MAT)
    rois_range_list = get_interval_list(len(rois_slice), num_cores)
    log.INFO("ROIS range list: %s" % rois_range_list)
    log.INFO("ROIS slice: %s" % rois_slice)

    data_input_list = [
        (rois_slice[roi_range[0]:roi_range[1]], atlas_dict, threshold, volume)
        for roi_range in rois_range_list]
    #pool.map(wrap_cluster_reporting_tool, data_input_list)
    wrap_cluster_reporting_tool(data_input_list[0])
    print("RG here8")
    log.INFO("CSV reports for batch: [%s %s) done!" % (batch_start, batch_end))

  log.INFO("Hip Hip Hurray! %s stat CSV reports have been created for all the "
           "ROIs." % stat)
