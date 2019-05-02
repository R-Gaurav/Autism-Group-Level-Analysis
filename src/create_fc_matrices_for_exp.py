"""
This file creates the 5D FC matrices on which Group Level GLM is supposed to run.

To execute this file: python2 create_FC_matrice_for_exp.py

Note: Change the value of `is_abide1` variable to toggle between ABIDE1 & ABIDE2.

Make sure that the design_matrix used in file do_group_level_glm_analysis.py
and in do_statistical_analysis.py are same.
Also, the BATCH_SIZE should remain same across create_fc_matrices_for_exp.py,
do_group_level_glm_analysis.py, and do_statistical_analysis.py .
"""

import pickle
import pandas as pd
import time
import datetime

from utility.dcp_utilities import (get_valid_subject_ids_with_complete_phenotype,
                                   get_asd_and_tdh_subs_row_indices)
from utility.exp_utilities import (
    get_valid_subject_ids_wrt_fc_data, construct_subs_batchwise_rois_fc_5D_mats)
from utility.constants import (
    ROI_WITH_ZERO_FC_MAT, TOTAL_NUMBER_OF_ROIS, OUTPUT_FILE_PATH, LOG_FILE_PATH,
    PHENO_DATA_PATH_1, PHENO_DATA_PATH_2, FC_FILES_PATH_1, FC_FILES_PATH_2,
    BATCH_SIZE, ABIDE_1, ABIDE_2, ABIDE_1_SUBS_MD, ABIDE_2_SUBS_MD)
from utility import log

log.configure_log_handler(
    "%s_%s.log" % (LOG_FILE_PATH+__file__, datetime.datetime.now()))

is_abide1 = True # If abide1 is True, it prepares FC files for ABIDE1 else ABIDE2.

if is_abide1:
  pheno_data_path = PHENO_DATA_PATH_1
  fc_files_path = FC_FILES_PATH_1
  abide = ABIDE_1
  buggy_subs_file = "abide_1_bug_sub_ids_from_varun.p"
  all_subs_metadata = ABIDE_1_SUBS_MD
else:
  pheno_data_path = PHENO_DATA_PATH_2
  fc_files_path = FC_FILES_PATH_2
  abide = ABIDE_2
  buggy_subs_file = "abide_2_bug_sub_ids_from_varun.p"
  all_subs_metadata = ABIDE_2_SUBS_MD

log.INFO("FC files creation started for %s dataset" % abide)

################################################################################
############ Get the row IDs of all valid subjects ############
################################################################################

asd_subs, tdh_subs = get_valid_subject_ids_with_complete_phenotype(
    pheno_data_path, is_abide1)
all_subs = asd_subs + tdh_subs

log.INFO("Number of subjects in %s dataset with complete phenotypic data: %s"
         % (abide, len(all_subs)))

buggy_subs = pickle.load(
    open("/mnt/project1/home1/varunk/ramashish/data/buggy_subjects/%s"
    % buggy_subs_file, "rb"))

log.INFO("Removing buggy subs in file: %s" % buggy_subs_file)
all_subs = list(set(all_subs) - set(buggy_subs))

log.INFO("After removal of buggy subs, number of subs: %s" % len(all_subs))

log.INFO("Now getting the valid subs who have FC data available...")
valid_subs, no_fc_subs, zero_fc_subs, incmp_fc_subs = (
    get_valid_subject_ids_wrt_fc_data(fc_files_path, all_subs, is_abide1))

log.INFO("Number of valid subs in %s dataset for whom FC matrices would be "
         "created: %s, number of subs with NO FC data: %s, number of subs with "
         "zero FC data for ROIs other than %s ROI: %s, number of subs who do "
         "not have all the FC data: %s" % (abide, len(valid_subs),
         len(no_fc_subs), ROI_WITH_ZERO_FC_MAT, len(zero_fc_subs),
         len(incmp_fc_subs)))

log.INFO("SUB IDs of valid subs for whom 5D mats are created: %s" % valid_subs)
pickle.dump(valid_subs, open(OUTPUT_FILE_PATH+all_subs_metadata+"/valid_subs.p",
            "wb"))
log.INFO("SUB IDs of subjects with no FC data: %s" % no_fc_subs)
pickle.dump(no_fc_subs, open(OUTPUT_FILE_PATH+all_subs_metadata+
            "/subs_with_no_fc_data.p", "wb"))
log.INFO("SUB IDs of subjects with zero FC data for ROIs other than %s ROI: %s"
         % (ROI_WITH_ZERO_FC_MAT, zero_fc_subs))
pickle.dump(zero_fc_subs, open(OUTPUT_FILE_PATH+all_subs_metadata+
            "/subs_having_zero_values_fc_data.p", "wb"))
log.INFO("SUB IDs of subjects who do not all the %s FC matrices: %s"
         % (TOTAL_NUMBER_OF_ROIS, incmp_fc_subs))
pickle.dump(incmp_fc_subs, open(OUTPUT_FILE_PATH+all_subs_metadata+
            "/subs_not_having_all_rois_fc_data.p", "wb"))

df = pd.read_csv(pheno_data_path)
asd_row_ids, tdh_row_ids = get_asd_and_tdh_subs_row_indices(df, valid_subs)

log.INFO("Row IDs of ASD and TDH subs obtained!")

################################################################################
############# Creation of batchwise subs ROIs FC matrices #############
################################################################################

start = time.time()
construct_subs_batchwise_rois_fc_5D_mats(
    fc_files_path, df, asd_row_ids, tdh_row_ids, BATCH_SIZE, is_abide1)
log.INFO("Hurray! FC matrices for %s dataset created! Time taken: %s" % (
         abide, time.time() - start))

################################################################################
################################################################################
