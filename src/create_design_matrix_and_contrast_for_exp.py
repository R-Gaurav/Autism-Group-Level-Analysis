#
# Autism Group Level Analysis
#
# Author: Ramashish Gaurav
#
# The file prepares the design matrix to be used by both the files:
# do_group_level_glm_analysis.py and do_statistical_analysis.py
# such that both files use the same design matrix, a mandatory requirement.
#
# Note: Change the values of `Design Matrix Entries` MACROS for variations in
#       the design matrices.
#

import numpy as np
import numpy.linalg as npl
import pandas as pd
import pickle

from utility.constants import (
    PHENO_DATA_PATH_1, PHENO_DATA_PATH_2, OUTPUT_FILE_PATH, ABIDE_1_SUBS_MD,
    ABIDE_2_SUBS_MD, ABIDE_1, ABIDE_2, FIRST_LEVEL_DATA)
from utility.dcp_utilities import (
    get_processed_design_matrix, get_asd_and_tdh_subs_row_indices)
from utility.exp_utilities import get_regressors_mask_list
from utility import log

# Design Matrix Entries
ASD = True
TDH = False
LEH = True
RIH = True
EYO = True
EYC = False
FIQ = True
AGE = True
IPT = True # Intercept

def get_design_matrix_for_the_exp(is_abide1):
  """
  Prepares and returns the design matrix for the experiment, and the regressors
  string vector.

  Args:
    is_abide1 (bool): `True` if exp is for ABIDE-1 else `False`.

  Returns:
    numpy.ndarray, str: The design matrix, Regressors string vector.
  """
  if is_abide1:
    abide = ABIDE_1
    pheno_data_path = PHENO_DATA_PATH_1
    all_subs_metadata = ABIDE_1_SUBS_MD
  else:
    abide = ABIDE_2
    pheno_data_path = PHENO_DATA_PATH_2
    all_subs_metadata = ABIDE_2_SUBS_MD

  df = pd.read_csv(pheno_data_path, encoding="ISO-8859-1") # Read the phenotype dataframe.
  valid_subs = pickle.load(
      open(OUTPUT_FILE_PATH + abide + FIRST_LEVEL_DATA + all_subs_metadata +
      "/valid_subs.p", "rb"), encoding="bytes")
  asd_row_ids, tdh_row_ids = get_asd_and_tdh_subs_row_indices(df, valid_subs)
  regressors_mask, regressors_strv = get_regressors_mask_list(
      asd=ASD, tdh=TDH, leh=LEH, rih=RIH, eyo=EYO, eyc=EYC, fiq=FIQ, age=AGE,
      ipt=IPT)

  if not IPT:
    regressors_strv = regressors_strv[:-1] # Get rid of the last "_".

  design_matrix = get_processed_design_matrix(
      df, asd_row_ids, tdh_row_ids, regressors_mask)

  log.INFO("Design Matrix structure and column order: ASD = %s, TDH = %s, "
           "LEH = %s, RIH = %s, EYO = %s, EYC = %s, FIQ = %s, AGE = %s, "
           "Intercept = %s" % (ASD, TDH, LEH, RIH, EYO, EYC, FIQ, AGE, IPT))
  log.INFO("Design matrix shape: {}".format(design_matrix.shape))
  log.INFO("Design matrix rank: %s" % npl.matrix_rank(design_matrix))

  # Get the condition number of regressor_matrix, if it is very high then the
  # solutions of the linear system of equations (GLM) is prone to large
  # numerical errors as the matrix is mostly singular i.e. non invertible.
  log.INFO("Design matrix condition number: %s"
          % npl.cond(design_matrix.T.dot(design_matrix)))

  # Assert that regressors_matrix is full rank i.e. columns are independent.
  assert npl.matrix_rank(design_matrix) == design_matrix.shape[1]

  return design_matrix, regressors_strv

def get_contrast_vector_for_exp():
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
  contrast_str = "_".join(str(c[0,0]) for c in contrast)

  return contrast, contrast_str
