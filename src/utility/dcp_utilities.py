#
# Author: Ramashish Gaurav
#
# This file contains utilities for cleaning and preprocessing the data files.
#

from scipy import stats

import numpy as np
import pandas as pd

def get_row_indices_with_no_missing_vals_list(df, clm_name_list):
  """
  Returns a list of indices of those rows which do not have any missing values
  in any of the columns passed in `clm_name_list`.

  Args:
    df (pandas.DataFrame): A dataframe.
    clm_name_list ([]): A list of column names.

  Returns
    []: A list of row's absolute indices.
  """
  for clm_name in clm_name_list:
    # Get rows absolute indices with no missing vals in column `clm_name`.
    rows_absolute_index = df[clm_name].dropna().index
    # Get the data frame using .loc[] to fetch rows with absolute indices. On the
    # other hand, .iloc[] uses relative idexing.
    df = df.loc[rows_absolute_index]
  return df.index.tolist()

def get_row_indices_without_discarded_value_list(df, clm_name_dv_tuple_list):
  """
  Returns a list of indices of those rows which do not have the `discard_value`
  in the specified column name `clm_name`.

  Args:
    df (pandas.DataFrame): A dataframe.
    clm_name_dv_tuple_list []: A list of tuples where each tuple has a column
        name and corresponding value to be discarded from the that column.

  Returns:
    []: A list of row's absolute indices.
  """
  for clm_name, dval in clm_name_dv_tuple_list:
    rows_absolute_index = df.loc[df[clm_name] != dval].index
    df = df.loc[rows_absolute_index]
  return df.index.tolist()

def get_phenotypes_regressors_matrix(df, asd_row_ids, tdh_row_ids,
                                     clm_name_list):
  """
  Creates and returns a regressor matrix.

  Args:
    df (pandas.DataFrame): A data frame of subject phenotypes.
    asd_row_ids ([int]): Autism Spectrum Disorder subject's row indices in `df`.
    tdh_row_ids ([int]): Typically Developing Healthy subject's row indices in
        `df`.
    clm_name_list ([str]): A list of phenotype column names for creating columns
        in regressors matrix. Note: The clm_name_list should not have "DX_GROUP",
        "HANDEDNESS_CATEGORY" and "EYE_STATUS_AT_SCAN" column names, as these
        columns are accounted separately.

  Returns:
    numpy.ndarray: A regressor matrix (design matrix).

  Note:
    The order of columns in regressors_matrix is: "ASD", "TDH", "LEFT_H",
    "RIGHT_H", "EYE_OPEN", "EYE_CLOSED", "FIQ", "AGE_AT_SCAN".
  """
  num_subs = len(asd_row_ids)+len(tdh_row_ids)
  all_subs_indices = []
  all_subs_indices.extend(asd_row_ids)
  all_subs_indices.extend(tdh_row_ids)
  reg_col_list = []

  #################### PREPARE REGRESSOR MATRIX COLUMNS ###################
  # First set of subjects are suffering from ASD, next set are TDH subjects.
  asd_clm = [1 if _ < len(asd_row_ids) else 0 for _ in range(num_subs)]
  tdh_clm = [0 if _ < len(asd_row_ids) else 1 for _ in range(num_subs)]

  # Create two columns for "HANDEDNESS_CATEGORY" and "EYE_STATUS_AT_SCAN".
  handedness = df.loc[all_subs_indices, "HANDEDNESS_CATEGORY"].tolist()
  eye_status = df.loc[all_subs_indices, "EYE_STATUS_AT_SCAN"].tolist()

  # ABIDE I has "L", "R", "Ambi", "L->R" and "Mixed" notation. ABIDE II has 1,
  # 2 and 3 notation for right hand, left hand and ambidextrous respectively.
  left_hand = [1 if (_ == "L" or _ == "Ambi" or _ == "L->R" or _ == "Mixed" or
               _ == 2 or _ == 3) else 0 for _ in handedness]
  right_hand = [1 if (_ == "R" or _ == "Ambi" or _ == "L->R" or _ == "Mixed" or
                _ == 1 or _ == 3) else 0 for _ in handedness]

  # In ABIDE I and ABIDE II, 1 stands for eyes opened, 2 stands for eyes closed.
  eye_open = [1 if _ == 1 else 0 for _ in eye_status]
  eye_closed = [1 if _ == 2 else 0 for _ in eye_status]
  ######################## CREATE REGRESSOR MATRIX #######################
  reg_col_list.append(asd_clm)
  reg_col_list.append(tdh_clm)
  reg_col_list.append(left_hand)
  reg_col_list.append(right_hand)
  reg_col_list.append(eye_open)
  reg_col_list.append(eye_closed)

  # Append rest of the column values. ["FIQ", "AGE_AT_SCAN"]
  for clm_name in clm_name_list:
    reg_col_list.append(df.loc[all_subs_indices, clm_name].tolist())
  ######################### PREPARATION DONE #############################

  regressors_matrix = np.matrix(reg_col_list).T
  # Assert that the number of rows in regressors_matrix is equal to total number
  # of subjects.
  assert regressors_matrix.shape[0] == num_subs
  return regressors_matrix

def get_no_missing_vals_df(df):
  """
  Returns a dataframe with no missing values in the columns specified in this
  function.

  Args:
    df (pandas.DataFrame):  A phenotypic dataframe.

  Returns:
    pandas.DataFrame
  """
  dc_clm_names = [
      "DX_GROUP", "FIQ", "AGE_AT_SCAN", "HANDEDNESS_CATEGORY",
      "EYE_STATUS_AT_SCAN"]
  rows_indices_no_na_list = get_row_indices_with_no_missing_vals_list(
      df, dc_clm_names)
  no_na_df = df.loc[rows_indices_no_na_list]
  return no_na_df

def get_valid_subject_ids_with_complete_phenotype(pheno_data_path, is_abide1):
  """
  Returns lists of subject ids who have no missing values in their phenotype
  data and have certain values discarded from their phenotypes.

  Args:
    pheno_data_path (str): /path/to/phenotype/data.csv
    is_abide1 (bool); Does the phenotype csv belong to ABIDE1?

  Returns:
    list, list: List of ASD subjects, List of TDH subjects (SUB_IDs list).
  """
  pheno_df = pd.read_csv(pheno_data_path, encoding="ISO-8859-1")
  # Remove the NA values from phenotype_df with respect to the considered
  # regressor columns (Note: Column names are mentioned are in
  # get_no_missing_vals_df func).
  pheno_no_na_df = get_no_missing_vals_df(pheno_df)

  # Discard the "-9999" values in "HANDEDNESS_CATEGORY", -9999 values in "FIQ"
  # column from pheno_no_na_df. Note that "HANDEDNESS_CATEGORY" in ABIDE II is a
  # float, and is a string in ABIDE I. Therefore accordingly change the type of
  # the value to be discarded. Also discard the 0 values in "EYE_STATUS_AT_SCAN".

  if is_abide1:
    hdness_disc_val = "-9999"
  else:
    hdness_disc_val = -9999
  row_indices_no_discarded_list = get_row_indices_without_discarded_value_list(
      pheno_no_na_df, [("HANDEDNESS_CATEGORY", hdness_disc_val), ("FIQ", -9999),
      ("EYE_STATUS_AT_SCAN", 0)])
  pheno_no_na_no_dval_df = pheno_no_na_df.loc[row_indices_no_discarded_list]

  # The selected phenotype dataframe does not have any NA or empty values. It
  # also does not have any discarded values. Get the ASD subjects list and TDH
  # subjects list.
  # DX_GROUP = 1 => ASD and DX_GROUP = 2 => TDH. (in ABIDE1 and ABIDE2)
  asd_subjects_list = pheno_no_na_no_dval_df.loc[
      pheno_no_na_no_dval_df["DX_GROUP"] == 1]["SUB_ID"].tolist()
  tdh_subjects_list = pheno_no_na_no_dval_df.loc[
      pheno_no_na_no_dval_df["DX_GROUP"] == 2]["SUB_ID"].tolist()

  return asd_subjects_list, tdh_subjects_list

def get_asd_and_tdh_subs_row_indices(df, sub_id_list):
  """
  Returns two individual lists of row indices for ASD and TDH subjects.
  If "DX_GROUP" is 1 => Autism/ASD.
  If "DX_GROUP" is 2 => Control/TDH.

  Args:
    df(pandas.DataFrame): DataFrame of subject phenotypes.
    sub_id_list([]): A list of SUB_ID.

  Returns:
    [int], [int]: List of ASD subs row indices, List of TDH subs row indices.
  """
  df = df[df["SUB_ID"].isin(sub_id_list)]
  asd_row_ids = df[df["DX_GROUP"] == 1].index.tolist()
  tdh_row_ids = df[df["DX_GROUP"] == 2].index.tolist()
  return asd_row_ids, tdh_row_ids

def get_processed_design_matrix(pheno_df, asd_row_ids, tdh_row_ids,
                                regressors_mask,
                                clm_name_list=["FIQ", "AGE_AT_SCAN"]):
  """
  Returns a design matrix with only those regressors which are mentioned in
  the regressors_mask.

  Args:
    pheno_df (pandas.DataFrame): The pehnotype dataframe of subjects.
    asd_row_ids ([int]): ASD subjects' row indices in the pheno_df.
    tdh_row_ids ([int]): TDH subjects' row indices in the pehno_df.
    clm_name_list ([str]): List of column names of pheno_df to be included in
        the design matrix. Note: The clm_name_list should not have "DX_GROUP",
        "HANDEDNESS_CATEGORY" and "EYE_STATUS_AT_SCAN" column names, as these
        columns are accounted separately in get_phenotypes_regressors_matrix().

  Returns:
    numpy.ndarray: The processed design matrix.
  """
  pheno_regressors_mat = get_phenotypes_regressors_matrix(
      pheno_df, asd_row_ids, tdh_row_ids, clm_name_list)

  # The sequence of columns in pheno_regressors_mat returned by
  # get_phenotypes_regressors_matrix() is in the following order:
  # ["ASD", "TDH", "LEFT_H", "RIGHT_H", "EYE_OPEN", "EYE_CLOSED", "FIQ",
  # "AGE_AT_SCAN"].
  # Standardize FIQ and AGE_AT_SCAN
  pheno_regressors_mat[:, 6] = stats.zscore(pheno_regressors_mat[:, 6])
  pheno_regressors_mat[:, 7] = stats.zscore(pheno_regressors_mat[:, 7])

  # Since the last element of regressors_mask always denotes the intercept, get
  # rid of it in below calculations since pheno_regressors_mat obtained from the
  # phenotypes CSV data does not have the intercept.
  assert len(regressors_mask[:-1]) == pheno_regressors_mat.shape[1]
  clms_indices = [
      i for i in xrange(len(regressors_mask[:-1])) if regressors_mask[i]]
  ret_dsgn_mat = pheno_regressors_mat[:, clms_indices]

  # Last element of regressors_mask indicates whether or not to include the
  # intercept of 1s as the last column of the design matrix.
  if regressors_mask[-1]:
    ret_dsgn_mat = np.c_[ret_dsgn_mat, np.ones(pheno_regressors_mat.shape[0])]

  return ret_dsgn_mat
