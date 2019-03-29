#
# Author: Ramashish Gaurav
#
# This file contains utilities for cleaning and preprocessing the data files.
#

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

def get_regressors_matrix(df, asd_subs_list, tdh_subs_list, clm_name_list):
  """
  Creates and returns a regressor matrix.

  Args:
    df (pandas.DataFrame): A data frame of subject phenotypes.
    asd_subs_list (int): Autism Spectrum Disorder subject's row indices in `df`.
    tdh_subs_list (int): Typically Developing Healthy subject's row indices in
        `df`.
    clm_name_list ([]): A list of phenotype column names for creating columns in
        regressors matrix. Note: The clm_name_list should not have "DX_GROUP",
        "HANDEDNESS_CATEGORY" and "EYE_STATUS_AT_SCAN" column names, as these
        columns are accounted separately.

  Returns:
    numpy.ndarray: A regressor matrix (design matrix).
  """
  num_subs = len(asd_subs_list)+len(tdh_subs_list)
  all_subs_indices = []
  all_subs_indices.extend(asd_subs_list)
  all_subs_indices.extend(tdh_subs_list)
  reg_col_list = []

  #################### PREPARE REGRESSOR MATRIX COLUMNS ###################
  # First set of subjects are suffering from ASD, next set are TDH subjects.
  asd_clm = [1 if _ < len(asd_subs_list) else 0 for _ in xrange(num_subs)]
  tdh_clm = [0 if _ < len(asd_subs_list) else 1 for _ in xrange(num_subs)]

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
