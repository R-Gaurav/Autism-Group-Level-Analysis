#
# Author: Ramashish Gaurav
#
# This file contains the utility functions which aid in experiments.
#

import nibabel as nib
import numpy as np
import pickle

ROI_WITH_ZERO_FC_MAT = 254 # 0 based indexing.
TOTAL_NUMBER_OF_ROIS = 274
OUTPUT_FILE_PATH = "/mnt/scratch/svn50/ramashish/all_subs_batch_wise_ROI_fc_matrix"

def get_interval_list(num, num_cores):
  """
  Returns a list of tuple of ranges with length equal to the number of cores.

  Args:
    num (int): Number which has to be divided into intervals.
    num_cores (int): Number of cores.

  Returns:
    []: A list of (roi_start_of_range, roi_end_of_range).
  """
  num_sample = num // num_cores
  num_range_list = []
  for core in range(num_cores):
    if core == num_cores-1:
      num_range_list.append((core*num_sample, num))
    else:
      num_range_list.append((core*num_sample, (core+1)*num_sample))

  return num_range_list

def get_valid_subject_ids_with_buggy_subs(is_abide1):
  """
  Returns a list of subject IDs which are buggy (identified by Varun).

  Args:
    is_abide1 (bool): Which dataset's buggy subjects IDs to return?

  Returns:
    [int]
  """
  buggy_subs_path = "/mnt/project1/home1/varunk/ramashish/data/buggy_subjects/"
  if is_abide1:
    buggy_subs_path += "abide_1_bug_sub_ids_from_varun.p"
  else:
    buggy_subs_path += "abide_2_bug_sub_ids_from_varun.p"

  return pickle.load(open(buggy_subs_path, "rb"))

def get_valid_subject_ids_wrt_fc_data(fc_files_path, sub_list, is_abide1):
  """
  Returns lists of subjects. First list denotes the valid subjects SUB_ID which
  have only the 254th ROI (0 based indexing) FC matrix zero or no ROI FC matrix
  as all zero. Second list denotes the subjects' SUB_ID who have no FC matrices.
  Third list of subjects' SUB_ID have ROIs other than 254th one as all zero FC
  matrix.

  The subject IDs with no FC matrices are discarded. Subject IDs with zero
  valued FC matrices other than the 254th one are also discarded.

  Args:
    fc_files_path (str): /Path/to/FC/matrices/func2std_xform
    is_abide1 (bool): Are the passed args with respect to ABIDE-1?
    sub_list ([int]): A list of subject IDs for which FC matrices are analysed.

  Returns:
    [int], [int], [int]: valid subs, subs with no FC, subs with zero FCs.
        Note that these lists contain the SUB_ID of subjects.
  """
  if is_abide1:
    fc_files_path += "/_subject_id_{}/func2std_xform/00{}_fc_map_flirt.nii.gz"
  else:
    fc_files_path += "/_subject_id_{}/func2std_xform/{}_fc_map_flirt.nii.gz"

  # Remove the buggy subjects identified by Varun.
  buggy_subjects = get_valid_subject_ids_with_buggy_subs(is_abide1)
  sub_list = list(set(sub_list) - set(buggy_subjects))

  subs_with_no_fc_mats = [] # Contains subject IDs having no FC matrices.

  # Contains subject IDs having all zero FC matrices at ROI other than 254th.
  subs_with_zero_fc_mats = []

  # Contains subject IDs which have either no ROI FC matrix as all 0 or have
  # only the 254th ROI FC matrix as 0.
  all_valid_subs = []

  for sub_id in sub_list:
    try:
      data = nib.load(fc_files_path.format(sub_id, sub_id)).get_fdata() # 4D mat.
      _, _, _, num_rois = data.shape
      sub_zero_rois = [] # Contains the subject's ROIs indices which are all 0.
      for roi in range(num_rois):
        if np.sum(data[:, :, :, roi]) == 0:
          sub_zero_rois.append(roi)

      # Ignore subjects having zero FC matrix at ROI other than the 254th one.
      if sub_zero_rois == [] or sub_zero_rois == [ROI_WITH_ZERO_FC_MAT]:
        all_valid_subs.append(sub_id)
      else:
        print("Subject: %s has zero FC matrices at ROIs: %s"
              % (sub_id, sub_zero_rois))
        subs_with_zero_fc_mats.append(sub_id)

    except Exception as e: # Ignore subjects having no FC matrices.
      print("Error: {}, Subject with ID {} does not have a FC brain map".format(
            e, sub_id))
      subs_with_no_fc_mats.append(sub_id)

  return all_valid_subs, subs_with_no_fc_mats, subs_with_zero_fc_mats

def construct_subs_batchwise_rois_fc_5D_mats(fc_files_path, df, asd_row_ids,
                                             tdh_row_ids, batch_size, is_abide1):
  """
  This function constructs the 5D matrices of all_subs x ROIs x 3D brains in
  batches of ROIs such that batches of all_subs x [batch_start, batch_end]ROIs x
  3D brain is saved in each iteration of batches.

  This function expects row IDs of subjects who are considered valid, and
  removes the 254th ROI FC matrix.

  Note: Output files are save at OUTPUT_FILE_PATH (macro defined above).

  Args:
    fc_files_path (str): File path to the FC matrices.
    df (pandas.DataFrame): ABIDE1 or ABIDE2 Phenotype dataframe.
    asd_row_ids ([int]): List of ASD subs row indices.
    tdh_row_ids ([int]): List of TDH subs row indices.
    batch_size (int): Batch size of ROIs to be saved.
    is_abide1 (bool): True if passed args are with respect to ABIDE1 else False.
  """
  sub_ids = df.loc[asd_row_ids]["SUB_ID"].tolist()
  sub_ids.extend(df.loc[tdh_row_ids]["SUB_ID"].tolist())

  if is_abide1:
    fc_files_path += "/_subject_id_{}/func2std_xform/00{}_fc_map_flirt.nii.gz"
  else:
    fc_files_path += "/_subject_id_{}/func2std_xform/{}_fc_map_flirt.nii.gz"

  for batch_start in xrange(0, TOTAL_NUMBER_OF_ROIS, batch_size):
    batch_end = min(batch_start + batch_size, TOTAL_NUMBER_OF_ROIS)
    roi_slice = range(batch_start, batch_end)
    # Remove the all zero FC matrix at 254th ROI.
    if ROI_WITH_ZERO_FC_MAT >= batch_start and ROI_WITH_ZERO_FC_MAT <= batch_end:
      roi_slice.remove(ROI_WITH_ZERO_FC_MAT)

    all_subs_batch_fc_matrix = []
    for sub_id in sub_ids:
      data = nib.load(fc_files_path.format(sub_id, sub_id)).get_fdata() # 4D mat.
      sub_batch_fc_matrix = []
      for roi in roi_slice:
        sub_batch_fc_matrix.append(data[:, :, :, roi])
      all_subs_batch_fc_matrix.append(sub_batch_fc_matrix)

    all_subs_batch_fc_matrix = np.array(all_subs_batch_fc_matrix)
    print("ROI_SLICE: %s, all_subs_batch_fc_matrix.shape: %s" % (
          roi_slice, all_subs_batch_fc_matrix.shape))
    assert len(all_subs_batch_fc_matrix.shape) == 5
    np.save(OUTPUT_FILE_PATH+"/all_subs_%s_start_%s_end_ROIs_fc_matrix.npy" % (
            batch_start, batch_end), all_subs_batch_fc_matrix)
    print("Batch %s to %s done!" % (batch_start, batch_end))

def get_regressors_mask_list(asd=None, tdh=None, leh=None, rih=None, eyo=None,
                             eyc=None, fiq=None, age=None):
  """
  Creates and returns a regressors mask.

  Args:
    asd (bool): True if ASD column is to be included in the design matrix.
    tdh (bool): True if TDH column is to be included in the design matrix.
    leh (bool): True if left hand column is to be included in the design matrix.
    rih (bool): True if right hand column is to be included in the design matrix.
    eyo (bool): True if eye open column is to be included in the design matrix.
    eyc (bool): True if eye closed column is to be included in the design matrix.
    fiq (bool): True if FIQ column is to be included in the design matrix.
    age (bool): True if AGE_AT_SCAN column is to be included in the design matrix.

  Returns:
    [int]: list of bools e.g. [1, 0, 0, ...]
  """
  regressors_mask = []

  regressors_mask.append(1 if asd else 0)
  regressors_mask.append(1 if tdh else 0)
  regressors_mask.append(1 if leh else 0)
  regressors_mask.append(1 if rih else 0)
  regressors_mask.append(1 if eyo else 0)
  regressors_mask.append(1 if eyc else 0)
  regressors_mask.append(1 if fiq else 0)
  regressors_mask.append(1 if age else 0)

  return regressors_mask
