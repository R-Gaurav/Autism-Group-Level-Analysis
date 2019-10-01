#
# Author: Ramashish Gaurav
#
# This file contains the utility functions which aid in experiments.
#

import nibabel as nib
import numpy as np
import pickle

from . import log
from . constants import (
    ROI_WITH_ZERO_FC_MAT, TOTAL_NUMBER_OF_ROIS, OUTPUT_FILE_PATH, ABIDE_1,
    ABIDE_2, ABIDE_1_BW_ROI_FC_DATA, ABIDE_2_BW_ROI_FC_DATA, EXP_DIR,
    FIRST_LEVEL_DATA, ABIDE_1_ALL_ROIS_LOGQ_MATS, ABIDE_2_ALL_ROIS_LOGQ_MATS)

def get_interval_list(num, num_cores):
  """
  Returns a list of tuple of ranges with length equal to the number of cores.

  Args:
    num (int): Number which has to be divided into intervals.
    num_cores (int): Number of cores.

  Returns:
    []: A list of (roi_start_of_range, roi_end_of_range).
  """
  num_cores = num if num < num_cores else num_cores
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
  buggy_subs_path = "%s/data/buggy_subjects/" % EXP_DIR
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
  matrix. Fourth list of subjects' SUB_ID have ROIS matrices but not equal to
  TOTAL_NUMBER_OF_ROIS.

  The subject IDs with no FC matrices are discarded. Subject IDs with zero
  valued FC matrices other than the 254th one are also discarded, as well as
  subject IDs who have ROIs less than (or more than?) the TOTAL_NUMBER_OF_ROIS.

  Args:
    fc_files_path (str): /Path/to/FC/matrices/func2std_xform
    is_abide1 (bool): Are the passed args with respect to ABIDE-1?
    sub_list ([int]): A list of subject IDs for which FC matrices are analysed.

  Returns:
    [int], [int], [int], [int]: valid subs, subs with no FC, subs with zero FCs.
        subs with ROIs not equal to TOTAL_NUMBER_OF_ROIS.
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

  # Contains subject IDs having all zero FC matrices for ROI other than 254th.
  subs_with_zero_fc_mats = []

  # Contains subject IDs who do not have all the 274 FC matrices.
  subs_with_incmp_fc_mats = []

  # Contains subject IDs which have either no ROI FC matrix as all 0 or have
  # only the 254th ROI FC matrix as 0.
  all_valid_subs = []

  for sub_id in sub_list:
    try:
      data = nib.load(fc_files_path.format(sub_id, sub_id)).get_fdata() # 4D mat.
      _, _, _, num_rois = data.shape
      if num_rois != TOTAL_NUMBER_OF_ROIS:
        log.INFO("Subject: %s has incomplete number of ROIs" % sub_id)
        subs_with_incmp_fc_mats.append(sub_id)
        continue

      sub_zero_rois = [] # Contains the subject's ROIs indices which are all 0.

      for roi in range(num_rois):
        if np.all(data[:, :, :, roi] == 0): # Are all the cells 0?
          sub_zero_rois.append(roi)

      # Ignore subjects having zero FC matrix at ROI other than the 254th one.
      if sub_zero_rois == [] or sub_zero_rois == [ROI_WITH_ZERO_FC_MAT]:
        all_valid_subs.append(sub_id)
      else:
        log.INFO("Subject: %s has zero FC matrices at ROIs: %s"
                 % (sub_id, sub_zero_rois))
        subs_with_zero_fc_mats.append(sub_id)

      if data.any(): # If there is an ob ref by `data` delete it to save memory.
        del data # Delete the object reference.

    except Exception as e: # Ignore subjects having no FC matrices.
      log.ERROR("Error: {}, Subject with ID {} does not have a FC brain "
               "map".format(e, sub_id))
      subs_with_no_fc_mats.append(sub_id)

  return (all_valid_subs, subs_with_no_fc_mats, subs_with_zero_fc_mats,
          subs_with_incmp_fc_mats)

def construct_subs_batchwise_rois_fc_5D_mats(fc_files_path, df, asd_row_ids,
                                             tdh_row_ids, batch_size, is_abide1):
  """
  This function constructs the 5D matrices of all_subs x ROIs x 3D brains in
  batches of ROIs such that batches of all_subs x [batch_start, batch_end]ROIs x
  3D brain is saved in each iteration of batches.

  This function expects row IDs of subjects who are considered valid, and
  removes the 254th ROI FC matrix.

  Note: Output files are saved at OUTPUT_FILE_PATH (macro defined above).

  Args:
    fc_files_path (str): File path to the FC matrices.
    df (pandas.DataFrame): ABIDE_1 or ABIDE_2 Phenotype dataframe.
    asd_row_ids ([int]): List of ASD subs row indices.
    tdh_row_ids ([int]): List of TDH subs row indices.
    batch_size (int): Batch size of ROIs to be saved.
    is_abide1 (bool): True if passed args are with respect to ABIDE_1 else False.
  """
  output_file_path = OUTPUT_FILE_PATH

  sub_ids = df.loc[asd_row_ids]["SUB_ID"].tolist()
  sub_ids.extend(df.loc[tdh_row_ids]["SUB_ID"].tolist())

  if is_abide1:
    fc_files_path += "/_subject_id_{}/func2std_xform/00{}_fc_map_flirt.nii.gz"
    output_file_path += (ABIDE_1 + FIRST_LEVEL_DATA + ABIDE_1_BW_ROI_FC_DATA)
  else:
    fc_files_path += "/_subject_id_{}/func2std_xform/{}_fc_map_flirt.nii.gz"
    output_file_path += (ABIDE_2 + FIRST_LEVEL_DATA + ABIDE_2_BW_ROI_FC_DATA)

  for batch_start in xrange(0, TOTAL_NUMBER_OF_ROIS, batch_size):
    batch_end = min(batch_start + batch_size, TOTAL_NUMBER_OF_ROIS)
    roi_slice = range(batch_start, batch_end)
    # Remove the all zero FC matrix at 254th ROI.
    if ROI_WITH_ZERO_FC_MAT >= batch_start and ROI_WITH_ZERO_FC_MAT < batch_end:
      roi_slice.remove(ROI_WITH_ZERO_FC_MAT)

    all_subs_batch_fc_matrix = []
    for sub_id in sub_ids: # sub_ids has first ASD subs and then TDH subs.
      data = nib.load(fc_files_path.format(sub_id, sub_id)).get_fdata() # 4D mat.
      sub_batch_fc_matrix = []
      for roi in roi_slice:
        sub_batch_fc_matrix.append(data[:, :, :, roi].copy())
      all_subs_batch_fc_matrix.append(sub_batch_fc_matrix)
      del data
      del sub_batch_fc_matrix
      log.INFO("Batch: [%s, %s], sub ID: %s done."
               % (batch_start, batch_end-1, sub_id))

    log.INFO("Converting all_subs_batch_fc_matrix to np.array")
    all_subs_batch_fc_matrix = np.array(all_subs_batch_fc_matrix)
    log.INFO("ROI_SLICE: %s, all_subs_batch_fc_matrix.shape: %s" % (
             roi_slice, all_subs_batch_fc_matrix.shape))
    assert len(all_subs_batch_fc_matrix.shape) == 5
    np.save(output_file_path+"/all_subs_%s_start_%s_end_ROIs_fc_5D_matrix.npy"
            % (batch_start, batch_end), all_subs_batch_fc_matrix)
    log.INFO("ROIs batch %s to %s inclusive done!" % (batch_start, batch_end-1))
    del all_subs_batch_fc_matrix

def get_regressors_mask_list(asd=None, tdh=None, leh=None, rih=None, eyo=None,
                             eyc=None, fiq=None, age=None, ipt=None):
  """
  Creates and returns a regressors mask and a regressors string vector.

  Args:
    asd (bool): True if ASD column is to be included in the design matrix.
    tdh (bool): True if TDH column is to be included in the design matrix.
    leh (bool): True if left hand column is to be included in the design matrix.
    rih (bool): True if right hand column is to be included in the design matrix.
    eyo (bool): True if eye open column is to be included in the design matrix.
    eyc (bool): True if eye closed column is to be included in the design matrix.
    fiq (bool): True if FIQ column is to be included in the design matrix.
    age (bool): True if AGE_AT_SCAN column is to be included in the design matrix.
    ipt (bool): True if Intercept column is to be included in the design matrix.

  Returns:
    [int], str: list of bools e.g. [1, 0, 0, ...], regressors string vector.
  """
  regressors_mask = []
  regressors_strv = ""

  if asd:
    regressors_mask.append(1)
    regressors_strv += "ASD_"
  else:
    regressors_mask.append(0)

  if tdh:
    regressors_mask.append(1)
    regressors_strv += "TDH_"
  else:
    regressors_mask.append(0)

  if leh:
    regressors_mask.append(1)
    regressors_strv += "LEH_"
  else:
    regressors_mask.append(0)

  if rih:
    regressors_mask.append(1)
    regressors_strv += "RIH_"
  else:
    regressors_mask.append(0)

  if eyo:
    regressors_mask.append(1)
    regressors_strv += "EYO_"
  else:
    regressors_mask.append(0)

  if eyc:
    regressors_mask.append(1)
    regressors_strv += "EYC_"
  else:
    regressors_mask.append(0)

  if fiq:
    regressors_mask.append(1)
    regressors_strv += "FIQ_"
  else:
    regressors_mask.append(0)

  if age:
    regressors_mask.append(1)
    regressors_strv += "AGE_"
  else:
    regressors_mask.append(0)

  # Make sure that this mask for the intercept is always at last.
  if ipt:
    regressors_mask.append(1)
    regressors_strv += "IPT"
  else:
    regressors_mask.append(0)

  return regressors_mask, regressors_strv

def get_3d_stat_mat_from_tp_mat(tp_tpl_3d_mat, is_t_val=True):
  """
  Constructs a 3D matrix with each and every voxel having either a t value or
  p-value depending on the value of `is_t_val`.

  Args:
    tp_tpl_3d_mat (numpy.ndarray): A 3D matrix with each voxel containing a
        tuple of (t-value, p-value) in the same order.
    is_t_val (bool): True if t-value is the stat of interest else False for p-val.

  Returns:
    numpy.ndarray: A 3D matrix with all voxels containing either a t-val or p-val.
  """
  bx, by, bz =  tp_tpl_3d_mat.shape
  stat_3d_mat = np.zeros((bx, by, bz))
  # First value in tuple is a t-value and second value is a p-value.
  tp_index = 0 if is_t_val else 1

  for x in range(bx):
    for y in range(by):
      for z in range(bz):
        if not np.isnan(tp_tpl_3d_mat[x,y,z][tp_index][0,0]):
          stat_3d_mat[x,y,z] = tp_tpl_3d_mat[x,y,z][tp_index][0,0]

  return stat_3d_mat

def get_3d_stat_mat_to_nii_gz_format(params):
  """
  Obtains 3d stat matrices in nii.gz format.

  Args:
    params (tuple): A 2 element tuple with format below:
      tp_tpl_stat_4d_mat, is_t_val = params[0], params[1]

    tp_tpl_stat_4d_mat (numpy.ndarray): A 4D tp tuple stat matrix.
    is_t_val (bool): If True construct a 3D mat with only t-values else p-values.

  Returns:
    [Nifti1Image() object]: A list of objects to a 3D stat matrix.
  """
  tp_tpl_stat_4d_mat, is_t_val = params[0], params[1]
  rois, _, _, _ = tp_tpl_stat_4d_mat.shape

  all_rois_stat_object_list = []

  for roi in range(rois):
    stat_3d_mat = get_3d_stat_mat_from_tp_mat(
        tp_tpl_stat_4d_mat[roi], is_t_val=is_t_val)
    stat_img = nib.Nifti1Image(stat_3d_mat, np.eye(4))
    all_rois_stat_object_list.append(stat_img)

  return all_rois_stat_object_list

def create_3d_brain_nii_from_4d_mat(mat_4d, regressors_strv, contrast_str,
                                    stat, is_abide1):
  """
  Creates 3d brain file from the passed `mat_4d` and saves it in nii.gz format.

  Args:
    mat_4d (numpy.ndarray): A 4D matrix of dimension (bx, by, bz, #ROIs)
    regressors_strv (str): A string denoting the regressors chosen.
    contrast_str (str): A string denoting the contrast chosen.
    stat (str): Which stat? e.g. p, t, q, logq, etc.
    is_abide1 (bool): True if this function is called for ABIDE-1 else False.
  """
  if is_abide1:
    abide = ABIDE_1
    all_rois_logq_output = ABIDE_1_ALL_ROIS_LOGQ_MATS
  else:
    abide = ABIDE_2
    all_rois_logq_output = ABIDE_2_ALL_ROIS_LOGQ_MATS

  _, _, _, num_rois = mat_4d.shape
  for roi in range(num_rois):
    brain_file = mat_4d[:, :, :, roi]
    brain_img = nib.Nifti1Image(brain_file, np.eye(4))
    nib.save(brain_img,
             OUTPUT_FILE_PATH + abide + FIRST_LEVEL_DATA + all_rois_logq_output +
             "%s_ROI_%s_stat_dsgn_%s_contrast_%s_3d_brain_file.nii.gz" % (
             roi, stat, regressors_strv, contrast_str))
    log.INFO("%s ROI %s stat brain file saved!" % (roi, stat))
