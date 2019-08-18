#
# Author: Ramashish Gaurav
#
# This file removes the 3D FC matrix which has all 0's for a particular ROI.
# That particular ROI is 254th ROI (0 based indexing). All subjects are expected
# to have their 254th ROI FC matrix as all zeros matrix. In case there are some
# other FC matrices which are all zeros, then the subject ID and the ROI ID are
# printed.
#

from collections import defaultdict
import nibabel as nib
import numpy as np
import pandas as pd
import pickle
import sys

ROI_WITH_ZERO_FC_MAT = 254 # 0 based indexing
NUM_VALID_ROIS = 273

def construct_subs_all_rois_fc_5D_mats(sub_list):
  """
  This function constructs the complete 5D matrix of all_subs x ROIs x 3D brains
  and removes the 3D FC matrix (254th ROI) of subjects which is all zeros from
  the group of all such 274 3D matrices of a subject. This is done for all the
  subjects in sub_list.

  Args:
    sub_list([]): A list of all subject IDs.
  """
  subs_all_rois_fc_mat = [] # To contain all the subjects' all ROIs FC matrix.
  subs_with_no_fc_mats = [] # To contain all the subjects' IDs with no FC matrix.
  subs_with_fc_maps = [] # To contain all the subjects' IDs with FC matrix.
  # To contain subject IDs as keys and ROIs with all zero FC matrix as values.
  subs_zero_roi_indices = defaultdict(list)

  for sub_id in sub_list:
    try:
      data = nib.load(file_path.format(sub_id, sub_id)).get_fdata() # 4D matrix.
      _, _, _, num_rois = data.shape
      #sub_all_rois_fc_mat = [] # To contain all FC mats except 254th of a subject.
      is_254th_roi_mat_zero = False

      for roi in range(num_rois):
        if np.sum(data[:, :, :, roi]) == 0:
          subs_zero_roi_indices[sub_id].append(roi)
          if roi == ROI_WITH_ZERO_FC_MAT:
            is_254th_roi_mat_zero = True
            continue
          else:
            print(("Subject with ID {} has an all zero FC matrix at {}th "
                   "ROI.".format(sub_id, roi)))

        #sub_all_rois_fc_mat.append(data[:, :, :, roi])

      #sub_all_rois_fc_mat = np.array(sub_all_rois_fc_mat)

      #if sub_all_rois_fc_mat.shape[0] != NUM_VALID_ROIS:
      #  print(("Subject with ID {} does not have {} ROI FC matrices but has "
      #         "{} ROI FC matrices".format(
      #         sub_id, NUM_VALID_ROIS, sub_all_rois_fc_mat.shape[3])))
      #  sys.exit()

      if not is_254th_roi_mat_zero:
        print(("Subject with ID {} does not have 254th ROI FC matrix as all "
               "zeros.".format(sub_id)))
        #sys.exit()

      subs_with_fc_maps.append(sub_id)
      #subs_all_rois_fc_mat.append(sub_all_rois_fc_mat)

    except Exception as e:
      subs_with_no_fc_mats.append(sub_id)
      print("Error: {}, Subject with ID {} does not have a FC brain map".format(
            e, sub_id))

  #subs_all_rois_fc_mat = np.array(subs_all_rois_fc_mat)
  #num_dim = len(subs_all_rois_fc_mat.shape)
  #if num_dim != 5:
  #  print("subs_all_rois_fc_mat is not 5D matrix but {}D matrix".format(num_dim))
  #  sys.exit()

  return (subs_with_fc_maps, subs_all_rois_fc_mat, subs_with_no_fc_mats,
          subs_zero_roi_indices)


if __name__ == "__main__":
  sub_list = pd.read_csv("/home/others/ramashish/Autism-Group-Level-Analysis/"
                         "ABIDE_1_sub_ids.csv")["SUB_ID"].tolist()
  file_path = sys.argv[1]
  output_dir = sys.argv[2]

  file_path = file_path + "/_subject_id_{}/func2std_xform/00{}_fc_map_flirt.nii.gz"
  (subs_with_fc_maps, subs_all_rois_fc_mat, subs_with_no_fc_mats,
      subs_zero_roi_indices) = construct_subs_all_rois_fc_5D_mats(sub_list)
  np.save(output_dir+"/all_subs_all_rois_fc_5D_mat.npy", subs_all_rois_fc_mat)
  np.save(output_dir+"/all_subs_ids_with_fc_mats_list.npy", subs_with_fc_maps)
  np.save(
      output_dir+"/all_subs_ids_with_no_fc_mats_list.npy", subs_with_no_fc_mats)
  pickle.dump(
      subs_zero_roi_indices,
      open(output_dir+"/all_subs_roi_list_with_zero_val_FC_mats.p", "wb"))
  print("DONE!")
