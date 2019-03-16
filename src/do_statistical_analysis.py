#
# Author: Ramashish Gaurav
#
# This file finds the t value and p value for each brain voxel for each ROI. A
# voxel's t value and p value correspond across all subjects (autistic +
# healthy).
#
# Note: It does a two tailed t test.
#
# Usage: python do_statistical_analysis.py <path/to/beta_error_tuple_mat.npy> <
# path/to/regressor_mat.npy> <num_cores>
#

from multiprocessing import Pool
import numpy as np
import numpy.linalg as npl
import sys

from utility.gla_utilities import get_t_val_p_val_for_voxels
from utility.exp_utilities import get_rois_range_list

def do_statistical_analysis(params):
  """
  Does a statistical analysis i.e.finds t and p values for each brain voxel in
  the passed 3D matrix.

  Args:
    params (tuple): A 3 element tuple with below format:
      rois_4D_be_tuple_matrix, contrast, regressors_matrix = (
          params[0], params[1], params[2])

    rois_4D_be_tuple_matrix (numpy.ndarray): A 4D matrix where the first
        dimension corresponds to the number of ROIs and last three dimensions
        corresponds to the dimensions of the brain for a single ROI.

    contrast (numpy.nderray): A 1D column matrix which denotes the contrast of
        the regressors.

    regressors_matrix (numpy.ndarray): A 2D matrix of regressors, where number
        of rows is equal to the number of subjects and number of columns is
        equal to the number of regressors.
  """
  rois_4d_be_tuple_matrix, contrast, regressors_matrix = (
      params[0], params[1], params[2])
  rois, bx, by, bz = rois_4d_be_tuple_matrix.shape
  all_rois_tp_tuple_3d_mat_list = []

  for roi in range(rois):
    # Get a brain map for the ROI where each cell is (t-value, p-value) tuple.
    single_roi_t_p_tuple_3d_matrix = get_t_val_p_val_for_voxels(
        rois_4d_be_tuple_matrix[roi], contrast, regressors_matrix)
    all_rois_tp_tuple_3d_mat_list.append(single_roi_t_p_tuple_3d_matrix)
    print "Statistical Analysis, ROI: %s Done!" % roi

  # Note: all_rois_tp_tuple_3d_mat_list is a list, where each element is a 3D
  # matrix of brain in which each cell stores a tuple of (t-value, p-value) with
  # respect to each voxel of the brain. Each element corresonds to a single ROI.
  return all_rois_tp_tuple_3d_mat_list

if __name__ == "__main__":
  all_rois_be_tuple_matrix_path = sys.argv[1] # A *.npy file.
  regressor_matrix_path = sys.argv[2] # A *.npy file.
  num_cores = int(sys.argv[3])

  all_rois_be_tuple_matrix = np.load(all_rois_be_tuple_matrix_path)
  regressors_matrix = np.load(regressor_matrix_path) # num_subjects x num_regrs
  rois, bx, by, bz = all_rois_be_tuple_matrix.shape
  pool = Pool(num_cores)

  # Create contrast of autism - healthy with 0's for other regressors.
  contrast = np.matrix([1, -1, 0, 0]).T
  # Assert that number of rows of contrast is equal to the number of columns
  # in columns in regressor matrix.
  assert contrast.shape[0] == regressors_matrix.shape[1]
  # Assert that regressors_matrix is full rank i.e. columns are independent.
  assert npl.matrix_rank(regressors_matrix) == regressors_matrix.shape[1]

  # Get the condition number of regressor_matrix, if it is very high then the
  # solutions of the linear system of equations (GLM) is prone to large
  # numerical errors as the matrix is mostly singular i.e. non invertible.
  print "Condition number of regressors_matrix (design matrix): %s" % npl.cond(
      regressors_matrix)

  # Do data parallelization to divide all_rois_be_tuple_matrix to each processor.
  # Division is on ROIs.
  rois_range_list = get_rois_range_list(rois, num_cores)
  data_input_list = [(
      all_rois_be_tuple_matrix[roi_range[0]:roi_range[1]], contrast,
      regressors_matrix) for roi_range in rois_range_list]

  all_rois_tp_stat_mat_list = pool.map(do_statistical_analysis, data_input_list)
  # all_rois_tp_stat_mat is a list of lists (e.g. [[3D_mat1, 3D_mat2, ...], ...,
  # [3D_mat1, 3D_mat2, ...]]), such that each element list is a list of 3D
  # matrices. The element lists have length equal to the number of ROIs they
  # intook. Each ROI corresponds to one 3D matrix. The external list has length
  # equal to the number of processors passed in arguments.
  result = []
  for rois_tp_stat_mat_list in all_rois_tp_stat_mat_list:
    result.extend(rois_tp_stat_mat_list)
  result = np.array(result)
  assert result.shape == (rois, bx, by, bz)
  np.save("all_rois_tp_stat_matrix", result)
  print "Result saved in all_rois_tp_stat_matrix.npy matrix"
