#
# Author: Ramashish Gaurav
#
# This file implements group level GLM to regress out various regressors to find
# the significance of autistic and healthy individuals. After finding the beta
# values of regressors, it also finds the t value and p value for each brain
# voxel for each ROI. A voxel's t value and p value correspond across all
# subjects (autistic + healthy).
#
# do_group_level_analysis(...) is the main function to be called with appropriate
# arguments. Make sure to accordingly set the global `contrast` variable.
#
# Usage: python get_t_and_p_stat_from_group_glm.py <path/to/func_conn_mat.npy> <
# path/to/regressor_mat.npy> <num_cores>
#
# For more information see the __main__ function.
#
# TODO: Include the condition number criteria too.
#

from multiprocessing import Pool
import numpy as np
import numpy.linalg as npl
import sys

from utility.gla_utilities import (
    get_beta_group_level_glm, get_t_val_p_val_for_voxels)
# Create contrast of autism - healthy with 0's for other regressors.
contrast = np.matrix([1, -1, 0, 0]).T

def do_group_level_glm(params):
  """
  Does a group level analysis.

  Args:
    params (tuple): A 2 element tuple, with below format.
        all_subjects_all_roi_fc_matrix, regressors_matrix = params[0], params[1]

      all_subjects_all_roi_fc_matrix (numpy.ndarray): A 5 dimensional matrix
          where the first dimension is that of the number of subjects (autistic +
          healthy), second dimension denotes the number of ROIs for each subject.
          The last three dimensions are with respect to the brain voxles'
          (x, y, z) functional connectivity scores for a single ROI.

          For example: Suppose dimension of brain is 36 x 36 x 64 and number of
          ROIs is 274 and number of subjects is 120 then the dimension of
          all_subjects_all_roi_fc_matrix will be: 120 x 274 x 36 x 36 x 64.

      regressors_matrix (numpy.ndarray): A 2 dimensional matrix of regressors.
          The first dimension (rows) is with respect to number of subjects and
          the second dimension (columns) is with respect to number of regressors.

  Note:
    Make sure that the regressors_matrix passed in this function has full rank
    i.e. all the columns are linearly independent. Make sure that the order of
    subjects in all_subjects_all_roi_fc_matrix is consistent with the order of
    subjects in regressors_matrix (where each column is a phenotype of subjects).

    One can parallelize this function through data parallelization by dividing
    the ROIs across processors.

  Returns:
    numpy.ndarray: A 4D matrix where first dimension correspond to number of ROIs
        and last three dimensions corresponds to the dimension of brain map where
        each cell is a tuple of (beta-values, error-value) of each voxel in the
        brain obtained after GLM. In other words, number of brains returned =
        number of ROIs.
  """
  all_subjects_all_roi_fc_matrix, regressors_matrix = params[0], params[1]
  num_subs_fc_matrix, rois, bx, by, bz = all_subjects_all_roi_fc_matrix.shape
  num_subs_reg_matrix, num_regressors = regressors_matrix.shape
  # Assert number of subjects in functional connectivity matrix equal to the
  # number of subjects in regressors_matrix.
  assert num_subs_fc_matrix == num_subs_reg_matrix
  # Assert that regressors_matrix is full rank i.e. columns are independent.
  assert npl.matrix_rank(regressors_matrix) == num_regressors

  # List to store the 3D maps of (beta, error) tuple (for each voxel) for rois.
  all_rois_be_tuple_4d_list = []

  for roi in xrange(rois):
    # Do GLM for a single ROI to get a brain map of (beta values, error) tuple.
    single_roi_be_tuple_3d_matrix = _do_glm_for_single_roi_all_subjects(
        all_subjects_all_roi_fc_matrix[:, roi], regressors_matrix)
    # Get a brain map for the ROI where each cell is (t-value, p-value) tuple.
    #single_roi_t_p_tuple_3d_matrix = get_t_val_p_val_for_voxels(
    #    single_roi_be_tuple_3d_matrix, contrast, regressors_matrix)
    all_rois_be_tuple_4d_list.append(single_roi_be_tuple_3d_matrix)

  # Note: all_rois_be_tuple_4d_list is a list, where each element is a 3D matrix
  # in which each cell stores the group level beta and error values as a tuple
  # obtained for respective brain voxels for a single ROI.
  return all_rois_be_tuple_4d_list

def _do_glm_for_single_roi_all_subjects(single_roi_all_subjects_fc_4d_matrix,
                                        regressors_matrix):
  """
  Does a GLM for all the voxels for a single ROI, spanning across all subjects.

  Args:
    single_roi_all_subjects_fc_4d_matrix (numpy.ndarray): A 4D matrix where the
        dimensions are: (number of subjects, bx, by, bz) where (bx, by, bz) are
        the dimensions of a subject's brain.

    regressors_matrix (numpy.ndarray): A 2 dimensional matrix of regressors.
        The first dimension (rows) is with respect to number of subjects and
        the second dimension (columns) is with respect to number of regressors.


  Returns:
    numpy.ndarray: A 3D beta-error matrix of tuples where each cell has a tuple
        of (beta values vector, residual/error vector) corresponding to the voxel.
  """
  num_subjects, bx, by, bz = single_roi_all_subjects_fc_4d_matrix.shape
  _, num_regressors = regressors_matrix.shape

  # Matrix operations are very fast using numpy. So leverage this by creating a
  # 2D matrix Y such that each column corresponds to a brain voxel's correlation
  # score across all subjects. The number of columns in Y will be equal to number
  # of brain voxels, number of rows will be equal to the number of subjects.
  Y = []

  for x in xrange(bx):
    for y in xrange(by):
      for z in xrange(bz):
        Y.append(single_roi_all_subjects_fc_4d_matrix[:, x, y, z])

  Y = np.array(Y).T
  # Assert shape of Y to be equal to the number of subject x number of voxels.
  assert Y.shape == (num_subjects, bx*by*bz)
  # Get the beta values and residual/error values.
  single_roi_all_voxels_beta_mat, single_roi_all_voxels_error_mat = (
      get_beta_group_level_glm(Y, regressors_matrix))

  # Assert shape of single_roi_all_voxels_beta_mat to be equal to number of
  # regressors x number of brain voxels.
  assert single_roi_all_voxels_beta_mat.shape == (num_regressors, bx*by*bz)
  # Assert shape of single_roi_all_voxels_error_mat to be equal to number of
  # subjects x number of voxels.
  assert single_roi_all_voxels_error_mat.shape == (num_subjects, bx*by*bz)
  # Contruct a 3D matrix where each cell stores a tuple of (1D vector of beta
  # values for regressors, 1D vector of residual/error) , corresonding to a
  # brain voxel.
  be_tuple_3d_matrix = np.zeros((bx, by, bz), dtype="object")

  for x in xrange(bx):
    for y in xrange(by):
      for z in xrange(bz):
        # To access a column "c" in single_roi_all_voxels_beta_mat and
        # single_roi_all_voxels_error_mat for a particular brain voxel [x, y, z]
        # set c = by*bz*x + bz*y + z.
        c = by*bz*x + bz*y + z
        be_tuple_3d_matrix[x, y, z] = (single_roi_all_voxels_beta_mat[:, c],
                                       single_roi_all_voxels_error_mat[:, c])

  return be_tuple_3d_matrix

if __name__ == "__main__":
  all_subjects_all_roi_fc_matrix_path = sys.argv[1] # A *.npy file.
  regressors_matrix_path = sys.argv[2] # A *.npy file.
  num_cores = int(sys.argv[3])
  all_subjects_all_roi_fc_matrix = np.load(all_subjects_all_roi_fc_matrix_path)
  regressors_matrix = np.load(regressors_matrix_path)

  # Do data parallelization by dividing all_subjects_all_roi_fc_matrix equally
  # into bins equal to num_cores. Division is on ROIs.
  num_subs_fc_matrix, rois, bx, by, bz = all_subjects_all_roi_fc_matrix.shape
  pool = Pool(num_cores)

  rois_sample = rois / num_cores
  rois_range_list = []
  for core in xrange(num_cores):
    if core == num_cores-1:
      rois_range_list.append((core*rois_sample, rois))
    else:
      rois_range_list.append((core*rois_sample, (core+1)*rois_sample))

  data_input_list = [(
      all_subjects_all_roi_fc_matrix[:, roi_range[0]:roi_range[1]],
      regressors_matrix) for roi_range in rois_range_list]

  all_rois_be_mat_list = pool.map(do_group_level_glm, data_input_list)
  # all_rois_be_mat_list is a list of lists (e.g. [[3D_mat1, 3D_mat2, ...], ...,
  # [3D_mat1, 3D_mat2, ...]]), such that each element list is a list of 3D
  # matrices. The element lists have length equal to the number of rois
  # they intook. Each ROI correspond to one 3D matrix. The extenal list has
  # length equal to the number of processors.
  result = []
  for rois_be_mat_list in all_rois_be_mat_list:
    result.extend(rois_be_mat_list)
  result = np.array(result)
  assert result.shape == (rois, bx, by, bz)
  np.save("all_rois_beta_error_tuple_4D_matrix", result)
