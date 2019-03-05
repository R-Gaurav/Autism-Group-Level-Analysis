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
from scipy import stats
import time

# Create contrast of autism - healthy with 0's for other regressors.
contrast = np.matrix([1, -1, 0, 0]).T

def do_group_level_analysis(params):
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
        and last three dimensions corresponds to the dimension of brain where
        each cell is a tuple of (t-value, p-value) of each voxel in the brain for
        the given `contrast` of regressors. In other words, each ROI has a brain
        map of statistical values i.e. number of brains returned = number of ROIs.
  """
  all_subjects_all_roi_fc_matrix, regressors_matrix = params[0], params[1]
  num_subs_fc_matrix, rois, bx, by, bz = all_subjects_all_roi_fc_matrix.shape
  num_subs_reg_matrix, num_regressors = regressors_matrix.shape
  # Assert number of subjects in functional connectivity matrix equal to the
  # number of subjects in regressors_matrix.
  assert num_subs_fc_matrix == num_subs_reg_matrix
  # Assert that regressors_matrix is full rank i.e. columns are independent.
  assert npl.matrix_rank(regressors_matrix) == num_regressors

  all_rois_stat_values_list = [] # List to store the brain maps of (t, p) stats.

  for roi in xrange(rois):
    # Do GLM for a single ROI to get a brain map of (beta values, error) tuple.
    single_roi_be_tuple_3d_matrix = _do_glm_for_single_roi_all_subjects(
        all_subjects_all_roi_fc_matrix[:, roi], regressors_matrix)
    # Get a brain map for the ROI where each cell is (t-value, p-value) tuple.
    single_roi_t_p_tuple_3d_matrix = get_t_val_p_val_for_voxels(
        single_roi_be_tuple_3d_matrix, contrast, regressors_matrix)

    all_rois_stat_values_list.append(single_roi_t_p_tuple_3d_matrix)

  all_rois_stat_values_matrix = np.array(all_rois_stat_values_list)
  # Assert shape of all_rois_stat_values_matrix to be equal to number_of_rois x
  # x brain's_x_dimension x brain's_y_dimension x brain's_z_dimension.
  assert all_rois_stat_values_matrix.shape == (rois, bx, by, bz)

  # Note: all_rois_stat_values_matrix is a 4D matrix, where each 3D matrix (last
  # three dimensions) denotes the group level stat values obtained for all brain
  # voxels for a single ROI.
  return all_rois_stat_values_matrix

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

def _calculate_t_score_and_p_score(voxel_beta_vec, voxel_error_vec, contrast,
                                   dsgn_matrix_x):
  """
  Calculates a t-score for given contrast and beta vector. The null hypothesis
  is np.dot(contrast.T, voxel_beta_vec) = 0. Also calculates two-tailed p-value
  for the calculated t-score.

  Args:
    voxel_beta_vec (numpy.ndarray): A 1D column matrix of beta values of
        regressors related to a voxel; rows = number of regressors, column = 1.
    voxel_error_vec (numpy.ndarray): A 1D column matrix of residual/error values
        related to a voxel; rows = number of subjects, column = 1.
    contrast (numpy.ndarray): A 1D column matrix of contrast values of beta vals;
        rows = number of regressors, column = 1.
    dsgn_matrix_x (numpy.ndarray): A 2D matrix of regressor vectors, where each
        column is a regressor vector. Number of rows and columns is equal to
        number of subjects and number of regressors considered respectively.

  Returns:
    float, float: A t statistic score, A two-tailed p value.
  """
  # Calculate t-value.
  numerator = contrast.T.dot(voxel_beta_vec)
  var_e = voxel_error_vec.var(ddof=2) # Set DDOF = 2 as two groups are accounted.
  xtx_inv = npl.inv(dsgn_matrix_x.T.dot(dsgn_matrix_x))
  denominator = np.sqrt(var_e * contrast.T.dot(xtx_inv).dot(contrast))
  t_score = float(numerator) / denominator

  # Calculate p-value.
  # Degree of Freedom = Number of subjects - 2 (since 2 groups are accounted).
  dof = voxel_error_vec.shape[0] - 2
  p_value = stats.t.sf(t_score, dof) * 2 # A two tailed p-value.
  return t_score, p_value

def get_t_val_p_val_for_voxels(be_tuple_3d_matrix, contrast, dsgn_matrix_x):
  """
  Calculates t statistic for beta values and gets two tailed p values for those.

  Args:
    be_tuple_3d_matrix: A 3D matrix where each cell stores a tuple of (1D vector
        of beta values for the regressors, 1D vector of residual/error).

    contrast: A 1D array denoting the contrast of the regressors for whom t
        statistic scores have to be obtained. For example, if the regressor
        columns in X are arranged as [autistic, healthy, iq, handedness] and we
        intend to find the contrast between "autistic" and "healthy", the
        "contrast" array should be a 1D column vector:[1, -1, 0, 0].

    dsgn_matrix_x (numpy.ndarray): A 2D matrix of regressor vectors, where each
        column is a regressor vector; rows = number of subjects, columns = number
        of regressors considered.

  Returns:
    numpy.ndarray: A 3D matrix where each cell represent a brain voxel and stores
        a tuple of (T value, P value) for each voxel.
  """
  bx, by, bz = be_tuple_3d_matrix.shape # Get the shape of 3D brain.
  # Create a 3D matrix where each cell denotes a voxel and stores a tuple of
  # (t-value, p-value) corresponding to that voxel.
  stat_3D_matrix = np.zeros((bx, by, bz), dtype="object")

  for x in xrange(bx):
    for y in xrange(by):
      for z in xrange(bz):
        voxel_beta_vec, voxel_error_vec = (
            be_tuple_3d_matrix[x, y, z][0], be_tuple_3d_matrix[x, y, z][1])
        voxel_t_score, voxel_p_score = _calculate_t_score_and_p_score(
            voxel_beta_vec.reshape(-1, 1), voxel_error_vec.reshape(-1, 1),
            contrast, dsgn_matrix_x)
        stat_3D_matrix[x, y, z] = (voxel_t_score, voxel_p_score)

  return stat_3D_matrix

def get_beta_group_level_glm(Y, X):
  """
  Args:
    Y (numpy.ndarray): An 2D array of correlation values (between seed voxel and
                       other brain voxels).
                       e.g.: array([[0.23, 0.12, 0.11, ..., 0.23, 0.87, 0.45,
                                    0.88, ...,  0.76], [0.11, 0.56, ..., 0.88]])
                       where in a column first few values correspond to
                       correlation scores of autistic individuals, rest scores
                       correspond to healthy individuals. Shape of Y is equal to
                       number of subjects x number of voxels.

    X (numpy.ndarray): A 2D array of regressors, where each column is a regressor
                       vector e.g.: array([[1, 2, 3, ..., 4], ..., [2, 3, 4, ...,
                       4]]). Shape of X is equal to number of number of subjects
                       x number of regressors.

  Returns:
    numpy.ndarray, numpy.ndarray: beta matrix (number of regressors x number of
        voxels), residual/error matrix (number of subjects x number of voxels).
  """
  # Get beta values-> B.
  B = npl.inv(X.T.dot(X)).dot(X.T).dot(Y)
  # Get residual/error values-> R.
  R = Y - X.dot(B)

  return B, R

if __name__ == "__main__":
  all_subjects_all_roi_fc_matrix_path = sys.argv[1] # A *.npy file.
  regressors_matrix_path = sys.argv[2] # A *.npy file.
  num_cores = int(sys.argv[3])
  all_subjects_all_roi_fc_matrix = np.load(all_subjects_all_roi_fc_matrix_path)
  regressors_matrix = np.load(regressors_matrix_path)

  # Do data parallelization by dividing all_subjects_all_roi_fc_matrix equally
  # into bins equal to num_cores. Division is on ROIs.
  bx, by, bz, rois, num_subs_fc_matrix = all_subjects_all_roi_fc_matrix.shape
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

  result = pool.map(do_group_level_analysis, data_input_list)
  print len(result)
  print result
