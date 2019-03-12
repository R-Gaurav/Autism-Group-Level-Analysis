#
# Author: Ramashish Gaurav
#
# This file contains the utility functions which aid in performing group level
# analysis.
#

import numpy.linalg as npl
import numpy as np
from scipy import stats

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
