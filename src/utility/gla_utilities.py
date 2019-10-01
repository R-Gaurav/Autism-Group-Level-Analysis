#
# Author: Ramashish Gaurav
#
# This file contains the utility functions which aid in performing group level
# analysis.
#

import numpy.linalg as npl
import numpy as np
from scipy import stats

import log

def get_beta_and_error_group_level_glm(Y, X):
  """
  Does GLM:
      http://www.brainvoyager.com/bvqx/doc/UsersGuide/StatisticalAnalysis/TheGeneralLinearModel.html

  Args:
    Y (numpy.ndarray): An 2D array of correlation values (between seed voxel and
                       other brain voxels).
                       e.g.: array([[0.23, 0.12, 0.11, ..., 0.23, 0.87, 0.45,
                                    0.88, ...,  0.76], [0.11, 0.56, ..., 0.88]])
                       where in a column first few values correspond to
                       correlation scores of autistic individuals, rest scores
                       correspond to healthy individuals. Shape of Y is equal to
                       number of subjects x number of voxels in the brain.

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

  Ref:
    http://www.brainvoyager.com/bvqx/doc/UsersGuide/StatisticalAnalysis/TheGeneralLinearModel.html
    https://matthew-brett.github.io/teaching/glm_intro.html
    Basic Econometrics: Book by Damodar N. Gujarati, Appendix C

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
  # Error should be that obtained after GLM for a voxel.
  # Degree of Freedom = Number of subjects - rank of dsgn_matrix_x. In the
  # get_design_matrix_for_the_exp() of file
  # create_design_matrix_and_contrast_for_exp.py it has been asserted that
  # npl.matrix_rank(dsgn_matrix_x) is equal to the number of columns in the
  # dsgn_matrix_x. Calculation of npl.matrix_rank() for such a similar matrix
  # takes 0.02 secs on average, hence to save time, one can also set rank of
  # dsgn_matrix_x = # number of columns in dsgn_matrix_x.
  assert voxel_error_vec.shape[0] == dsgn_matrix_x.shape[0]
  dof = voxel_error_vec.shape[0] - npl.matrix_rank(dsgn_matrix_x)

  # In BrainVoyager, variance of error (i.e. Var(e)) is used instead of Mean
  # Residual Sum of Squares (MRSS). One should note that MRSS is equal to Var(e)
  # in case mean of error vector is 0. Hence, one could also use following
  # formula to calculate Standard Error (i.e. "denominator"):
  # var_e = voxel_error_vec.var(ddof=dsgn_matrix_x.shape[1]) in place of "mrss".
  # I have verified in case when mean of voxel_error_vec is 0, mrss and var_e are
  # same. The OLS solution of GLM assumes that residual/error (here
  # voxel_error_vec) has a multivariate Normal distribution with mean 0. Hence
  # an ideal solution of GLM will always produce voxel_error_vec with mean 0.
  mrss = float(voxel_error_vec.T.dot(voxel_error_vec)) / dof
  xtx_inv = npl.inv(dsgn_matrix_x.T.dot(dsgn_matrix_x))
  denominator = np.sqrt(mrss * contrast.T.dot(xtx_inv).dot(contrast))
  t_score = float(numerator) / denominator

  # Calculate p-value.
  p_value = stats.t.sf(np.abs(t_score), dof) * 2 # A two tailed p-value.

  return t_score, p_value

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
  num_subs_reg_matrix, _ = regressors_matrix.shape
  # Assert number of subjects in functional connectivity matrix equal to the
  # number of subjects in regressors_matrix.
  assert num_subs_fc_matrix == num_subs_reg_matrix

  # List to store the 3D maps of (beta, error) tuple (for each voxel) for rois.
  all_rois_be_tuple_3d_mat_list = []

  for roi in xrange(rois):
    # Do GLM for a single ROI to get a brain map of (beta values, error) tuple.
    single_roi_be_tuple_3d_matrix = _do_glm_for_single_roi_all_subjects(
        all_subjects_all_roi_fc_matrix[:, roi], regressors_matrix)
    all_rois_be_tuple_3d_mat_list.append(single_roi_be_tuple_3d_matrix)
    log.INFO("Group Level GLM, ROI: %s Done!" % roi)

  # Note: all_rois_be_tuple_3d_mat_list is a list, where each element is a 3D
  # matrix in which each cell stores the group level beta and error values as a
  # tuple obtained for respective brain voxels for a single ROI.
  return np.array(all_rois_be_tuple_3d_mat_list)

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
      get_beta_and_error_group_level_glm(Y, regressors_matrix))
  del Y
  del single_roi_all_subjects_fc_4d_matrix

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

def do_statistical_analysis(params):
  """
  Does a statistical analysis i.e. finds t and p values for each brain voxel in
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

  Returns:
    list: The list of the tuple of t and p values.
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
    log.INFO("Statistical Analysis, ROI: %s Done!" % roi)

  # Note: all_rois_tp_tuple_3d_mat_list is a list, where each element is a 3D
  # matrix of brain in which each cell stores a tuple of (t-value, p-value) with
  # respect to each voxel of the brain. Each element corresonds to a single ROI.
  return all_rois_tp_tuple_3d_mat_list
