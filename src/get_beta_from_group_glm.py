#
# Author: Ramashish Gaurav
#
# This file implements group level GLM to regress out various regressors to find
# the beta values of autistic and healthy individuals.
#

from multiprocessing import Pool
import numpy as np
import numpy.linalg as npl
import sys
import time

def do_group_level_glm(params):
  """
  Does a group level analysis.

  Args:
    params (tuple): A 2 element tuple, with below format.
        all_subjects_all_roi_fc_matrix, regressors_matrix = params[0], params[1]

      all_subjects_all_roi_fc_matrix (numpy.ndarray): A 5 dimensional matrix
          where the first dimension is that of the number of subjects, second
          dimension denotes the number of ROIs for each subject. The last three
          dimensions are with respect to the brain voxles' (x, y, z) functional
          connectivity scores for a single ROI.

          For example: Suppose dimension of brain voxel's is 36 x 36 x 64 and
          number of ROIs is 274 and number of subjects is 120 then the dimension
          of all_subjects_all_roi_fc_matrix will be: 120 x 274 x 36 x 36 x 64.

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
  numpy.ndarray, int, int, int: A 3D matrix, number of ROIs, by and bz, where by
      and bz are the yth and zth dimension of brain volume.
  """
  all_subjects_all_roi_fc_matrix, regressors_matrix = params[0], params[1]
  subs_fc_matrix, rois, bx, by, bz = all_subjects_all_roi_fc_matrix.shape
  subs_reg_matrix, num_regressors = regressors_matrix.shape

  # Assert number of subjects in functional connectivity matrix equal to the
  # number of subjects in regressors_matrix.
  assert subs_fc_matrix == subs_reg_matrix
  # Assert that regressors_matrix is full rank i.e. columns are independent.
  assert npl.matrix_rank(regressors_matrix) == num_regressors

  all_rois_beta_values_list = []

  # Matrix operations are very fast using numpy. So leverage this by creating a
  # 2D matrix Y such that each column corresponds to a brain voxel's correlation
  # score across all subjects. The number of columns in Y will be equal to number
  # of brain voxels, number of rows will be equal to the number of subjects.

  for roi in xrange(rois):
    Y = []
    for x in xrange(bx):
      for y in xrange(by):
        for z in xrange(bz):
          Y.append(all_subjects_all_roi_fc_matrix[:, roi, x, y, z])

    Y = np.array(Y).T
    # Assert shape of Y to be equal to the number of subject x number of voxels.
    assert Y.shape == (subs_fc_matrix, bx*by*bz)
    # Get the beta values for current "roi".
    single_roi_all_voxels_beta_vals = get_beta_group_level_glm(
        Y, regressors_matrix)
    # Assert shape of single_roi_all_voxels_beta_vals to be equal to number of
    # regressors x number of brain voxels.
    assert single_roi_all_voxels_beta_vals.shape == (num_regressors, bx*by*bz)
    all_rois_beta_values_list.append(single_roi_all_voxels_beta_vals)

  all_rois_beta_values_matrix = np.array(all_rois_beta_values_list)
  # Assert shape of all_rois_beta_values_matrix to be equal to number of rois x
  # number of regressors x number of brain voxels.
  assert all_rois_beta_values_matrix.shape == (rois, num_regressors, bx*by*bz)

  # Note: all_rois_beta_values_matrix is a 3D matrix, where each 2D matrix
  # denotes the group level beta values obtained for all brain voxels for a
  # single ROI. Each column in the 2D matrix corresponds to the beta values of
  # all regressors obtained for one voxel's correlation score. Hence, to access
  # a column "c" for a particular brain voxel [x,y,z] where c = by*bz*x + bz*y +
  # z for a particular ROI, you need to access it as
  # all_rois_beta_values_matrix[roi][:,c].
  return all_rois_beta_values_matrix, rois, by, bz

def get_beta_group_level_glm(Y, X):
  """
  Args:
    Y (numpy.ndarray): An 1D array of correlation values (between seed voxel and
                       other brain voxels).
                       e.g.: array([0.23, 0.12, 0.11, ..., 0.23, 0.87, 0.45,
                                    0.88, ...,  0.76])
                       where first few values correspond to correlation scores of
                       autistic individuals, rest scores correspond to healthy
                       individuals.
    X (numpy.ndarray): A 2D array of regressors, where each row is a regressor
                       vector of same length as "Y"'s length. e.g.: array([[1,
                       2, 3, ..., 4], ..., [2, 3, 4, ..., 4]])

  Returns:
    numpy.ndarray
  """
  B = npl.inv(X.T.dot(X)).dot(X.T).dot(Y)
  return B

if __name__ == "__main__":
  all_subjects_all_roi_fc_matrix_path = sys.argv[1] # A *.npy file.
  regressors_matrix_path = sys.argv[2] # A *.npy file.
  num_cores = int(sys.argv[3])
  all_subjects_all_roi_fc_matrix = np.load(all_subjects_all_roi_fc_matrix_path)
  regressors_matrix = np.load(regressors_matrix_path)

  # Do data parallelization by dividing all_subjects_all_roi_fc_matrix equally
  # into bins equal to num_cores. Division is on ROIs.
  bx, by, bz, rois, subs_fc_matrix = all_subjects_all_roi_fc_matrix.shape
  pool = Pool(num_cores)

  rois_sample = rois / num_cores
  rois_range_list = []
  for core in xrange(num_cores):
    if core == num_cores-1:
      rois_range_list.append((core*rois_sample, rois))
    else:
      rois_range_list.append((core*rois_sample, (core+1)*rois_sample))
  print rois_range_list

  data_input_list = [(
      all_subjects_all_roi_fc_matrix[:, roi_range[0]:roi_range[1]],
      regressors_matrix) for roi_range in rois_range_list]

  # TODO: Implement t-values code and accept those as return of below function.
  print pool.map(do_group_level_glm, data_input_list)
