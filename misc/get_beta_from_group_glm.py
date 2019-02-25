#
# Author: Ramashish Gaurav
#
# This file uses nipy GLM to regress out various regressors to find the beta
# values of autistic and healthy individuals.
#

from multiprocessing import Pool
import nipy as nip
import numpy as np
import numpy.linalg as npl
import sys
import time

from nipy.modalities.fmri.glm import GeneralLinearModel
from nipy.modalities.fmri.design import block_amplitudes

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
  # The regressor vectors are in rows first format. Make them column vectors.
  X = X.T
  xtx = np.dot(X.T, X)

  print "X shape: ", X.shape
  print "Y shape: ", Y.shape
  print "Shape of design matrix (X): {0}".format(xtx.shape)
  print "Rank of design_matrix (X): %s" % npl.matrix_rank(xtx)

  start = time.time()
  model = GeneralLinearModel(X) # Here design matrix is regressor matrix.
  model.fit(Y)
  print "Nipy GLM time taken: ", time.time() - start
  print "Nipy GLM Betas: ", model.get_beta()
  print "*" * 80

  start = time.time()
  beta_cap = np.dot(npl.pinv(X), Y)
  print "Python GLM with pinv time taken: ", time.time() - start
  print "Python GLM pinv Betas: ", beta_cap
  print "*" * 80

  start = time.time()
  beta_cap = np.dot(np.dot(npl.inv(np.dot(X.T, X)), X.T), Y)
  print "Python standard GLM time taken: ", time.time() - start
  print "Python standard GLM Betas: ", beta_cap
  print "*" * 80

 start = time.time()
 B = npl.inv(X.T.dot(X)).dot(X.T).dot(Y)
 #print "Python standard GLM-Varun time taken: ", time.time() - start
 print "Python standard GLM-Varun Betas: ", B
 print "Python standard GLM-Varun Betas Shape: ", B.shape
 print "*" * 80
 return B

if __name__ == "__main__":
  all_subjects_all_roi_fc_matrix_path = sys.argsv[1] # A *.npy file.
  regressors_matrix_path = sys.argsv[2] # A *.npy file.
  num_cores = int(sys.argv[3])
  all_subjects_all_roi_fc_matrix = np.load(all_subjects_all_roi_fc_matrix_path)
  regressors_matrix = np.load(regressors_matrix_path)

  # Do data parallelization by dividing all_subjects_all_roi_fc_matrix equally
  # into bins equal to num_cores. Division is on ROIs.
  bx, by, bz, rois, subs_fc_matrix = all_subjects_all_roi_fc_matrix.shape
  pool = Pool(num_cores)
  rois = 274
  num_cores = 4
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
      regressors_matrix) for roi in rois_range_list]

  # TODO: Implement t-values code and accept those as return of below function.
  pool.map(do_group_level_glm, data_input_list)
