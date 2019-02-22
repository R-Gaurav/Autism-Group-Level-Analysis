#
# Author: Ramashish Gaurav
#
# This file uses nipy GLM to regress out various regressors to find the beta
# values of autistic and healthy individuals.
#

import nipy as nip
import numpy as np
import numpy.linalg as npl
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
    numpy.ndarray: Shape-> number of regressors X 1
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
  print "Python standard GLM-Varun time taken: ", time.time() - start
  print "Python standard GLM-Varun Betas: ", B
  print "*" * 80
