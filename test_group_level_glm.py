# coding: utf-8
from src.get_beta_from_group_glm import get_beta_group_level_glm, do_group_level_glm
import numpy as np

def generate_y(n):
  return np.random.rand(n,1)

def generate_x(n):
  intercept = [1 for i in xrange(n)]
  autism = [1 if i < n/2 else 0 for i in xrange(n)]
  healthy = [0 if i < n/2 else 1 for i in xrange(n)]
  iq = [np.random.randint(160) for i in range(n)]
  handedness = [np.random.choice((1,0)) for i in range(n)]
  #matrix = np.array([intercept, autism, healthy, iq, handedness])
  matrix = np.array([autism, healthy, iq, handedness])
  return matrix.T

y = []
def generate_all_subs_all_rois_fc_matrix(subs, rois, bx, by, bz):
  all_subs_all_rois_fc_matrix = []
  for sub in xrange(subs):
    all_rois_fc_matrix = []
    for roi in xrange(rois):
      k = np.random.random((bx, by, bz))
      all_rois_fc_matrix.append(k)
      y.append(k[0,0,0])
    all_subs_all_rois_fc_matrix.append(all_rois_fc_matrix)

  return np.array(all_subs_all_rois_fc_matrix)

num_subs = 12
x = generate_x(num_subs)
all_subs_all_rois_matrix = generate_all_subs_all_rois_fc_matrix(
    num_subs, 4, 3, 4, 5)
np.save("all_subs_all_rois_matrix.npy", all_subs_all_rois_matrix)
np.save("x.npy", x)
#y = np.array(y).T
#print "get_beta_group_level_glm ans: "
#get_beta_group_level_glm(y,x)
print "Shape all_subs_all_rois_matrix from test: ", all_subs_all_rois_matrix.shape
print "Shape regressors_matrix from test: ", x.shape
all_rois_beta_values_matrix, rois, by, bz = do_group_level_glm(
    (all_subs_all_rois_matrix, x))
print "do_group_level_glm ans: ", all_rois_beta_values_matrix
