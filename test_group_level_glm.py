# coding: utf-8
from get_beta_from_group_glm import get_beta_group_level_glm
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
    return matrix

y = generate_y(120)
x = generate_x(120)

get_beta_group_level_glm(y,x)
