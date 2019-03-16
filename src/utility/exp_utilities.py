#
# Author: Ramashish Gaurav
#
# This file contains the utility functions which aid in experiments.
#

def get_rois_range_list(num_rois, num_cores):
  """
  Returns a list of tuple of ROI ranges with length equal to the number of cores.

  Args:
    num_rois (int): Number of ROIs.
    num_cores (int): Number of cores.

  Returns:
    []: A list of (roi_start_of_range, roi_end_of_range).
  """
  rois_sample = num_rois / num_cores
  rois_range_list = []
  for core in xrange(num_cores):
    if core == num_cores-1:
      rois_range_list.append((core*rois_sample, num_rois))
    else:
      rois_range_list.append((core*rois_sample, (core+1)*rois_sample))

  return rois_range_list
