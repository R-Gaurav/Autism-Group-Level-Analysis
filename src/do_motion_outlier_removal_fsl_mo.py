#
# Author: Ramashish Gaurav
#
# This file does motion outlier removal on the MCFLIRT output data of ABIDE1 and
# ABIDE2.
#
# FIXME: When the "metric" trait in MotionOutliers is mentioned as "dvars" the
# nipype "fsl" code goes buggy and is not able to output "out_file". It says
# file not found. If "metric" is not mentioned and left to default everything
# runs fine. Experiments are run with "metric" set as "dvars" and "fd", hence
# this file couldn't be used. Note: "metric" set as "fd" works.
#

from multiprocessing import Pool
from nipype.interfaces import fsl
from pathlib import Path
import os
import sys

from utility.exp_utilities import get_interval_list

def do_motion_outlier_removal(mcf_func_data_files):
  """
  It calls FSL's fsl_motion_outlier to find out the brain volumes which can be
  deemed as motion outliers.

  Args:
    mcf_func_data_files ([str]): A list of file names of 4D functional data which
                                 are the output of MCFLIRT on raw functional data.
  """
  for func_data in mcf_func_data_files[:5]:
    mo = fsl.MotionOutliers()
    mo.inputs.in_file = files_path+"/"+func_data
    mo.inputs.metric = "dvars"
    try:
      mo.run()
    except:
      print("*"*50)
      print("DVARS")
      print("*"*50)
      pass
    mo.inputs.metric = "fd"
    try:
      mo.run()
    except:
      print("*"*50)
      print("FD")
      print("*"*50)
      pass

if __name__=="__main__":
  files_path = sys.argv[1] # MCFLIRT output data path.
  output_file_path = sys.argv[2]
  num_cores = int(sys.argv[3])

  all_func_data_files = os.listdir(files_path)
  num_range_list = get_interval_list(len(all_func_data_files), num_cores)
  data_input_list = [all_func_data_files[num_range[0]:num_range[1]]
      for num_range in num_range_list]

  pool = Pool(num_cores)
  pool.map(do_motion_outlier_removal, data_input_list)
  #do_motion_outlier_removal(data_input_list[0])
