#
# Author: Ramashish Gaurav
#
# This file does motion correction on ABIDE1 and ABIDE2 functional data.
#

from multiprocessing import Pool
from nipype.interfaces import fsl
import os
import sys

from utility.exp_utilities import get_interval_list

def do_motion_correction(func_data_files):
  """
  It calls FSL's MCFLIRT to do motion correction/alignment operation. Note
  that, even after running MCFLIRT it is not able to correct extreme motion
  outliers. Hence, run fsl_motion_outliers with appropriate options and get
  rid of the motion outliers.

  Args:
    func_data_files ([str]): A list of file names of 4D functional data on which
                             MCFLIRT is to be run.

  """
  for func_data in func_data_files:
    mcflirt = fsl.MCFLIRT(in_file=files_path+"/"+func_data)
    mcflirt.inputs.cost = "mutualinfo"
    mcflirt.inputs.out_file = output_file_path+"/mcf_"+func_data
    res = mcflirt.run()
    print("MCFLIRT on file: %s ended!" % func_data)


if __name__=="__main__":
  files_path = sys.argv[1] # Raw Functional data path.
  output_file_path = sys.argv[2]
  num_cores = int(sys.argv[3])

  all_func_data_files = os.listdir(files_path)
  num_range_list = get_interval_list(len(all_func_data_files), num_cores)
  data_input_list = [all_func_data_files[num_range[0]:num_range[1]]
      for num_range in num_range_list]

  pool = Pool(num_cores)
  pool.map(do_motion_correction, data_input_list)
