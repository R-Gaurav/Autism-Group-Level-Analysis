from utility.dcp_utilities import (get_valid_subject_ids_with_complete_phenotype,
    get_asd_and_tdh_subs_row_indices, get_processed_design_matrix)
from utility.exp_utilities import (get_valid_subject_ids_wrt_fc_data,
    construct_subs_batchwise_rois_fc_5D_mats, get_regressors_mask_list)
from do_group_level_glm import do_group_level_glm
from do_statistical_analysis import do_statistical_analysis

import pickle
import pandas as pd
import nibabel as nib
import numpy as np
import numpy.linalg as npl
import time

pheno_data_path_1 = "/mnt/project1/home1/varunk/ramashish/data/phenotype_csvs/ABIDE1_Phenotypic.csv"
pheno_data_path_2 = "/mnt/project1/home1/varunk/ramashish/data/phenotype_csvs/ABIDE2_Phenotypic.csv"
fc_files_path1 = "/mnt/project2/home/varunk/fMRI/results/resultsABIDE1_4/functional_connectivity/calc_residual0smoothing1filt1calc_residual_options/"
fc_files_path2 = "/mnt/project2/home/varunk/fMRI/results/resultsABIDE2_1/functional_connectivity/calc_residual0smoothing1filt1calc_residual_options/"

asd_subs, tdh_subs = get_valid_subject_ids_with_complete_phenotype(
    pheno_data_path_1, True)
all_subs = asd_subs + tdh_subs
#print len(all_subs)

k = pickle.load(open("/mnt/project1/home1/varunk/ramashish/data/buggy_subjects/abide_1_bug_sub_ids_from_varun.p", "rb"))

#print len(list(set(all_subs) - set(k)))

#valid_subs, no_fc_subs, zero_fc_subs = get_valid_subject_ids_wrt_fc_data(
#    fc_files_path2, all_subs, False)
#print(len(valid_subs), len(no_fc_subs), len(zero_fc_subs))
#print(valid_subs)


df = pd.read_csv(pheno_data_path_1)
#print df.shape
asd_row_ids, tdh_row_ids = get_asd_and_tdh_subs_row_indices(df, all_subs)
#print df.shape

#start = time.time()
#construct_subs_batchwise_rois_fc_5D_mats(
#    fc_files_path1, df, asd_row_ids[:5], tdh_row_ids[:5], 32, True)
#print time.time() - start

"""
data = nib.load(fc_files_path1+"/_subject_id_{}/func2std_xform/00{}_fc_map_flirt.nii.gz".format(sub_ids[0], sub_ids[0])).get_fdata()
print data[:, :, :, 254].shape, np.sum(data[:, :, :, 254])
k = data[:,:,:, 2:10]
print k.shape
m = data[:,:,:, [2,3,4,10]]
print m.shape
print np.sum(m[:,:,:,3] == data[:,:,:,10])
print data.shape
data1 = np.delete(data, 254, 3)
print data1.shape
#for row_id, sub_id in zip(asd_row_ids + tdh_row_ids, sub_ids):
#  print row_id, sub_id

"""

regressors_mask = get_regressors_mask_list(
    asd=True, tdh=True, leh=True, rih=True, eyo=True, eyc=True, fiq=True, age=True)
design_matrix = get_processed_design_matrix(pd.read_csv(pheno_data_path_1),
    asd_row_ids, tdh_row_ids, regressors_mask, include_intercept=True)
print design_matrix.shape
print npl.matrix_rank(design_matrix)
print npl.cond(design_matrix.T.dot(design_matrix))

#all_subs_all_rois_matrix = generate_all_subs_all_rois_fc_matrix(
#    design_matrix.shape[0], 4, 3, 4, 5)
#contrast = np.array([1, 0, 0, 0, 0, 0, 0]).T
#all_rois_beta_error_values_mat = np.array(do_group_level_glm(
#    (all_subs_all_rois_matrix, design_matrix)))
#all_rois_tp_tuple_mat = do_statistical_analysis(
#    (all_rois_beta_error_values_mat, contrast, design_matrix))
#
#
#print all_rois_tp_tuple_mat
