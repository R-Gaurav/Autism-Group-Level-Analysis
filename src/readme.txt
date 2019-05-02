This directory contains files which are required for the end to end second level
analysis pipeline i.e. Group Level Analysis.

> create_fc_matrices_for_exp.py: This file creates the 5D matrices
    (all_valid_subs x batch_ROIs x bx x by x bz) from the raw FC matrices for
    each of the 274 ROIs of each subjects obtained as the result of first level
    analysis.

> do_group_level_glm_analysis.py: This file intakes the 5D matrices
    (all_valid_subs x batch_ROIs x bx x by x bz) obtained from the
    create_fc_matrices_for_exp.py and does a Group Level GLM on those matrices
    to output 4D matrices (batch_ROIs x bx x by x bz) where each element of a
    certain ROIth matrix (a 3D matrix) is a tuple of (beta_values, error_values)
    obtained after GLM.

> do_statistical_analysis.py: This file intakes the 4D matrices
    (batch_ROIs x bx x by x bz) obtained from the do_group_level_glm_analysis.py
    and performs statistical analysis on it to create 4D matrices
    (batch_ROIs x bx x by x bz) where each element of a certain ROIth matrix (a
    3D matrix) is a tuple of (t-value, p-value) which is used to create the
    diagram of a particular ROI's connectivity with rest of the brain areas.

> create_design_matrix_for_exp.py: This file creates the desired design matrix
    to be used in the execution of both `do_group_level_glm_analysis.py` and
    `do_statistical_analysis.py` files. Note that, both these files use the same
    design matrix.

> utility/ : This dir contains the utility files.
