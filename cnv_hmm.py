import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cptac
import biograder
#The copy number variation homework is #4
biograder.download("bio462_hw4")
hw = biograder.bio462_hw4(student_id = 'CG_Final_Project') #Instantiate the homework grader object.

# !pip install cptac

### General Outline
# Read and grab the CNV data (similar to Q3 of HW4)
# Use HMMs to predict the states for affecting CNV

# download BRCA data
cptac.download('brca')
brca = cptac.Brca()
brca_CNV = brca.get_CNV()

brca_CNV_t = brca_CNV.transpose()

# get locations data
locations = hw.getData(name='gene_locations')
locations = locations.copy()
locations = locations.set_index("Database_ID")

# join locations and BRCA data
merged_df = locations.merge(brca_CNV_t, left_on='Database_ID', right_on='Database_ID')
merged_nd = merged_df[~merged_df.index.duplicated(keep='first')]

# select for chromosme 17 and order by starting position
brca_cnv_chr = merged_nd.loc[merged_nd['chromosome'] == '17']
brca_cnv_chr = brca_cnv_chr.sort_values(by=['start_bp'])

# re-organize (transpose BRCA data)
brca_cnv_chr.drop(brca_cnv_chr.columns[[0,1,2]], axis=1, inplace=True)
brca_cnv_chr_t = brca_cnv_chr.transpose()

# print out first few rows of current BRCA CNV table
print(brca_cnv_chr_t.head(3))

# next steps
# Use HMM models to figure out how to predict the BRCA stuff
# Can do this with other cancer types too?
# Each state has its own emission values (effect) on CNV fold change...
# we will use the CNV values to generate estimates of the HMM parameters and states?

# initialize parameters
# v_init = 0.5
# w_init = 0.5
# vw_trans = 0.25
# wv_trans = 0.25
# vv_trans = 1 - vw_trans
# ww_trans = 1 - wv_trans
# v_obs = {'A': 0.3, 'C': 0.2, 'G': 0.2, 'T': 0.3}
# w_obs = {'A': 0.2, 'C': 0.3, 'G': 0.3, 'T': 0.2}

# Viterbi decoding
# def RunViterbiDecoding(v_init, w_init, v_obs, w_obs, vv_trans, vw_trans, wv_trans, ww_trans, obs_0):
#     obs_len = len(obs_0)
#     viterbi_matrix = np.zeros((obs_len+1, obs_len+1))

#     viterbi_matrix[0][0] = 0

#     viterbi_v = v_init * v_obs[obs_0[0]]
#     viterbi_w = w_init * w_obs[obs_0[0]]
#     for t in range(obs_len): # update first row and first col
#         viterbi_matrix[t+1][0] = viterbi_v
#         viterbi_v = viterbi_v * vv_trans * v_obs[obs_0[t]]
#         viterbi_matrix[0][t+1] = viterbi_w
#         viterbi_w = viterbi_w * ww_trans * w_obs[obs_0[t]]
    
#     for i in range(1, obs_len+1):
#         for j in range(1, obs_len+1):
#             viterbi_matrix[i][j] = max(viterbi_matrix[i][j-1] * v_trans )
#             aligned_matrix[i][j] = max(aligned_matrix[i-1][j-1] + score(seq1[i-1], seq2[j-1]), aligned_matrix[i-1][j] + gap, aligned_matrix[i][j-1] + gap)


#####
    # viterbi_v = v_init * v_obs[obs_0[0]]
    # bt_v = ''
    # viterbi_w = w_init * w_obs[obs_0[0]]
    # bt_w = ''

    # # range through the entire observation sequence
    # for t in range(1, len(obs_0)):
    #     new_viterbi_v = max(viterbi_v * vv_trans * v_obs[obs_0[t]], viterbi_w * wv_trans * v_obs[obs_0[t]])

    #     index = str(np.argmax([viterbi_v * vv_trans, viterbi_w * wv_trans])+1)
    #     temp_btv = bt_v

    #     if index == '1':
    #         bt_v = bt_v + index
    #     else: # switch
    #         bt_v = bt_w + index
        
    #     new_viterbi_w = max(viterbi_v * vw_trans * w_obs[obs_0[t]], viterbi_w * ww_trans * w_obs[obs_0[t]])
    #     index = str(np.argmax([viterbi_v * vw_trans, viterbi_w * ww_trans])+1)

    #     if index == '1':
    #         bt_w = temp_btv + index
    #     else:
    #         bt_w = bt_w + index

    #     viterbi_v = new_viterbi_v
    #     viterbi_w = new_viterbi_w
        
    # # picking best path of the two (one ending with vampires, the other werewolves)
    # best_path = np.argmax([viterbi_v, viterbi_w]) + 1
    # if best_path == 1:
    #     states_set = bt_v + '1'
    # if best_path == 2:
    #     states_set = bt_w + '2'

    # return states_set

# # update the emission probabilities
# def UpdateEmissionProbabilities(states_set, obs_0, v_obs, w_obs):
#     obs_va = 0
#     obs_vc = 0
#     obs_vg = 0
#     obs_vt = 0
#     obs_wa = 0
#     obs_wc = 0
#     obs_wg = 0
#     obs_wt = 0
#     v_count = 0
#     w_count = 0
#     for t in range(len(obs_0)):
#         curr_state = states_set[t]
#         if curr_state == '1': # 'V'
#             v_count += 1
#             if obs_0[t] == 'A':
#                 obs_va += 1
#             if obs_0[t] == 'C':
#                 obs_vc += 1
#             if obs_0[t] == 'G':
#                 obs_vg += 1
#             if obs_0[t] == 'T':
#                 obs_vt += 1
#         else: # 'W'
#             w_count += 1
#             if obs_0[t] == 'A':
#                 obs_wa += 1
#             if obs_0[t] == 'C':
#                 obs_wc += 1
#             if obs_0[t] == 'G':
#                 obs_wg += 1
#             if obs_0[t] == 'T':
#                 obs_wt += 1
#     if v_count != 0:
#         va_freq = obs_va / v_count
#         vc_freq = obs_vc / v_count
#         vg_freq = obs_vg / v_count
#         vt_freq = obs_vt / v_count
#         v_obs = {'A': va_freq, 'C': vc_freq, 'G': vg_freq, 'T': vt_freq}
#     if w_count != 0:
#         wa_freq = obs_wa / w_count
#         wc_freq = obs_wc / w_count
#         wg_freq = obs_wg / w_count
#         wt_freq = obs_wt / w_count
#         w_obs = {'A': wa_freq, 'C': wc_freq, 'G': wg_freq, 'T': wt_freq}
#     return (v_obs, w_obs)

# # update the transition probabilities
# def UpdateTransitionProbabilities(states_set, obs_0):
#     vw_counts = 0
#     wv_counts = 0
#     for i in range(len(states_set) - 1):
#         curr_state = states_set[i]
#         next_state = states_set[i+1]
#         if curr_state == '1' and next_state == '2':
#             vw_counts += 1
#         if curr_state == '2' and next_state == '1':
#             wv_counts += 1
#         vw_prob = vw_counts / (len(states_set) - 1)
#         wv_prob = wv_counts / (len(states_set) - 1)
#     return (vw_prob, wv_prob)

# # run until converge
# print('Start iterations')
# for i in range(2000):
#     states_set = ''
#     for observation in train_fasta.values():
#         obs_seq = str(observation)[2:-2]
#         # print("new seq")
#         # print(v_obs)
#         # print(w_obs)
#         states_set += (RunViterbiDecoding(v_init, w_init, v_obs, w_obs, vv_trans, vw_trans, wv_trans, ww_trans, obs_seq))

#     # getting new values for the parameters
#     new_v_obs, new_w_obs = UpdateEmissionProbabilities(states_set, obs_seq, v_obs, w_obs)
#     new_vw_trans, new_wv_trans = UpdateTransitionProbabilities(states_set, obs_seq)
#     new_vv_trans = 1-new_vw_trans
#     new_ww_trans = 1-new_wv_trans

#     if new_v_obs == v_obs and new_w_obs == w_obs and new_vw_trans == vw_trans and new_wv_trans == wv_trans and new_vv_trans == vv_trans and new_ww_trans == ww_trans:
#         break

#     # set old parameters to new parameters (updating them)
#     v_obs = new_v_obs
#     w_obs = new_w_obs
#     vw_trans = new_vw_trans
#     wv_trans = new_wv_trans
#     vv_trans = new_vv_trans
#     ww_trans = new_ww_trans
#     # print("WW:", ww_trans, "VV:", vv_trans, "WV", wv_trans, "VW", vw_trans)
# print("after training:", wv_trans)