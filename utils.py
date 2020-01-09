import numpy as np
import random
import tensorly
from tensorly.decomposition import robust_pca
import sparse


def find_top_k(score,k):
    score_copy = score.copy()
    max_num = np.amax(score)
    top_idxs = []
    for i in range(k):     
        min_idx = np.unravel_index(np.argmin(score_copy, axis=None), score.shape)
        score_copy[min_idx] = max_num
        top_idxs.append(min_idx)
    return top_idxs

def check_top(idx,top_idxs):
    return 1 if idx in top_idxs else 0


def generate_sparse_score_three_way(score,density_or_samples, is_density=True):
    tot_grid_points = score.size
    num_sample_points = int(round(density_or_samples * score.size)) if is_density else density_or_samples

#     print(num_sample_points)
    testpoints = sorted(random.sample(range(tot_grid_points), num_sample_points))
#     print(testpoints)
    dim_0,dim_1,dim_2 = score.shape
    score_sparse = np.zeros_like(score)
    grid_count = 0
    sample_count = 0
    for k in range(dim_0):
        for i in range(dim_1):    
            for l in range(dim_2):
                if sample_count < num_sample_points and grid_count == testpoints[sample_count]:
                    #print(sample_count)
                    #print("k: ", k)
                    #print("i: ", i)
                    sample_count = sample_count + 1
                    score_sparse[k, i, l] = score[k, i, l]
                grid_count = grid_count + 1
    return score_sparse

def generate_sparse_score_two_way(score,density_or_samples, is_density=True):
    tot_grid_points = score.size
    num_sample_points = int(round(density_or_samples * score.size)) if is_density else density_or_samples
#     print(num_sample_points)
    testpoints = sorted(random.sample(range(tot_grid_points), num_sample_points))
#     print(testpoints)
    dim_0,dim_1 = score.shape
    score_sparse = np.zeros_like(score)
    grid_count = 0
    sample_count = 0
    for k in range(dim_0):
        for i in range(dim_1):                
            if sample_count < num_sample_points and grid_count == testpoints[sample_count]:
                #print(sample_count)
                #print("k: ", k)
                #print("i: ", i)
                sample_count = sample_count + 1
                score_sparse[k, i] = score[k, i]
            grid_count = grid_count + 1
    return score_sparse

def calculate_rec_score(score_sparse,reg_E=1, reg_J=1, mu_init=0.0001, mu_max=10000000000.0, learning_rate=1.1, n_iter_max=100, random_state=None, verbose=1):
    mask = np.zeros_like(score_sparse)
    mask[score_sparse > 0] = 1
    rec_score_tensor,sparse_error_tensor = robust_pca(score_sparse, mask = mask,reg_E=reg_E, reg_J=reg_J, 
        mu_init=mu_init, mu_max=mu_max, learning_rate=learning_rate, n_iter_max=n_iter_max, random_state=random_state, verbose=verbose)
    return rec_score_tensor


def random_search(names,min_vals,max_vals,is_discrete_set):
    assert len(min_vals) == len(max_vals)
    assert len(names) == len(min_vals)
    assert len(min_vals) == len(is_discrete_set)
    config = {}
    for i in range(len(names)):
        name,min_val,max_val,is_discrete = names[i],min_vals[i],max_vals[i],is_discrete_set[i]
        val = np.random.uniform(min_val,max_val)
        if is_discrete:
            val = np.around(val)
        config[name] = val
    return config

def calc_percentage(svals):
    total_val = np.sum(svals)
    return svals/total_val 