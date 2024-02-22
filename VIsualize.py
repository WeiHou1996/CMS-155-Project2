import numpy as np

def projection(u,v, bias_movie=0, bias_user=0):
    [u_U, S, u_Vh] = np.linalg.svd(np.transpose(u))
    [v_U, S, v_Vh] = np.linalg.svd(np.transpose(v))
    u_U_2 = u_U[:,:2]
    u_proj = np.matmul(np.transpose(u_U_2), np.transpose(u))
    v_U_2 = v_U[:,:2]
    v_proj = np.matmul(np.transpose(v_U_2), np.transpose(v))

    return u_proj, v_proj