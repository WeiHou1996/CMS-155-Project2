import numpy as np
import matplotlib.pyplot as plt

def projection(u,v, bias_movie=0, bias_user=0):
    u_mean_cetered = u - np.mean(u, axis=0)[np.newaxis,:]
    v_mean_cetered = v - np.mean(v, axis=0)[np.newaxis,:]
    [u_U, S, u_Vh] = np.linalg.svd(np.transpose(u_mean_cetered))
    [v_U, S, v_Vh] = np.linalg.svd(np.transpose(v_mean_cetered))
    u_U_2 = u_U[:,:2]
    u_proj = np.matmul(np.transpose(u_U_2), np.transpose(u_mean_cetered))
    v_U_2 = v_U[:,:2]
    v_proj = np.matmul(np.transpose(v_U_2), np.transpose(v_mean_cetered))

    return u_proj, v_proj

def plot(u,v, filename, movie_select, movie_names, moving_names = "", moving_dist_x = 0, moving_dist_y = 0):
    u_proj, v_proj = projection(u,v)
    fig,ax = plt.subplots()
    #fig.set_size_inches(10,10)
    v_proj_sub0 = v_proj[0, movie_select]
    v_proj_sub1 = v_proj[1, movie_select]
    ax.scatter(v_proj_sub0, v_proj_sub1)
    for i in range(len(movie_select)):
        stringv = movie_names[i, 1]
        print(stringv)
        if stringv == moving_names:
            ax.annotate(stringv, (v_proj_sub0[i] + moving_dist_x, v_proj_sub1[i]+moving_dist_y), size=8)
        else:
            ax.annotate(stringv, (v_proj_sub0[i], v_proj_sub1[i]), size=8)
        

    #ax.set_xlim(-1,1.5)
    ax.grid(which='major')
    plt.xlabel("Latent Feature 1")
    plt.ylabel("Latent Feature 2")
    plt.savefig(filename)
    return