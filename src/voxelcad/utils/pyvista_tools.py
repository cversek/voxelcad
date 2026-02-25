import voxelcad.environment as ENV

def downscale_trimesh(trimesh, repeat=1, smooth_iters = 100, decimation_factor=0.5):
    for i in range(repeat):
        if smooth_iters > 0:
            trimesh = trimesh.smooth(n_iter=smooth_iters, progress_bar=ENV.progress_bar)
        trimesh = trimesh.decimate_pro(decimation_factor,progress_bar=ENV.progress_bar)
    return trimesh
