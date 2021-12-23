def downscale_trimesh(trimesh, repeat=1, smooth_iters = 100, decimation_factor=0.5):
    for i in range(repeat):
        trimesh = trimesh.smooth(n_iter=smooth_iters, progress_bar=True)
        trimesh = trimesh.decimate_pro(decimation_factor,progress_bar=True)
    return trimesh