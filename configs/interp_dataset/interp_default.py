from copy import deepcopy

expname = None                    # experiment name
basedir = './logs/'               # where to store ckpts and logs

''' Template of data options
'''
data = dict(
    datadir=None,                 # path to dataset root folder
    dataset_type=None,            
    load2gpu_on_the_fly=True,    # do not load all images into gpu (to save gpu memory)
    testskip=1,                   # subsample testset to preview results
    white_bkgd=False,             # use white background (note that some dataset don't provide alpha and with blended bg color)
    half_res=True,              
    factor=4,                     
    ndc=False,                    # use ndc coordinate (only for forward-facing; not support yet)
    spherify=False,               # inward-facing
    llffhold=8,                   # testsplit
    load_depths=False,            # load depth
    use_bg_points=True,
    add_cam=False,
    data_interval = 4

)

''' Template of training options
'''
train_config = dict(
    N_iters=20000,                # number of optimization steps
    N_rand=2048,                  # batch size (number of random rays per optimization step)
    lrate_feature=1e-1,           # lr of voxel grid
    lrate_featurenet=1e-3,
    lrate_deformation_net=7e-4,
    lrate_densitynet=1e-3,
    lrate_timenet=1e-3,
    lrate_camnet=1e-3,
    lrate_rgbnet=1e-3,           # lr of the mlp 
    lrate_decay=20,               # lr decay by 0.1 after every lrate_decay*1000 steps
    ray_sampler='in_maskcache',        # ray sampling strategies
    weight_main=1.0,              # weight of photometric loss
    weight_entropy_last=0.001,
    weight_rgbper=0.01,            # weight of per-point rgb loss
    tv_every=1,                   # count total variation loss every tv_every step
    tv_after=0,                   # count total variation loss from tv_from step
    tv_before=1e9,                   # count total variation before the given number of iterations
    tv_feature_before=10000,            # count total variation densely before the given number of iterations
    weight_tv_feature=1e-5,
    pg_scale=[2000, 4000, 6000, 8000],
    skip_zero_grad_fields=['feature'],
    wreg=False,
    interval=0.1,
    sample_size = 4096,
    weight_wreg = 0.1,
    wreg_off = 20000,
    gpunum = 7,
    gaussian_smoothing = False,
    weightnet = False,
    interp = False,
    kernel = 'linear',
    smoothing = 0.0,
    epsilon = 1.,
    diff = False,
    rndm_bck = False,
    depthloss = False,
    interp_dataset = False,
)

''' Template of model and rendering options
'''

model_and_render = dict(
    num_voxels=160**3,          # expected number of voxel
    num_voxels_base=160**3,      # to rescale delta distance
    voxel_dim=6,                 # feature voxel grid dim
    defor_depth=3,               # depth of the deformation MLP 
    net_width=256,             # width of the  MLP
    alpha_init=1e-3,              # set the alpha values everywhere at the begin of training
    fast_color_thres=1e-4,           # threshold of alpha value to skip the fine stage sampled point
    stepsize=0.5,                 # sampling stepsize in volume rendering
    world_bound_scale=1.05,
)



del deepcopy
