from os import system

PROJECT_PATH = '/tmp/pycharm_project_165'
ACTIVATE_VENV = '. path_to_your_virtualenv/bin/activate'

def tmux(command):
    system(f'tmux ' + command)

#tmux send-keys -t test10.0 "source activate pytorch" ENTER


def tmux_shell(command, session):


  #  print(f'send-keys -t ' + session + '.0 ' + command + ' ENTER')
    command = command.replace(' ', ' Space ')
    tmux(f'send-keys -t ' + session + '.0 ' + command + ' ENTER')

# example: one tab with vim, other tab with two consoles (vertical split)
# with virtualenvs on the project, and a third tab with the server running

# vim in project
# tmux('new -d -s test11')
# tmux_shell(f'cd Space {PROJECT_PATH}', 'test11')
# tmux_shell('source Space activate Space pytorch', 'test11')
# tmux('rename-window "vim"')

# read a text file and upload to s3 bucket




dataset = 'wheel'

# experiments = [
#     f'{dataset}/base-{dataset}',
#     f'{dataset}/base-{dataset}-wreg',
#     f'{dataset}/base-{dataset}-wreg-r',
#     f'{dataset}/base-{dataset}-wreg-l',
#     f'{dataset}/base-{dataset}-wreg-l-ns',
#     f'{dataset}/base-{dataset}-wreg-t-ns',
#     f'{dataset}/base-{dataset}-wreg-l-r',
#     f'{dataset}/base-{dataset}-wreg-l-ns-r',
# ]

# experiments = [
#     f'{dataset}/base-{dataset}',
#     f'{dataset}/base-{dataset}-wreg',
#     f'{dataset}/base-{dataset}-wreg-r',
#     f'{dataset}/base-{dataset}-r',
#     f'{dataset}/base-{dataset}-wreg-l-ns',
#     f'{dataset}/base-{dataset}-wreg-t-ns',
#     f'{dataset}/base-{dataset}-wreg-l-r',
#     f'{dataset}/base-{dataset}-wreg-l-ns-r',
# ]

# configs = [
#     f'--config configs/iphone_dataset/{dataset}.py --wreg 0 --rndm_bck 0',
#     f'--config configs/iphone_dataset/{dataset}.py --wreg 1 --interp 0 --diff 0',
#     f'--config configs/iphone_dataset/{dataset}.py --wreg 1 --interp 0 --diff 0 --rndm_bck 1',
#     f'--config configs/iphone_dataset/{dataset}.py --wreg 1 --interp 1 --diff 0 --kernel linear --smoothing 1.0',
#     f'--config configs/iphone_dataset/{dataset}.py --wreg 1 --interp 1 --diff 0 --kernel linear --smoothing 0.0',
#     f'--config configs/iphone_dataset/{dataset}.py --wreg 1 --interp 1 --diff 0 --kernel thin_plate_spline',
#     f'--config configs/iphone_dataset/{dataset}.py --wreg 1 --interp 1 --diff 0 --kernel linear --smoothing 1.0 --rndm_bck 1',
#     f'--config configs/iphone_dataset/{dataset}.py --wreg 1 --interp 1 --diff 0 --kernel linear --smoothing 0.0 --rndm_bck 1',
# ]

# security

#Iterating over every two elements in a list







import time
train = True

dataset_list = ['aleks-teapot', 'torchocolate', 'hand', 'lemon', 'chickchicken']
import waitGPU

for j in range(1):
  #  print(f'Waiting to train {dataset_list[i]} and {dataset_list[i+1]}')
   # waitGPU.wait(utilization=10, memory_ratio=0.2, available_memory=300,
    #             gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7], interval=10 * 60, nproc=1, ngpu=8)


    # dataset1 = dataset_list[i]
    # dataset2 = dataset_list[i+1]
    # dataset2 = dataset_list[i + 1]

    experiments = [
        f'{dataset_list[0]}/base-{dataset_list[0]}-interp',
        f'{dataset_list[1]}/base-{dataset_list[1]}-interp',
        f'{dataset_list[2]}/base-{dataset_list[2]}-interp',
        f'{dataset_list[3]}/base-{dataset_list[3]}-interp',
        f'{dataset_list[4]}/base-{dataset_list[4]}-interp'
    ]

    configs = [
        f'--config configs/interp_dataset/{dataset_list[0]}.py --interp_dataset 1  --wreg 1 --interp 1 --kernel linear --smoothing 0.5',
        f'--config configs/interp_dataset/{dataset_list[1]}.py --interp_dataset 1  --wreg 1 --interp 1 --kernel linear --smoothing 0.5',
        f'--config configs/interp_dataset/{dataset_list[2]}.py --interp_dataset 1  --wreg 1 --interp 1 --kernel linear --smoothing 0.5',
        f'--config configs/interp_dataset/{dataset_list[3]}.py --interp_dataset 1  --wreg 1 --interp 1 --kernel linear --smoothing 0.5',
        f'--config configs/interp_dataset/{dataset_list[4]}.py --interp_dataset 1  --wreg 1 --interp 1 --kernel linear --smoothing 0.5'
    ]

    #print(f'Training {dataset_list[i]} and {dataset_list[i+1]}')
    for i, exp in enumerate(experiments):
            session = exp.split('/')[-1].partition('-')[-1] #
            cmd_create_session = f'new -d -s {session}'
            tmux(cmd_create_session)
            tmux_shell(f'cd {PROJECT_PATH}', session)
            tmux_shell('source activate pytorch', session)
            tmux_shell(f'python run.py {configs[i]} --gpunum {i} --expname {exp}', session)

    for i, exp in enumerate(experiments):
            session = exp.split('/')[-1].partition('-')[-1]
            tmux_shell(f'cd {PROJECT_PATH}', session)
            tmux_shell('source activate pytorch', session)
            tmux_shell(f'python run.py {configs[i]} --render_test --render_only --eval_psnr --gpunum {i} --expname {exp}', session)

    time.sleep(5*60)

#check the gpu memory usage
# tmux_shell('python run.py --config configs/iphone_dataset/paper-windmill/base-paper-windmill.py --render_test --render_only --eval_psnr --gpunum 0 --expname paper-windmill/base-paper-wind
# stop tmux session
# tmux_shell('exit', session)




# def perturb_camera(R, T, focal_points=-100.0, noise_scale=0.2):
#
#     def m3dLookAt_np(mz, up):
#         mx = np.cross(up, mz)
#         mx /= np.linalg.norm(mx)
#         my = np.cross(mz, mx)
#         my /= np.linalg.norm(my)
#         return np.asarray([mx[0], my[0], mz[0], mx[1], my[1], mz[1], mx[2], my[2], mz[2]])
#
#     up = np.asarray([0, 1, 0])
#     up = R @ up
#
#     direction = R[:, 2]
#     far_point = T + direction * focal_points
#
#     T_add_random = noise_scale * np.random.uniform(size=[3])
#     T_new = T + T_add_random
#     direction_new = (T_new - far_point) / np.linalg.norm(T_new - far_point)
#     R_new = np.reshape(m3dLookAt_np(direction_new, up), [3,3])
#
#     return R_new, T_new
















