from os import system

PROJECT_PATH = 'path_to_your_project'
ACTIVATE_VENV = '. path_to_your_virtualenv/bin/activate'

def tmux(command):
    system(f'tmux ' + command)

def tmux_shell(command, session):


  #  print(f'send-keys -t ' + session + '.0 ' + command + ' ENTER')
    command = command.replace(' ', ' Space ')
    tmux(f'send-keys -t ' + session + '.0 ' + command + ' ENTER')

dataset = 'wheel'


import time
train = True

dataset_list = ['sriracha-tree', 'pillow', 'mochi-high-five']
import waitGPU
for dataset in dataset_list:
    print(f'Waiting to train {dataset}')
    waitGPU.wait(utilization=10, memory_ratio=0.2, available_memory=300,
                 gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7], interval=10 * 60, nproc=1, ngpu=8)


    experiments = [
        f'{dataset}/base-{dataset}',
        f'{dataset}/base-{dataset}-r',
        f'{dataset}/base-{dataset}-d',
        f'{dataset}/base-{dataset}-w-i-l-s',
        f'{dataset}/base-{dataset}-w-r-i-l-s',
        f'{dataset}/base-{dataset}-w-d-i-l-s',
        f'{dataset}/base-{dataset}-w-r-d-i-l-s',
        f'{dataset}/base-{dataset}-w',
    ]


    configs = [
        f'--config configs/iphone_dataset/{dataset}.py',
        f'--config configs/iphone_dataset/{dataset}.py --rndm_bck 1',
        f'--config configs/iphone_dataset/{dataset}.py --depthloss 1',
        f'--config configs/iphone_dataset/{dataset}.py --wreg 1 --interp 1 --kernel linear --smoothing 1.0',
        f'--config configs/iphone_dataset/{dataset}.py --wreg 1 --rndm_bck 1 --interp 1 --kernel linear --smoothing 1.0',
        f'--config configs/iphone_dataset/{dataset}.py --wreg 1 --depthloss 1 --interp 1 --kernel linear --smoothing 1.0',
        f'--config configs/iphone_dataset/{dataset}.py --wreg 1 --rndm_bck 1 --depthloss 1 --interp 1 --kernel linear --smoothing 1.0',
        f'--config configs/iphone_dataset/{dataset}.py --wreg 1',
    ]

    print(f'Training {dataset}')
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


