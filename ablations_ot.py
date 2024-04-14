from os import system

PROJECT_PATH = 'path_to_your_project'
ACTIVATE_VENV = '. path_to_your_virtualenv/bin/activate'

def tmux(command):
    system(f'tmux ' + command)

def tmux_shell(command, session):
    command = command.replace(' ', ' Space ')
    tmux(f'send-keys -t ' + session + '.0 ' + command + ' ENTER')

dataset = 'wheel'

import time
train = True

dataset_list = ['apple', 'block', 'pillow', 'paper-windmill', 'space-out', 'spin', 'teddy', 'wheel']
import waitGPU

for j in range(1):

    experiments = [
        f'{dataset_list[0]}/base-{dataset_list[0]}-interp-sinkhorn',
        f'{dataset_list[1]}/base-{dataset_list[1]}-interp-sinkhorn',
        f'{dataset_list[2]}/base-{dataset_list[2]}-interp-sinkhorn',
        f'{dataset_list[3]}/base-{dataset_list[3]}-interp-sinkhorn',
        f'{dataset_list[4]}/base-{dataset_list[4]}-interp-sinkhorn',
        f'{dataset_list[5]}/base-{dataset_list[5]}-interp-sinkhorn',
        f'{dataset_list[6]}/base-{dataset_list[6]}-interp-sinkhorn',
        f'{dataset_list[7]}/base-{dataset_list[7]}-interp-sinkhorn'
    ]

    configs = [
        f'--config configs/iphone_dataset/{dataset_list[0]}.py   --wreg 1 --interp 1 --kernel linear --smoothing 0.5 --ot sinkhorn',
        f'--config configs/iphone_dataset/{dataset_list[1]}.py   --wreg 1 --interp 1 --kernel linear --smoothing 0.5 --ot sinkhorn',
        f'--config configs/iphone_dataset/{dataset_list[2]}.py   --wreg 1 --interp 1 --kernel linear --smoothing 0.5 --ot sinkhorn',
        f'--config configs/iphone_dataset/{dataset_list[3]}.py  --wreg 1 --interp 1 --kernel linear --smoothing 0.5 --ot sinkhorn',
        f'--config configs/iphone_dataset/{dataset_list[4]}.py   --wreg 1 --interp 1 --kernel linear --smoothing 0.5 --ot sinkhorn',
        f'--config configs/iphone_dataset/{dataset_list[5]}.py   --wreg 1 --interp 1 --kernel linear --smoothing 0.5 --ot sinkhorn',
        f'--config configs/iphone_dataset/{dataset_list[6]}.py   --wreg 1 --interp 1 --kernel linear --smoothing 0.5 --ot sinkhorn',
        f'--config configs/iphone_dataset/{dataset_list[7]}.py   --wreg 1 --interp 1 --kernel linear --smoothing 0.5 --ot sinkhorn',
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

















