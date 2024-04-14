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




