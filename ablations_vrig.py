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

dataset_list = ['tail', 'toby-sit']
import waitGPU

for i in range(0, len(dataset_list), 2):
    print(f'Waiting to train {dataset_list[i]} and {dataset_list[i+1]}')
    waitGPU.wait(utilization=10, memory_ratio=0.2, available_memory=300,
                 gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7], interval=10 * 60, nproc=1, ngpu=8)


    dataset1 = dataset_list[i]
    dataset2 = dataset_list[i+1]

    experiments = [
        f'{dataset_list[i]}/base-{dataset_list[i]}',
        f'{dataset_list[i]}/base-{dataset_list[i]}-r',
        f'{dataset_list[i]}/base-{dataset_list[i]}-r-wreg',
        f'{dataset_list[i]}/base-{dataset_list[i]}-wreg',
        f'{dataset_list[i+1]}/base-{dataset_list[i+1]}',
        f'{dataset_list[i+1]}/base-{dataset_list[i+1]}-r',
        f'{dataset_list[i+1]}/base-{dataset_list[i+1]}-r-wreg',
        f'{dataset_list[i+1]}/base-{dataset_list[i+1]}-wreg',
    ]


    configs = [
        f'--config configs/vrig_iphone_dataset/{dataset_list[i]}.py',
        f'--config configs/vrig_iphone_dataset/{dataset_list[i]}.py --rndm_bck 1',
        f'--config configs/vrig_iphone_dataset/{dataset_list[i]}.py --wreg 1 --rndm_bck 1 --interp 1 --kernel linear --smoothing 1.0',
        f'--config configs/vrig_iphone_dataset/{dataset_list[i]}.py --wreg 1 --interp 0 --kernel linear --smoothing 1.0',
        f'--config configs/vrig_iphone_dataset/{dataset_list[i+1]}.py',
        f'--config configs/vrig_iphone_dataset/{dataset_list[i+1]}.py --rndm_bck 1',
        f'--config configs/vrig_iphone_dataset/{dataset_list[i+1]}.py --wreg 1 --rndm_bck 1 --interp 1 --kernel linear --smoothing 1.0',
        f'--config configs/vrig_iphone_dataset/{dataset_list[i+1]}.py --wreg 1 --interp 0 --kernel linear --smoothing 1.0',
    ]

    print(f'Training {dataset_list[i]} and {dataset_list[i+1]}')
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





















