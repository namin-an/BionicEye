import os
import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import torch
import time
import tracemalloc

import sys
sys.path.append(os.getcwd())
from training import ExperimentRL, ExperimentSL

@hydra.main(config_path = 'conf', config_name='config.yaml')
def main(cfg: DictConfig):
    # Print working directory
    working_dir = os.getcwd()
    print(f"The current working directory: {working_dir}")
    orig_cwd = hydra.utils.get_original_cwd()
    print(f"The original cwd: {orig_cwd}")

    # Check configurations
    assert cfg.environment.name in ['Bioniceye', 'CartPole-v1'], "Invalid environment!"
    assert cfg.algorithm.name in ['PPO', 'REINFORCE', 'AC', 'DQN'], "Invalid algorithm!"

    # Generate seeds
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Check GPUs
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Current cuda device: {torch.cuda.current_device()}")

    if cfg.train == 'RL':
        # Define local variables
        image_dir = f"{orig_cwd}\{cfg.reinforcement.directory.image_dir}"
        label_path = f"{orig_cwd}\{cfg.reinforcement.directory.label_path}"
        pred_dir = f"{orig_cwd}\{cfg.reinforcement.directory.pred_dir}"
        data_path = f"{orig_cwd}\{cfg.reinforcement.directory.data_path}"

        stim_type = cfg.reinforcement.data.stim_type
        top1 = cfg.reinforcement.data.top1
        class_num = cfg.reinforcement.data.class_num

        env_type = cfg.environment.name
        model_type = cfg.algorithm.name
        episode_num = cfg.environment.episode_num
        print_interval = cfg.reinforcement.training.print_interval
        learning_rate = cfg.reinforcement.training.learning_rate
        gamma = cfg.reinforcement.training.gamma
        batch_size = cfg.environment.batch_size
        render = cfg.reinforcement.training.render

        # Specify algorithm
        if model_type == 'PPO':
            lmbda = cfg.algorithm.lmbda
            eps_clip = cfg.algorithm.eps_clip
            argv = [lmbda, eps_clip]
        else:
            argv = []

        # Perform experiment
        exp = ExperimentRL(image_dir, label_path, pred_dir, stim_type, top1, data_path, class_num, env_type, model_type, episode_num, print_interval, learning_rate, gamma, batch_size, render, device, argv)
        if cfg.monitor_tm:
            start_time = time.time()
            tracemalloc.start()
            model, train_returns = exp.train()
            memory = tracemalloc.get_traced_memory()
            print(f"Total training time: {time.time() - start_time : .2f} (seconds) or  {(time.time() - start_time) / 60 : .2f} (minutes)")
            print(f"Peak Memory: {memory[-1]} [byte] or {memory[-1] / (1e6) : .2f} [mb]")
            tracemalloc.stop()
        else:
            model, train_returns = exp.train()

        # Save the information
        if cfg.save_info:
            os.makedirs(f"{cfg.output_dir}")
            torch.save(model.state_dict(), os.path.join(cfg.output_dir, f'{env_type}_{model_type}.pth'))
            train_returns_df = pd.DataFrame.from_dict(dict([(k, pd.Series(v, dtype='float64')) for k, v in train_returns.items()]))
            train_returns_df.to_csv(os.path.join(cfg.output_dir, f'TrainReturns_{env_type}_{model_type}.csv'))

    if cfg.train == 'SL':
        # Define local variables
        image_dir = f"{orig_cwd}\{cfg.supervised.directory.image_dir}"
        data_path = f"{orig_cwd}\{cfg.supervised.directory.data_path}"

        class_num = cfg.supervised.data.class_num

        epoch_num = cfg.supervised.training.epoch_num
        print_interval = cfg.supervised.training.print_interval
        learning_rate = cfg.supervised.training.learning_rate
        batch_size = cfg.supervised.training.batch_size
        seed = cfg.supervised.training.seed

        os.makedirs(f"{cfg.output_dir}")
        model_file_path = os.path.join(cfg.output_dir, f'CNN.pth')

        # Perform experiment
        exp = ExperimentSL(image_dir, data_path, class_num, epoch_num, print_interval, learning_rate, batch_size, seed, model_file_path, device)
        if cfg.monitor_tm:
            start_time = time.time()
            tracemalloc.start()
            exp.train()
            memory = tracemalloc.get_traced_memory()
            print(f"Total training time: {time.time() - start_time : .2f} (seconds) or  {(time.time() - start_time) / 60 : .2f} (minutes)")
            print(f"Peak Memory: {memory[-1]} [byte] or {memory[-1] / (1e6) : .2f} [mb]")
            tracemalloc.stop()
        else:
            exp.train()

if __name__ == '__main__':
    main()