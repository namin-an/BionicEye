import os
import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import torch

import sys
sys.path.append(os.getcwd())
from training import Experiment

@hydra.main(config_path = 'conf', config_name='config.yaml')
def main(cfg: DictConfig):
    # Working directory check
    working_dir = os.getcwd()
    print(f"The current working directory: {working_dir}")
    orig_cwd = hydra.utils.get_original_cwd()
    print(f"The original cwd: {orig_cwd}")

    # Configuration check
    assert cfg.environment.name in ['Bioniceye', 'CartPole-v1'], "Invalid environment!"
    assert cfg.algorithm.name in ['PPO', 'REINFORCE', 'AC', 'DQN'], "Invalid algorithm!"

    # General setup
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Define variables
    image_dir = f"{orig_cwd}\{cfg.directory.image_dir}"
    label_path = f"{orig_cwd}\{cfg.directory.label_path}"
    pred_dir = f"{orig_cwd}\{cfg.directory.pred_dir}"
    stim_type = cfg.data.stim_type
    top1 = cfg.data.top1

    data_path = f"{orig_cwd}\{cfg.directory.data_path}"
    class_num = cfg.data.class_num

    env_type = cfg.environment.name
    model_type = cfg.algorithm.name
    episode_num = cfg.training.episode_num
    print_interval = cfg.training.print_interval
    learning_rate = cfg.reinforcement.learning_rate
    gamma = cfg.reinforcement.gamma
    batch_size = cfg.training.batch_size
    render = cfg.training.render

    # Algorithm specificity
    if model_type == 'PPO':
        lmbda = cfg.algorithm.lmbda
        eps_clip = cfg.algorithm.eps_clip
        argv = [lmbda, eps_clip]
    else:
        argv = []

    # GPU check
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Current cuda device: {torch.cuda.current_device()}")

    # Perform experiment
    exp = Experiment(image_dir, label_path, pred_dir, stim_type, top1, data_path, class_num, env_type, model_type, episode_num, print_interval, learning_rate, gamma, batch_size, render, device, argv)
    model, train_returns, time_memory = exp.train()

    # Save the information
    if cfg.save_info:
        os.makedirs(f"{cfg.directory.output_dir}")
        torch.save(model.state_dict(), os.path.join(cfg.directory.output_dir, f'{env_type}_{model_type}.pth'))
        train_returns_df = pd.DataFrame.from_dict(dict([(k, pd.Series(v, dtype='float64')) for k, v in train_returns.items()]))
        train_returns_df.to_csv(os.path.join(cfg.directory.output_dir, f'TrainReturns_{env_type}_{model_type}.csv'))
        time_memory_df = pd.DataFrame.from_dict(dict([(k, pd.Series(v, dtype='float64')) for k, v in time_memory.items()]))
        time_memory_df.to_csv(os.path.join(cfg.directory.output_dir, f'TimeMemory_{env_type}_{model_type}.csv'))

if __name__ == '__main__':
    main()