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
    assert cfg.algorithm.name in ['PPO', 'REINFORCE', 'AC', 'DQN'], "Invalid algorithm!"

    # General setup
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Define variables
    image_dir = f"{orig_cwd}\{cfg.directory.image_dir}"
    label_path = f"{orig_cwd}\{cfg.directory.label_path}"
    pred_dir = f"{orig_cwd}\{cfg.directory.pred_dir}"
    stim_type = cfg.type.stim_type
    top1 = cfg.top1

    data_path = f"{orig_cwd}\{cfg.directory.data_path}"
    class_num = cfg.class_num

    check_time_usage = cfg.check_time_usage
    model_type = cfg.type.model_type
    episode_num = cfg.training.episode_num
    print_interval = cfg.training.print_interval
    learning_rate = cfg.reinforcement.learning_rate
    discount = cfg.reinforcement.discount
    average_window = cfg.evaluation.average_window

    # GPU check
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Current cuda device: {torch.cuda.current_device()}")

    # Experiment
    exp = Experiment(image_dir, label_path, pred_dir, stim_type, top1, data_path, class_num, device, check_time_usage, model_type, episode_num, print_interval, learning_rate, discount, average_window)
    model, train_returns, test_returns = exp.train()

    # Save the information
    if cfg.save_info:
        os.makedirs(f"{cfg.directory.output_dir}")
        torch.save(model.state_dict(), os.path.join(cfg.directory.output_dir, f'{model_type}.pth'))
        train_returns_df = pd.DataFrame.from_dict(train_returns)
        test_returns_df = pd.DataFrame.from_dict(test_returns)
        train_returns_df.to_csv(os.path.join(cfg.directory.output_dir, f'TrainReturns_{model_type}.csv'))
        test_returns_df.to_csv(os.path.join(cfg.directory.output_dir, f'TestReturns_{model_type}.csv'))

if __name__ == '__main__':
    main()