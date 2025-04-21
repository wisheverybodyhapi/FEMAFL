#!/usr/bin/env python
import torch
import argparse
import os
import time
import warnings
import numpy as np
import logging
import random
import torchvision
from datetime import datetime

from utils.result_utils import average_data
from utils.mem_utils import MemReporter
from utils.seed_utils import setup_seed

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverfemafl import FedFEMAFL

warnings.simplefilter("ignore")

def run(args):

    time_list = []
    reporter = MemReporter()

    for i in range(args.prev, args.times):
        start = time.time()

        # Generate args.model
        if args.model == "ResNet18":
            args.model = torchvision.models.resnet18(weights=None, num_classes=args.num_classes).to(args.device)
        else:
            raise NotImplementedError
        

        # select algorithm
        if args.algorithm == "FedAvg":
            server = FedAvg(args, i)
        elif args.algorithm == "FEMAFL":
            server = FedFEMAFL(args, i)
            # 如果指定了预训练RL模型路径，则加载模型
            if args.load_rl_agents:
                server.load_rl_agents(args.load_rl_agents)
        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    

    # Global average
    acc_mean, acc_std = average_data(results_path=args.save_folder_name, times=args.times)
    server.logger.write("mean for best accuracy: {}".format(acc_mean))
    server.logger.write("std for best accuracy: {}".format(acc_std))

    server.logger.write(f"Total {args.times} trial done!!!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-sd', "--seed", type=int, default=0, help="random seed")
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="mnist")
    parser.add_argument('-m', "--model", type=str, default="CNN")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.01,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=100)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, 
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=2,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='temp')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument('-vs', "--vocab_size", type=int, default=98635)
    parser.add_argument('-ml', "--max_len", type=int, default=200)
    parser.add_argument('-abl', "--ablation", type=int, default=0)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    
    # 新增RL相关超参数
    parser.add_argument('-rl_lr', "--rl_learning_rate", type=float, default=0.001,
                        help="Learning rate for RL agent")
    parser.add_argument('-gamma', "--discount_factor", type=float, default=0.95,
                        help="Discount factor for future rewards in RL")
    parser.add_argument('-eps', "--epsilon", type=float, default=1.0,
                        help="Initial epsilon for exploration in RL")
    parser.add_argument('-eps_decay', "--epsilon_decay", type=float, default=0.995,
                        help="Decay rate for epsilon in RL")
    parser.add_argument('-eps_min', "--epsilon_min", type=float, default=0.01,
                        help="Minimum epsilon value in RL")
    parser.add_argument('-mem_size', "--memory_size", type=int, default=2000,
                        help="Size of experience replay buffer in RL")
    parser.add_argument('-load_rl', "--load_rl_agents", type=str, default=None,
                        help="Path to pretrained RL agents folder")
    
    # 奖励函数权重
    parser.add_argument('-w1', "--accuracy_weight", type=float, default=1.0,
                        help="Weight for accuracy improvement in reward function")
    parser.add_argument('-w_eff', "--efficiency_weight", type=float, default=0.5,
                        help="Weight for efficiency in reward function")
    parser.add_argument('-w5', "--fairness_weight", type=float, default=0.3,
                        help="Weight for fairness in reward function")
    parser.add_argument('-alpha', "--variance_weight", type=float, default=0.2,
                        help="Weight for time variance in reward function")
    
    # 系统配置参数
    parser.add_argument('-bs_opt', "--batch_size_options", nargs='+', type=int, default=[16, 32, 64],
                        help="Available batch size options for clients")
    parser.add_argument('-quant_opt', "--quantization_options", nargs='+', type=int, default=[8, 16, 32],
                        help="Available quantization precision options (bits)")
    parser.add_argument('-max_ep', "--max_epochs", type=int, default=5,
                        help="Maximum number of local epochs")

    # 资源波动相关参数
    parser.add_argument('-res_fluct', "--resource_fluctuation", action="store_true", default=True,
                        help="Enable resource fluctuation during training")
    parser.add_argument('-fluct_freq', "--fluctuation_frequency", type=int, default=10,
                        help="Frequency of resource fluctuation (in rounds)")
    parser.add_argument('-fluct_scale', "--fluctuation_scale", type=float, default=0.2,
                        help="Scale of resource fluctuation (0.2 means ±20%)")

    args = parser.parse_args()

    # fix random seed
    setup_seed(args.seed)

    if args.save_folder_name == 'temp':
        args.save_folder_name_full = f"{args.algorithm}_{args.dataset}_{args.batch_size}_{args.num_classes}_{time.time()}"
    else:
        folder_name = f"{args.algorithm}_{args.dataset}_{args.batch_size}_{args.num_classes}"
        args.save_folder_name_full = os.path.join(args.save_folder_name, folder_name)
    args.save_folder_name = args.save_folder_name_full
    args.model_folder_name = os.path.join(args.save_folder_name, 'model')

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    run(args)
