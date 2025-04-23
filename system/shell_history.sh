# FedAvg
python -u main.py -ab 1 -jr 1 -lbs 32 -ls 1 -nc 20 -nb 10 -data Cifar10 -did 0 -gr 100 -algo FedAvg --model ResNet18 --save_folder_name results

# FEMAFL
# Train agents with double dqn
python -u main.py -ab 1 -jr 1 -lbs 32 -ls 1 -nc 20 -nb 10 -data Cifar10 -did 1 -gr 100 -algo FEMAFL --model ResNet18 --save_folder_name results/ddqn

python -u main.py -ab 1 -jr 1 -lbs 32 -ls 1 -nc 20 -nb 10 -data Cifar10 -did 1 -gr 200 -algo FEMAFL --model ResNet18 --save_folder_name results/ddqn_200epochs

# sac
python -u main.py -ab 1 -jr 1 -lbs 32 -ls 1 -nc 20 -nb 10 -data Cifar10 -did 0 -gr 200 -algo FEMAFL --model ResNet18 --save_folder_name results/sac

# Test
python -u main.py -ab 1 -jr 1 -lbs 32 -ls 1 -nc 20 -nb 10 -data Cifar10 -did 1 -gr 100 -algo FEMAFL --model ResNet18 --save_folder_name results -load_rl results/FEMAFL_Cifar10_32_10/rl_agents



