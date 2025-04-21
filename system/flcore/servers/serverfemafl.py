import time
from flcore.clients.clientfemafl import clientFEMAFL
from flcore.servers.serverbase import Server
from threading import Thread
import numpy as np
import copy
import os


class FedFEMAFL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFEMAFL)

        self.logger.write(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        self.logger.write("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        
        # RL相关参数 - 从args读取
        self.current_round = 0
        self.prev_accuracy = 0
        self.w1 = args.accuracy_weight  # 准确性权重
        self.w_eff = args.efficiency_weight  # 效率权重
        self.w5 = args.fairness_weight  # 公平性权重
        self.alpha = args.variance_weight  # 方差权重
        
        # 记录每个客户端的参与次数
        self.client_participation_counts = [0] * self.num_clients
        
        # 记录每轮的时间开销
        self.round_time_costs = []
        self.round_time_variances = []
        
        # 记录最后一轮的客户端参与情况
        self.last_round_participants = []

        # 记录当前轮次每个客户端的时间开销
        self.cur_global_round_client_time_cost = []
        # 记录上一轮每个客户端的时间开销
        self.last_global_round_client_time_cost = []

        # 记录每轮全局奖励
        self.global_reward_history = []

    def compute_rewards(self, global_accuracy):
        """计算每个客户端的奖励"""
        # 计算准确性提升
        accuracy_delta = global_accuracy - self.prev_accuracy
        self.prev_accuracy = global_accuracy
        
        # 计算时间延迟
        if len(self.global_round_time_cost) > 0:
            total_time = self.global_round_time_cost[-1]
            # 修复方差计算代码
            client_times = []
            for c in self.selected_clients:
                if len(c.train_time_cost['cur_train_time_cost']) > 0 and len(c.send_time_cost['cur_send_time_cost']) > 0:
                    client_time = c.train_time_cost['cur_train_time_cost'][-1] + c.send_time_cost['cur_send_time_cost'][-1]
                    client_times.append(client_time)
            time_variance = np.var(client_times) if client_times else 0
        else:
            total_time = 0
            time_variance = 0
        
        # 记录时间统计
        self.round_time_costs.append(total_time)
        self.round_time_variances.append(time_variance)
        
        # 计算客户端参与率
        min_participation_rate = min([count/max(1, self.current_round) for count in self.client_participation_counts])
        
        # 计算总体奖励
        total_reward = self.w1 * accuracy_delta - self.w_eff * (total_time + self.alpha * time_variance) + self.w5 * min_participation_rate
        
        # 分配给各个客户端
        for client in self.clients:
            # 为参与本轮的客户端计算奖励
            if client.id in self.last_round_participants:
                # 计算客户端时间与平均时间的相对比值
                client_time = client.train_time_cost['cur_train_time_cost'][-1] + client.send_time_cost['cur_send_time_cost'][-1] \
                                if len(client.train_time_cost['cur_train_time_cost']) > 0 and len(client.send_time_cost['cur_send_time_cost']) > 0 else 0
                relative_time = (total_time - client_time) / max(0.1, total_time)  # 越小越好
                
                # 为每个客户端分配奖励
                client_reward = total_reward + 0.2 * relative_time  # 给快速客户端额外奖励
                
                # 更新客户端智能体
                client.receive_reward(client_reward, global_accuracy, self.current_round == self.global_rounds)
                
                self.logger.write(f"Client {client.id} reward: {client_reward:.4f}")
            else:
                # 未参与客户端接收一个小的负奖励
                client.receive_reward(-0.1, global_accuracy, self.current_round == self.global_rounds)
        
        return total_reward

    def save_rl_agents(self):
        """保存所有客户端的RL agent模型"""
        agent_folder = os.path.join(self.save_folder_name, 'rl_agents')
        if not os.path.exists(agent_folder):
            os.makedirs(agent_folder)
        
        for client in self.clients:
            agent_path = os.path.join(agent_folder, f'agent_client_{client.id}.pt')
            client.agent.save_model(agent_path)
        
        self.logger.write(f"Saved RL agent models to {agent_folder}")

    def load_rl_agents(self, agent_folder=None):
        """加载所有客户端的RL agent模型
        
        参数:
            agent_folder: 包含RL模型的文件夹路径，如果为None则使用默认路径
        """
        if agent_folder is None:
            agent_folder = os.path.join(self.save_folder_name, 'rl_agents')
        
        if not os.path.exists(agent_folder):
            self.logger.write(f"Warning: RL agent folder {agent_folder} does not exist. Using random initialization.")
            return False
        
        success_count = 0
        for client in self.clients:
            agent_path = os.path.join(agent_folder, f'agent_client_{client.id}.pt')
            if client.agent.load_model(agent_path):
                success_count += 1
        
        self.logger.write(f"Successfully loaded {success_count}/{len(self.clients)} RL agent models from {agent_folder}")
        return success_count > 0

    def train(self):
        for i in range(1, 1 + self.global_rounds):
            self.current_round = i
            
            # 在每轮开始时应用资源波动
            self.apply_resource_fluctuation(i)
            
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            
            if i%self.eval_gap == 0:
                self.logger.write(f"\n-------------Round number: {i}-------------")
                self.logger.write("\nEvaluate global model")
                self.evaluate()
            
            clients_time_cost = []
            clients_train_time = []
            clients_trans_time = []
            self.last_round_participants = []
            self.last_global_round_client_time_cost = copy.deepcopy(self.cur_global_round_client_time_cost)
            self.cur_global_round_client_time_cost = []
            for client in self.selected_clients:
                # 传入当前轮次，让客户端能够根据轮次适应策略
                client_train_time, client_trans_time, client_time_cost = client.train(round_number=i)
                
                # 只有实际训练了的客户端才算参与
                if client_time_cost > 0:
                    self.last_round_participants.append(client.id)
                    self.client_participation_counts[client.id] += 1
                    clients_time_cost.append(client_time_cost)
                    clients_train_time.append(client_train_time)
                    clients_trans_time.append(client_trans_time)
                    self.cur_global_round_client_time_cost.append(client_time_cost)
                    self.all_clients_train_time.append(client_train_time)
                    self.all_clients_trans_time.append(client_trans_time)

            
            self.receive_models()
            self.aggregate_parameters()
            
            if len(clients_time_cost) > 0:
                self.global_round_time_cost.append(np.max(clients_time_cost))
                self.logger.write(f"The current global round takes {self.global_round_time_cost[-1]} seconds")
            else:
                self.global_round_time_cost.append(0)
                self.logger.write("No clients participated in this round")
            
            # 计算并分发奖励
            if i % self.eval_gap == 0 and len(self.rs_test_acc) > 0:
                current_accuracy = self.rs_test_acc[-1]
                total_reward = self.compute_rewards(current_accuracy)
                self.logger.write(f"Total reward for round {i}: {total_reward:.4f}")
                self.global_reward_history.append(total_reward)
                # 记录客户端计算能力和传输能力
                self.logger.write("\nClient capabilities and statistics:")
                for client in self.selected_clients:
                    self.logger.write(f"Client {client.id}: Compute Power = {client.compute_power:.2f}x, " + 
                                  f"Transmission Power = {client.transmission_power:.2f} Mbps, " +
                                  f"Participation count = {self.client_participation_counts[client.id]}, " +
                                  f"Last reward = {client.last_reward:.4f}, " + 
                                  f"Straggler ratio = {np.mean(client.straggler_history) if client.straggler_history else 0:.2f}")
            
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
        
        # 训练结束统计
        self.logger.write("Best accuracy: {}".format(max(self.rs_test_acc)))
        
        # 计算公平性指标
        fairness_index = min(self.client_participation_counts) / max(max(self.client_participation_counts), 1)
        self.logger.write(f"Fairness index (min/max participation): {fairness_index:.4f}")
        
        # 效率统计
        self.logger.write(f"Average round time cost: {np.mean(self.round_time_costs):.4f}")
        self.logger.write(f"Average time variance: {np.mean(self.round_time_variances):.4f}")
        self.logger.write(f"Averaged Global Round Trans Time Cost: {np.mean(self.all_clients_trans_time)}")
        self.logger.write(f"Averaged Global Round Train Time Cost: {np.mean(self.all_clients_train_time)}")
        self.logger.write(f"Averaged Global Round Time Cost: {np.mean(self.global_round_time_cost)}")

        # 打印全局奖励历史
        self.logger.write(f"Global reward history: {self.global_reward_history}")
        
        # 客户端统计
        participation_rates = {}
        for i, count in enumerate(self.client_participation_counts):
            rate = count / self.global_rounds
            participation_rates[i] = rate
            
        # 按参与率排序
        sorted_rates = sorted(participation_rates.items(), key=lambda x: x[1], reverse=True)
        self.logger.write("\nTop 10 most active clients:")
        for i, (client_id, rate) in enumerate(sorted_rates[:10]):
            self.logger.write(f"#{i+1}: Client {client_id}, participation rate: {rate:.4f}")
        
        self.logger.write("\nBottom 10 least active clients:")
        for i, (client_id, rate) in enumerate(sorted_rates[-10:]):
            self.logger.write(f"#{len(sorted_rates)-9+i}: Client {client_id}, participation rate: {rate:.4f}")
        
        # 保存RL agent模型
        self.save_rl_agents()
        
        self.save_results()
        self.save_global_model()

