import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client
import random
from collections import deque
import os


class RLAgent:
    def __init__(self, state_dim, action_dims, lr=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, memory_size=2000):
        """
        强化学习代理，使用DQN算法
        state_dim: 状态空间维度
        action_dims: 每个动作的可能取值数量的列表 [是否参与, batch_size选项数, 层数选项数, 量化精度选项数, epoch数选项数]
        """
        self.state_dim = state_dim
        self.action_dims = action_dims
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr
        
        # 创建Q网络
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.update_target_network()

        # 使用哪种RL算法
        self.use_double_dqn = True  # 启用双重DQN
        
    def _build_model(self):
        """创建神经网络模型"""
        model = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, np.prod(self.action_dims))
        )
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        return model
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action_indices, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action_indices, reward, next_state, done))
    
    def choose_action(self, state):
        """选择动作"""
        if np.random.rand() <= self.epsilon:
            # 探索：随机选择动作
            action_indices = [np.random.randint(dim) for dim in self.action_dims]
            return action_indices
        
        # 利用：选择Q值最大的动作
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor).detach().numpy().reshape(self.action_dims)
        
        # 找到多维数组中最大值的索引
        flat_idx = np.argmax(q_values)
        action_indices = np.unravel_index(flat_idx, self.action_dims)
        
        return list(action_indices)
    
    def replay(self, batch_size=32):
        """从经验回放中学习，使用双重DQN"""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor([m[0] for m in minibatch])
        action_indices = [m[1] for m in minibatch]
        rewards = torch.FloatTensor([m[2] for m in minibatch])
        next_states = torch.FloatTensor([m[3] for m in minibatch])
        dones = torch.FloatTensor([m[4] for m in minibatch])
        
        # 计算当前Q值
        current_q = self.q_network(states)
        # 将多维动作索引转换为一维索引
        flat_indices = [np.ravel_multi_index(idx, self.action_dims) for idx in action_indices]
        current_q_selected = current_q[range(batch_size), flat_indices]
        
        if self.use_double_dqn:
            # 双重DQN：使用当前网络选择动作，使用目标网络评估动作
            next_q_current = self.q_network(next_states).reshape(batch_size, -1)
            next_actions = torch.argmax(next_q_current, dim=1)
            next_q_target = self.target_network(next_states).reshape(batch_size, -1)
            next_q_selected = next_q_target[range(batch_size), next_actions]
            target_q = rewards + self.gamma * (1 - dones) * next_q_selected
        else:
            # 原始DQN
            target_q = rewards + self.gamma * (1 - dones) * torch.max(
                self.target_network(next_states).reshape(batch_size, -1), dim=1)[0]
        
        # 计算损失
        loss = torch.nn.MSELoss()(current_q_selected, target_q)
        
        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, path):
        """保存模型到指定路径"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load_model(self, path):
        """从指定路径加载模型"""
        if not os.path.exists(path):
            return False
        
        try:
            checkpoint = torch.load(path)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.epsilon = checkpoint['epsilon']
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


class clientFEMAFL(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # RL相关参数
        self.participation_count = 0  # 参与训练的次数
        self.straggler_history = []  # 历史straggler记录 (1表示是straggler, 0表示不是)
        self.computation_history = []  # 历史计算能力记录
        self.transmission_history = []  # 历史传输能力记录
        self.global_rounds = args.global_rounds  # 保存全局轮次数
        self.current_round = 0  # 添加当前轮次属性
        
        # 可选的batchsize列表 - 从args读取
        self.batch_size_options = args.batch_size_options
        # 可选的量化精度列表 - 从args读取
        self.quantization_options = args.quantization_options
        
        # 分析模型结构，获取可训练层
        self.trainable_layers = self.get_trainable_layers()
        self.layer_types = self.get_layer_types()
        self.layer_params_count = self.get_layer_params_count()
        
        # 状态维度：本地数据量，历史计算能力，当前计算负载，模型层数，历史传输能力，当前网络负载，参与次数，历史straggler记录，当前round
        self.state_dim = 9 + len(self.trainable_layers)  # 增加每层参数量信息作为状态
        
        # 动作维度：是否参与(2)，batchsize(选项数)，每层是否冻结(二进制向量)，量化精度(选项数)，epoch数(1到最大epoch数)
        self.action_dims = [2, len(self.batch_size_options), 2**len(self.trainable_layers), len(self.quantization_options), args.max_epochs]
        
        # 将离散动作空间转换为连续动作空间的边界
        action_bounds = []
        # 是否参与的边界 (0-1)
        action_bounds.append([0, 1])
        # batch_size选项的边界 (0到len(self.batch_size_options)-1)
        action_bounds.append([0, len(self.batch_size_options)-1])
        # 层冻结模式的边界 (0到2^len(self.trainable_layers)-1)
        action_bounds.append([0, 2**len(self.trainable_layers)-1])
        # 量化精度选项的边界 (0到len(self.quantization_options)-1)
        action_bounds.append([0, len(self.quantization_options)-1])
        # epoch数的边界 (0到args.max_epochs-1)
        action_bounds.append([0, args.max_epochs-1])
        
        # 创建SAC agent
        self.agent = SACAgent(
            self.state_dim, 
            len(action_bounds),  # 动作维度
            action_bounds,       # 动作边界
            actor_lr=args.rl_learning_rate,
            critic_lr=args.rl_learning_rate * 1.5,  # critic学习率通常略高
            alpha_lr=args.rl_learning_rate * 0.5,   # alpha学习率通常略低
            gamma=args.discount_factor,
            tau=0.005,  # 软更新参数
            memory_size=args.memory_size,
            hidden_size=64  # 隐藏层大小
        )
        
        # 当前round的状态和动作
        self.current_state = None
        self.current_action = None
        self.last_reward = 0
        self.last_accuracy = 0
        
        # 记录上一个round的准确率，用于计算Δaccuracy
        self.prev_accuracy = 0
        
        # 存储上一个状态和动作后的状态
        self.next_state = None
        
        # 当前冻结的层
        self.frozen_layers_mask = [0] * len(self.trainable_layers)  # 0表示不冻结，1表示冻结
        
    def get_trainable_layers(self):
        """识别模型中的可训练层"""
        trainable_layers = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # 获取层的名称(去掉参数名)
                layer_name = name.split('.')[0] if '.' in name else name
                if layer_name not in trainable_layers:
                    trainable_layers.append(layer_name)
        return trainable_layers
    
    def get_layer_types(self):
        """获取每个可训练层的类型"""
        layer_types = {}
        for name, module in self.model.named_modules():
            if name in self.trainable_layers:
                layer_types[name] = type(module).__name__
        return layer_types
    
    def get_layer_params_count(self):
        """计算每个可训练层的参数量"""
        params_count = {}
        for name in self.trainable_layers:
            count = 0
            for param_name, param in self.model.named_parameters():
                if param_name.startswith(name) and param.requires_grad:
                    count += param.numel()
            params_count[name] = count
        return params_count
    
    def get_state(self, round_number):
        """获取当前状态"""
        # 计算历史平均值
        avg_compute = np.mean(self.computation_history[max(0, round_number-10):round_number]) if self.computation_history else self.compute_power
        avg_transmission = np.mean(self.transmission_history[max(0, round_number-10):round_number]) if self.transmission_history else self.transmission_power
        avg_straggler = np.mean(self.straggler_history[max(0, round_number-10):round_number]) if self.straggler_history else 0
        
        # 计算当前激活层数
        active_layers_count = len(self.trainable_layers) - sum(self.frozen_layers_mask)
        
        # 构建状态向量
        state = [
            self.train_samples,  # 本地数据量
            avg_compute,         # 历史平均计算能力
            self.compute_power,  # 当前计算能力
            len(self.trainable_layers),  # 模型层数
            avg_transmission,    # 历史平均传输能力
            self.transmission_power,  # 当前传输能力
            self.participation_count,  # 参与次数
            avg_straggler,      # 历史平均straggler率
            round_number        # 当前round
        ]
        
        # 添加每层的参数量信息
        for layer_name in self.trainable_layers:
            state.append(self.layer_params_count[layer_name])
            
        return np.array(state, dtype=np.float32)
    
    def execute_action(self, action_indices):
        """执行动作"""
        # 解析动作
        participate = action_indices[0]  # 是否参与
        batch_size_idx = action_indices[1]  # batch_size选项索引
        layer_freeze_idx = action_indices[2]  # 层冻结模式索引
        quantization_idx = action_indices[3]  # 量化精度索引
        epochs = action_indices[4] + 1  # epoch数（加1因为索引从0开始）
        
        # 设置batch_size
        self.batch_size = self.batch_size_options[batch_size_idx]
        
        # 设置层冻结模式
        self.frozen_layers_mask = self.index_to_binary_mask(layer_freeze_idx, len(self.trainable_layers))
        self.apply_layer_freezing()
        
        # 设置量化精度
        self.quantization_bits = self.quantization_options[quantization_idx]
        
        # 设置epoch数
        self.local_epochs = epochs
        
        return participate
    
    def index_to_binary_mask(self, index, length):
        """将整数索引转换为二进制掩码数组"""
        binary = bin(index)[2:].zfill(length)  # 转换为二进制字符串并填充到指定长度
        return [int(bit) for bit in binary]  # 转换为整数列表
    
    def apply_layer_freezing(self):
        """根据冻结掩码冻结特定的层"""
        # 重置所有层为可训练
        for name, param in self.model.named_parameters():
            param.requires_grad = True
        
        # 根据掩码冻结指定层
        for i, layer_name in enumerate(self.trainable_layers):
            if self.frozen_layers_mask[i] == 1:  # 1表示冻结
                for name, param in self.model.named_parameters():
                    if name.startswith(layer_name):
                        param.requires_grad = False
    
    def get_active_parameters(self):
        """获取当前活跃的(未冻结的)参数"""
        return [param for name, param in self.model.named_parameters() if param.requires_grad]
    
    def update_statistics(self, is_straggler, train_time, transmission_time):
        """更新客户端统计信息"""
        self.straggler_history.append(1 if is_straggler else 0)
        self.computation_history.append(1.0 / train_time if train_time > 0 else 0)  # 计算能力与训练时间成反比
        self.transmission_history.append(1.0 / transmission_time if transmission_time > 0 else 0)  # 传输能力与传输时间成反比
        
        # 保持历史记录不会过长
        if len(self.straggler_history) > 10:
            self.straggler_history = self.straggler_history[-10:]
        if len(self.computation_history) > 10:
            self.computation_history = self.computation_history[-10:]
        if len(self.transmission_history) > 10:
            self.transmission_history = self.transmission_history[-10:]
    
    def train(self, round_number=0, max_local_epochs=None):
        """使用RL agent选择训练配置并执行训练"""
        # 更新当前轮次
        self.current_round = round_number
        
        # 获取当前状态
        state = self.get_state(round_number)
        self.current_state = state
        
        # 选择动作 - 使用SACAgent的choose_action方法
        continuous_actions = self.agent.choose_action(state)
        
        # 将连续动作转换为离散动作索引
        action_indices = []
        for i, action in enumerate(continuous_actions):
            low, high = self.agent.action_bounds[i]
            # 将连续值映射到离散索引
            index = int(round((action - low) / (high - low) * (high - low)))
            # 确保索引在有效范围内
            index = max(0, min(index, int(high)))
            action_indices.append(index)
        
        self.current_action = action_indices
        
        # 执行动作
        participate = self.execute_action(action_indices)
        
        if not participate:
            self.next_state = self.get_state(round_number)
            return 0, 0, 0
        
        # 应用层冻结
        self.apply_layer_freezing()
        
        # 加载训练数据并训练(始终使用全精度)
        trainloader = self.load_train_data()
        active_params = self.get_active_parameters()
        self.optimizer = torch.optim.SGD(active_params, lr=self.learning_rate)
        
        self.model.train()
        start_time = time.time()
        
        if max_local_epochs is None:
            max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)
        
        # 执行训练
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        # 计算训练时间成本
        train_time_cost = (time.time() - start_time) / self.compute_power
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += train_time_cost
        self.train_time_cost['cur_train_time_cost'].append(train_time_cost)
        
        # 训练后直接应用量化，用于传输
        self.apply_quantization_to_model(self.model, self.quantization_bits)
        
        # 计算传输时间（考虑量化因子）
        quantization_factor = 32.0 / self.quantization_bits
        transmission_time_cost = self.get_model_size() / (self.transmission_power * 1024 * 1024 / 8 * quantization_factor)
        
        self.send_time_cost['num_rounds'] += 1
        self.send_time_cost['total_cost'] += transmission_time_cost
        self.send_time_cost['cur_send_time_cost'].append(transmission_time_cost)
        
        # 总时间成本
        total_time_cost = train_time_cost + transmission_time_cost
        
        # 更新统计信息
        if hasattr(self, 'last_global_round_client_time_cost') and len(self.last_global_round_client_time_cost) > 0:
            threshold_time = np.mean(self.last_global_round_client_time_cost)
        else:
            # 第一轮时，没有历史数据，设置一个很大的阈值
            threshold_time = np.inf
        is_straggler = total_time_cost > threshold_time
        self.update_statistics(is_straggler, train_time_cost, transmission_time_cost)
        
        # 记录next_state
        self.next_state = self.get_state(round_number)
        
        # 记录冻结的层信息
        frozen_layers = [self.trainable_layers[i] for i, frozen in enumerate(self.frozen_layers_mask) if frozen == 1]
        frozen_layers_str = ", ".join(frozen_layers) if frozen_layers else "None"
        
        self.logger.write(f"Client {self.id} train time cost: {train_time_cost}, transmission time cost: {transmission_time_cost}, " +
                       f"total time cost: {total_time_cost}, batch_size: {self.batch_size}, " +
                       f"frozen layers: {frozen_layers_str}, active layers: {len(self.trainable_layers) - len(frozen_layers)}, " +
                       f"epochs: {self.local_epochs}, quantization: {self.quantization_bits}")
        
        return train_time_cost, transmission_time_cost, total_time_cost
    
    def apply_quantization_to_model(self, model, bits):
        """
        应用模型量化，将指定模型的参数量化为指定的位数
        
        参数:
            model: 要量化的模型
            bits: 量化位数，8表示8位量化，16表示16位量化，32表示全精度
        """
        if bits == 32:
            # 32位是全精度，不做量化处理
            return
        
        for name, param in model.named_parameters():
            if param.requires_grad:  # 只量化可训练参数
                # 记录原始参数值的范围
                orig_min = param.data.min()
                orig_max = param.data.max()
                
                # 计算量化步长
                scale = (orig_max - orig_min) / (2**bits - 1)
                
                if scale > 0:  # 避免除以零
                    # 量化：将浮点值转换为有限范围内的整数
                    quantized = torch.round((param.data - orig_min) / scale)
                    # 将量化后的值转回浮点域
                    param.data = quantized * scale + orig_min
    
    def receive_reward(self, reward, accuracy, is_last_round=False):
        """接收奖励并更新agent"""
        # 计算Δaccuracy
        delta_accuracy = accuracy - self.prev_accuracy
        self.prev_accuracy = accuracy
        
        # 更新状态
        self.next_state = self.get_state(self.current_round)
        
        # 存储经验
        self.agent.remember(
            self.current_state,
            self.current_action,
            reward,
            self.next_state,
            is_last_round
        )
        
        # 更新agent
        self.agent.replay()
        
        # 更新当前状态
        self.current_state = self.next_state
        
        # 保存模型
        if is_last_round:
            self.agent.save_model(f'agent_model_{self.id}.pth')

class SACAgent:
    def __init__(self, state_dim, action_dim, action_bounds, actor_lr=0.0003, critic_lr=0.0003, alpha_lr=0.0003, 
                 gamma=0.99, tau=0.005, memory_size=1000000, hidden_size=256):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.gamma = gamma
        self.tau = tau
        self.memory = deque(maxlen=memory_size)
        
        # 初始化策略网络
        self.actor = PolicyNetwork(state_dim, action_dim, hidden_size)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # 初始化两个Q网络
        self.critic1 = QNetwork(state_dim, action_dim, hidden_size)
        self.critic2 = QNetwork(state_dim, action_dim, hidden_size)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)
        
        # 初始化目标网络
        self.target_critic1 = QNetwork(state_dim, action_dim, hidden_size)
        self.target_critic2 = QNetwork(state_dim, action_dim, hidden_size)
        self.update_target_networks()
        
        # 自动调整温度参数
        self.target_entropy = -torch.prod(torch.Tensor([action_dim])).item()
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        
    def update_target_networks(self):
        """软更新目标网络"""
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
    def choose_action(self, state, evaluate=False):
        """选择动作，考虑离散动作空间"""
        state = torch.FloatTensor(state).unsqueeze(0)
        if evaluate:
            action, _ = self.actor.sample(state, deterministic=True)
        else:
            action, _ = self.actor.sample(state)
            
        # 将连续动作转换为离散动作
        discrete_actions = []
        for i, (action_value, (low, high)) in enumerate(zip(action[0], self.action_bounds)):
            # 将连续动作映射到离散选项
            if i == 0:  # 是否参与
                discrete_actions.append(1 if action_value > 0.5 else 0)
            else:
                # 将连续动作映射到离散选项
                normalized_value = (action_value - low) / (high - low)
                num_options = high - low + 1
                discrete_actions.append(int(normalized_value * num_options))
                
        return discrete_actions
        
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
        
    def replay(self, batch_size=16):
        """从经验回放中学习"""
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([m[0] for m in minibatch])
        actions = torch.FloatTensor([m[1] for m in minibatch])
        rewards = torch.FloatTensor([m[2] for m in minibatch]).unsqueeze(1)
        next_states = torch.FloatTensor([m[3] for m in minibatch])
        dones = torch.FloatTensor([m[4] for m in minibatch]).unsqueeze(1)
        
        # 计算目标Q值
        with torch.no_grad():
            next_actions, next_log_pi = self.actor.sample(next_states)
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_pi
            target_q = rewards + (1 - dones) * self.gamma * target_q
            
        # 更新Q网络
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        critic1_loss = torch.nn.MSELoss()(current_q1, target_q)
        critic2_loss = torch.nn.MSELoss()(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # 更新策略网络
        actions, log_pi = self.actor.sample(states)
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_pi - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 更新温度参数
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp()
        
        # 更新目标网络
        self.update_target_networks()
        
    @property
    def alpha(self):
        return self.log_alpha.exp()
        
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'log_alpha': self.log_alpha,
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict()
        }, path)
        
    def load_model(self, path):
        """加载模型"""
        if not os.path.exists(path):
            return False
            
        try:
            checkpoint = torch.load(path)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic1.load_state_dict(checkpoint['critic1'])
            self.critic2.load_state_dict(checkpoint['critic2'])
            self.log_alpha = checkpoint['log_alpha']
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
            self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

class PolicyNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.mean = torch.nn.Linear(hidden_dim, action_dim)
        self.log_std = torch.nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std
        
    def sample(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        if deterministic:
            action = mean
        else:
            normal = torch.distributions.Normal(mean, std)
            x = normal.rsample()
            action = torch.tanh(x)
            
        log_prob = normal.log_prob(x)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob

class QNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)