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
        """从经验回放中学习"""
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
        
        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_network(next_states)
            max_next_q = torch.max(next_q, dim=1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
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


class FEMAAction:
    """FEMA动作类，封装所有可能的动作"""
    def __init__(self, participate: bool, quantization_idx: int, epochs: int):
        self.participate = participate  # True表示参与，False表示不参与
        self.quantization_idx = quantization_idx  # 量化精度选项
        self.epochs = epochs  # 训练轮次

class FEMAState:
    """FEMA状态类，封装所有状态信息"""
    def __init__(self, train_samples: int, avg_compute: float, current_compute: float,
                 avg_transmission: float, current_transmission: float,
                 participation_count: int, avg_straggler: float):
        self.train_samples = train_samples
        self.avg_compute = avg_compute
        self.current_compute = current_compute
        self.avg_transmission = avg_transmission
        self.current_transmission = current_transmission
        self.participation_count = participation_count
        self.avg_straggler = avg_straggler

class FEMAActionSpace:
    """FEMA动作空间类，管理动作空间的定义和转换"""
    def __init__(self, quantization_options: list, max_epochs: int):
        self.quantization_options = quantization_options
        self.max_epochs = max_epochs
        
    def get_action_bounds(self):
        """获取动作空间的边界"""
        return [
            [0, 1],  # 是否参与 (0-1)
            [0, len(self.quantization_options)-1],  # 量化精度选项
            [0, self.max_epochs-1]  # epoch数
        ]
        
    def action_to_indices(self, action: FEMAAction) -> list:
        """将FEMA动作转换为动作索引"""
        return [
            1 if action.participate else 0,
            action.quantization_idx,
            action.epochs - 1  # 转换为0-based索引
        ]
        
    def indices_to_action(self, indices: list) -> FEMAAction:
        """将动作索引转换为FEMA动作"""
        return FEMAAction(
            participate=indices[0] == 1,
            quantization_idx=indices[1],
            epochs=indices[2] + 1  # 转换为1-based值
        )

class FEMARewardCalculator:
    """FEMA奖励计算器，负责计算和分配奖励"""
    def __init__(self, accuracy_weight: float = 0.7, time_weight: float = 0.3):
        self.accuracy_weight = accuracy_weight
        self.time_weight = time_weight
        
    def calculate_reward(self, accuracy_delta: float, time_cost: float, 
                        is_straggler: bool) -> float:
        """计算奖励值"""
        accuracy_reward = accuracy_delta * self.accuracy_weight
        time_penalty = -time_cost * self.time_weight
        straggler_penalty = -1.0 if is_straggler else 0.0
        
        return accuracy_reward + time_penalty + straggler_penalty

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

class clientFEMAFL(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # 初始化组件
        self.action_space = FEMAActionSpace(
            args.quantization_options,
            args.max_epochs
        )
        self.reward_calculator = FEMARewardCalculator()
        
        # RL相关参数
        self.participation_count = 0
        self.straggler_history = []
        self.computation_history = []
        self.transmission_history = []
        self.current_round = 0
        
        # 创建RL agent
        self.agent = RLAgent(
            state_dim=self.get_state_dim(),
            action_dims=[2, len(args.quantization_options), args.max_epochs],
            lr=args.rl_learning_rate,
            gamma=args.discount_factor,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            memory_size=args.memory_size
        )
        
        # 状态和动作记录
        self.current_state = None
        self.current_action = None
        self.next_state = None
        self.prev_accuracy = 0
        
    def get_state_dim(self) -> int:
        """获取状态维度"""
        return 7  # 基础状态：数据量、计算能力(平均/当前)、传输能力(平均/当前)、参与次数、掉队率
        
    def get_state(self, round_number: int) -> np.ndarray:
        """获取当前状态"""
        avg_compute = np.mean(self.computation_history[max(0, round_number-10):round_number]) if self.computation_history else self.compute_power
        avg_transmission = np.mean(self.transmission_history[max(0, round_number-10):round_number]) if self.transmission_history else self.transmission_power
        avg_straggler = np.mean(self.straggler_history[max(0, round_number-10):round_number]) if self.straggler_history else 0
        
        # 归一化状态
        state = [
            self.train_samples / 10000.0,  # 假设最大数据量为10000
            avg_compute / 100.0,           # 假设最大计算能力为100
            self.compute_power / 100.0,
            avg_transmission / 100.0,      # 假设最大传输能力为100
            self.transmission_power / 100.0,
            self.participation_count / 100.0,  # 假设最大参与次数为100
            avg_straggler                  # 已经是0-1之间的值
        ]
        
        return np.array(state, dtype=np.float32)
        
    def execute_action(self, action_indices) -> bool:
        """执行动作"""
        # 解析动作
        participate = action_indices[0] == 1
        quantization_idx = action_indices[1]
        epochs = action_indices[2] + 1  # 转为1-based
        
        if not participate:
            return False
            
        # 设置训练参数
        self.quantization_bits = self.action_space.quantization_options[quantization_idx]
        self.local_epochs = epochs
        
        self.participation_count += 1
        return True
        
    def train(self, round_number=0, max_local_epochs=None):
        """训练过程"""
        self.current_round = round_number
        
        # 获取当前状态
        state = self.get_state(round_number)
        self.current_state = state
        
        # 选择动作
        action_indices = self.agent.choose_action(state)
        self.current_action = action_indices
        
        # 执行动作
        if not self.execute_action(action_indices):
            self.next_state = self.get_state(round_number)
            return 0, 0, 0
            
        # 执行训练
        train_time_cost, transmission_time_cost, total_time_cost = self._execute_training(max_local_epochs)
        
        # 更新状态
        self.next_state = self.get_state(round_number)
        
        return train_time_cost, transmission_time_cost, total_time_cost
        
    def _execute_training(self, max_local_epochs=None):
        """执行实际的训练过程"""
        # 加载训练数据
        trainloader = self.load_train_data()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        
        # 训练模型
        start_time = time.time()
        self.model.train()
        
        if max_local_epochs is None:
            max_local_epochs = self.local_epochs
            
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
        # 计算时间成本
        train_time_cost = (time.time() - start_time) / self.compute_power
        self.apply_quantization_to_model(self.model, self.quantization_bits)
        transmission_time_cost = self.get_model_size() / (self.transmission_power * 1024 * 1024 / 8 * (32.0 / self.quantization_bits))
        
        return train_time_cost, transmission_time_cost, train_time_cost + transmission_time_cost
        
    def receive_reward(self, reward: float, accuracy: float, is_last_round: bool):
        """接收奖励并更新agent"""
        # 计算准确率改进
        accuracy_delta = accuracy - self.prev_accuracy
        self.prev_accuracy = accuracy
        
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