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
        
        # 创建RL agent - 从args读取超参数
        self.agent = RLAgent(
            self.state_dim, 
            self.action_dims, 
            lr=args.rl_learning_rate,
            gamma=args.discount_factor,
            epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay,
            epsilon_min=args.epsilon_min,
            memory_size=args.memory_size
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
        """获取当前状态，包括每层的参数量信息"""
        # 计算历史平均值
        avg_compute = np.mean(self.computation_history[max(0, round_number-10):round_number]) if self.computation_history else self.compute_power
        avg_transmission = np.mean(self.transmission_history[max(0, round_number-10):round_number]) if self.transmission_history else self.transmission_power
        avg_straggler = np.mean(self.straggler_history[max(0, round_number-10):round_number]) if self.straggler_history else 0
        
        # 计算当前激活层数
        active_layers_count = len(self.trainable_layers) - sum(self.frozen_layers_mask)
        
        base_state = [
            self.train_samples/10000,  # 归一化的本地数据量
            avg_compute,              # 历史计算能力
            self.compute_power,       # 当前计算负载
            active_layers_count/len(self.trainable_layers),  # 归一化的激活层数
            avg_transmission,         # 历史传输能力
            self.transmission_power,  # 当前网络负载
            self.participation_count/self.global_rounds,  # 归一化的参与次数
            avg_straggler,            # 历史straggler记录
            round_number/self.global_rounds  # 归一化的当前round
        ]
        
        # 添加每层参数量的归一化信息
        total_params = sum(self.layer_params_count.values())
        layer_info = [count/total_params for count in self.layer_params_count.values()]
        
        return base_state + layer_info
    
    def execute_action(self, action_indices):
        """执行选定的动作，包括选择性地冻结层"""
        # 解析动作索引
        participate = action_indices[0] == 1  # 0表示不参与，1表示参与
        batch_size_idx = action_indices[1]
        freeze_pattern_idx = action_indices[2]  # 层冻结模式的索引
        quantization_idx = action_indices[3]
        epoch_count = action_indices[4] + 1  # 从1开始，至少训练1个epoch
        
        # 设置训练参数
        if participate:
            self.batch_size = self.batch_size_options[batch_size_idx]
            self.quantization_bits = self.quantization_options[quantization_idx]
            
            # 将freeze_pattern_idx转换为二进制冻结模式
            freeze_pattern = self.index_to_binary_mask(freeze_pattern_idx, len(self.trainable_layers))
            
            # 确保至少有一个层是可训练的
            if sum(freeze_pattern) == len(freeze_pattern):  # 如果所有层都被冻结
                freeze_pattern[0] = 0  # 解除第一个层的冻结
            
            self.frozen_layers_mask = freeze_pattern
            
            self.local_epochs = epoch_count
            self.participation_count += 1
            return True
        else:
            return False
    
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
        # 获取当前状态
        state = self.get_state(round_number)
        self.current_state = state
        
        # 选择动作
        action_indices = self.agent.choose_action(state)
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
        """接收服务器计算的奖励，并更新RL agent"""
        if self.current_state is not None and self.current_action is not None and self.next_state is not None:
            # 计算准确率改进
            accuracy_delta = accuracy - self.prev_accuracy
            self.prev_accuracy = accuracy
            
            # 判断是否是最后一轮
            done = is_last_round  # 如果是最后一轮，则done为True
            
            self.agent.remember(self.current_state, self.current_action, reward, self.next_state, done)
            
            # 训练agent
            self.agent.replay()
            
            # 定期更新目标网络
            if self.participation_count % 5 == 0:
                self.agent.update_target_network()
            
            self.last_reward = reward
            self.last_accuracy = accuracy