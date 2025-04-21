import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client


class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

    def train(self, max_local_epochs=None):
        trainloader = self.load_train_data()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.model.train()
        
        start_time = time.time()

        if max_local_epochs == None:
            max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

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
        
        # 统计当前round客户端训练的gpu时间，以及模拟的传输时间
        cur_round_training_time_cost = (time.time() - start_time) / self.compute_power # 根据计算能力调整训练时间
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += cur_round_training_time_cost
        self.train_time_cost['cur_train_time_cost'].append(cur_round_training_time_cost)

        # get_model_size() 获取模型大小单位为字节，transmission_power单位为Mbps，所以需要转换为字节/s
        cur_round_transmission_time_cost = self.get_model_size() / (self.transmission_power * 1024 * 1024 / 8)
        self.send_time_cost['num_rounds'] += 1
        self.send_time_cost['total_cost'] += cur_round_transmission_time_cost
        self.send_time_cost['cur_send_time_cost'].append(cur_round_transmission_time_cost)
        # total time cost
        cur_round_time_total_cost = cur_round_training_time_cost + cur_round_transmission_time_cost
        
        self.train_time_cost['total_cost'] += time.time() - start_time
        self.logger.write(f"Client {self.id} train time cost: {cur_round_training_time_cost}, transmission time cost: {cur_round_transmission_time_cost}, total time cost: {cur_round_time_total_cost}")
        return cur_round_training_time_cost, cur_round_transmission_time_cost, cur_round_time_total_cost