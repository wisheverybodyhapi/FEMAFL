import torch
import os
import numpy as np
import h5py
import copy
import time
import random
import shutil
from utils.data_utils import read_client_data
from flcore.clients.clientbase import load_item, save_item
from utils.logger import Logger


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.local_learning_rate = args.local_learning_rate
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.top_cnt = 100
        self.auto_break = args.auto_break
        self.role = 'Server'
        self.save_folder_name = args.save_folder_name
        self.model_folder_name = args.model_folder_name
        self.cur_time = times
        self.total_times = args.times

        self.logger = Logger(self.save_folder_name, "log.log")
        self.logger.write("=" * 50 + ' {}th trial '.format(self.cur_time) + "=" * 50)
        self.logger.write("=" * 50)
        for arg in vars(args):
            self.logger.write("{} = {}".format(arg, getattr(args, arg)))
        self.logger.write("=" * 50)

        self.logger.write(f"Model is {args.model}")
        self.logger.write("=" * 50)
        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        # 记录当前轮次的时间开销
        self.global_round_time_cost = []
        
        # 记录整个训练周期中客户端的计算和传输时间
        self.all_clients_train_time = []
        self.all_clients_trans_time = []

        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate
        
        # 初始化全局模型，使其常驻内存
        self.global_model = copy.deepcopy(args.model).to(self.device)

    def set_clients(self, clientObj):
        client_bindwith = self.simulate_trans_power(self.num_clients)
        client_compute_power = self.simulate_compute_power(self.num_clients)

        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            logger=self.logger,
                            transmission_power=client_bindwith[i],
                            compute_power=client_compute_power[i])
            self.clients.append(client)

    def simulate_trans_power(self, num_clients):
        # Define performance ratios
        high_perf_ratio = 0.3  # 30% high-performance clients
        mid_perf_ratio = 0.5   # 50% medium-performance clients
        low_perf_ratio = 0.2   # 20% low-performance clients

        # Calculate number of clients for each performance level
        num_high_perf = int(self.num_clients * high_perf_ratio)
        num_mid_perf = int(self.num_clients * mid_perf_ratio)
        num_low_perf = self.num_clients - num_high_perf - num_mid_perf  # Ensure total matches self.num_clients

        # Create and shuffle client performance list
        client_performances = ['high'] * num_high_perf + ['mid'] * num_mid_perf + ['low'] * num_low_perf
        random.shuffle(client_performances)

        # Define bandwidth ranges (Mbps)
        bandwidth_ranges = {
            'high': (50, 100),  # High-speed network, 30%
            'mid': (10, 50),    # Medium-speed network, 50%
            'low': (1, 10)      # Low-speed network, 20%
        }

        client_bindwith = []

        for i in range(num_clients):
            # Assign bandwidth based on performance level
            perf_level = client_performances[i]
            min_bw, max_bw = bandwidth_ranges[perf_level]
            bandwidth = random.uniform(min_bw, max_bw)
            client_bindwith.append(bandwidth)

        return client_bindwith

    def simulate_compute_power(self, num_clients):
        # 定义计算能力比例
        high_perf_ratio = 0.3  # 30% 高性能客户端
        mid_perf_ratio = 0.5   # 50% 中等性能客户端
        low_perf_ratio = 0.2   # 20% 低性能客户端

        # 计算每个性能级别的客户端数量
        num_high_perf = int(self.num_clients * high_perf_ratio)
        num_mid_perf = int(self.num_clients * mid_perf_ratio)
        num_low_perf = self.num_clients - num_high_perf - num_mid_perf  # 确保总数匹配self.num_clients

        # 创建并打乱客户端性能列表
        client_performances = ['high'] * num_high_perf + ['mid'] * num_mid_perf + ['low'] * num_low_perf
        random.shuffle(client_performances)

        # 定义计算能力范围 (FLOPS倍数，相对于基准计算能力)
        compute_ranges = {
            'high': (2.0, 4.0),  # 高性能设备
            'mid': (1.0, 2.0),   # 中等性能设备
            'low': (0.25, 0.75)  # 低性能设备
        }

        client_compute_power = []

        for i in range(num_clients):
            # 根据性能级别分配计算能力
            perf_level = client_performances[i]
            min_compute, max_compute = compute_ranges[perf_level]
            compute_power = random.uniform(min_compute, max_compute)
            client_compute_power.append(compute_power)

        return client_compute_power

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            # 从 [self.num_join_clients, self.num_clients] 中随机选择1个数
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_ids(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        tot_samples = 0
        for client in active_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = np.mean(client.train_time_cost['cur_train_time_cost']) \
                                    + np.mean(client.send_time_cost['cur_send_time_cost'])
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
        
    def aggregate_parameters(self):
        if len(self.uploaded_ids) == 0:
            self.logger.write("警告：本轮没有客户端上传模型，保持全局模型不变。")
            return  # 跳过聚合，保持全局模型不变

        # 使用常驻内存的全局模型，而不是从硬盘加载
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, cid in zip(self.uploaded_weights, self.uploaded_ids):
            client = self.clients[cid]
            # 直接使用客户端的模型，而不是从硬盘加载
            for server_param, client_param in zip(self.global_model.parameters(), client.model.parameters()):
                server_param.data += client_param.data.clone() * w

        
    def save_results(self):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)

        # save results
        if (len(self.rs_test_acc)):
            file_path = os.path.join(self.save_folder_name, 'results' + '_' + str(self.cur_time) + '.h5')
            self.logger.write("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
        
        # if 'temp' in self.save_folder_name:
        #     try:
        #         shutil.rmtree(self.save_folder_name)
        #         self.logger.write('Deleted.')
        #     except:
        #         self.logger.write('Already deleted.')

    # For global model
    def save_global_model(self):
        if 'temp' not in self.save_folder_name:
            if os.path.exists(self.model_folder_name) == False:
                os.makedirs(self.model_folder_name)
            save_item(self.global_model, self.role, 'global_model', self.model_folder_name)
            self.logger.write('finish saving global model of server')

    # For Heterogeneous Clients
    def save_models(self):
        # only save last trial model
        if (1 + self.cur_time) != self.total_times:
            return
        
        if 'temp' not in self.save_folder_name:
            if os.path.exists(self.model_folder_name) == False:
                os.makedirs(self.model_folder_name)
            # save models
            try:
                for client in self.clients:
                    save_item(client.model, client.role, 'model', self.model_folder_name)
                self.logger.write('finish saving models of clients')
                if hasattr(self, 'global_model'):
                    save_item(self.global_model, self.role, 'global_model', self.model_folder_name)
                    self.logger.write('finish saving global model of server')
                if hasattr(self, 'generative_model'): # FedGen
                    save_item(self.generative_model, self.role, 'generative_model', self.model_folder_name)
                    self.logger.write('finish saving generative model of server')
                if hasattr(self, 'PROTO'): # FedTGP, FedOrth
                    save_item(self.PROTO, self.role, 'PROTO', self.model_folder_name)
                    self.logger.write('finish saving PROTO of server')
                if hasattr(self, 'global_protos'): # FedProto
                    save_item(self.global_protos, self.role, 'global_protos', self.model_folder_name)
                    self.logger.write('finish saving global_protos of server')
            except Exception as e:
                self.logger.write(f"An error occurred: {str(e)}")

            
        else:
            print('temp dir, no need to save models')

    def test_metrics(self):        
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            self.logger.write(f'Client {c.id}: Acc: {ct*1.0/ns}, AUC: {auc}')
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)
            self.logger.write(f'Client {c.id}: Loss: {cl*1.0/ns}')

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        # stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        # train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        # if loss == None:
        #     self.rs_train_loss.append(train_loss)
        # else:
        #     loss.append(train_loss)

        # self.logger.write("Averaged Train Loss: {:.4f}".format(train_loss))
        self.logger.write("Averaged Test Accuracy: {:.4f}".format(test_acc))
        self.logger.write("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        self.logger.write("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        self.logger.write("Std Test AUC: {:.4f}".format(np.std(aucs)))

    def print_(self, test_acc, test_auc, train_loss):
        self.logger.write("Average Test accuracy: {:.4f}".format(test_acc))
        self.logger.write("Average Test AUC: {:.4f}".format(test_auc))
        self.logger.write("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def apply_resource_fluctuation(self, round_number):
        """
        对所有客户端应用资源波动
        每隔 fluctuation_frequency 轮，改变客户端的计算能力和传输能力
        
        参数:
            round_number: 当前轮次
        """
        if self.args.resource_fluctuation == False:
            return
        
        # 只在指定频率的轮次应用波动
        if round_number % self.args.fluctuation_frequency != 0:
            return
        
        self.logger.write(f"\nApplying resource fluctuation in round {round_number}")
        
        # 为每个客户端生成随机波动因子
        fluctuation_scale = self.args.fluctuation_scale
        for client in self.clients:
            # 计算能力波动: 在 [1-scale, 1+scale] 范围内随机波动
            compute_factor = 1.0 + (2.0 * np.random.random() - 1.0) * fluctuation_scale
            # 传输能力波动: 在 [1-scale, 1+scale] 范围内随机波动
            transmission_factor = 1.0 + (2.0 * np.random.random() - 1.0) * fluctuation_scale
            
            # 应用波动
            original_compute = client.compute_power
            original_transmission = client.transmission_power
            
            client.compute_power *= compute_factor
            client.transmission_power *= transmission_factor
            
            # 确保不会变得过小或过大
            client.compute_power = max(0.1, min(5.0, client.compute_power))
            client.transmission_power = max(1.0, min(100.0, client.transmission_power))
            
            # 记录变化
            self.logger.write(f"Round {round_number}, Client {client.id}: Compute power {original_compute:.2f} -> {client.compute_power:.2f}, "
                        f"Transmission power {original_transmission:.2f} -> {client.transmission_power:.2f}")