import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
import numpy as np


class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        self.logger.write(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        self.logger.write("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self):
        for i in range(1, 1 + self.global_rounds):

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
            for client in self.selected_clients:
                client_train_time, client_trans_time, client_time_cost = client.train()
                # 只有实际训练了的客户端才算参与
                if client_time_cost > 0:
                    clients_time_cost.append(client_time_cost)
                    clients_train_time.append(client_train_time)
                    clients_trans_time.append(client_trans_time)
                    self.all_clients_train_time.append(client_train_time)
                    self.all_clients_trans_time.append(client_trans_time)

            self.receive_models()

            self.aggregate_parameters()

            self.global_round_time_cost.append(np.max(clients_time_cost))
            self.logger.write(f"The current global round takes {self.global_round_time_cost[-1]} seconds")

            # 记录客户端计算能力和传输能力
            if i % self.eval_gap == 0:
                self.logger.write("\nClient capabilities:")
                for client in self.selected_clients:
                    self.logger.write(f"Client {client.id}: Compute Power = {client.compute_power:.2f}x, Transmission Power = {client.transmission_power:.2f} Mbps")

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        self.logger.write("Best accuracy: {}".format(max(self.rs_test_acc)))
        # self.self.logger.write_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        self.logger.write(f"Averaged Global Round Trans Time Cost: {np.mean(self.all_clients_trans_time)}")
        self.logger.write(f"Averaged Global Round Train Time Cost: {np.mean(self.all_clients_train_time)}")
        self.logger.write(f"Averaged Global Round Time Cost: {np.mean(self.global_round_time_cost)}")
        


        self.save_results()
        self.save_global_model()

        # if self.num_new_clients > 0:
        #     self.eval_new_clients = True
        #     self.set_new_clients(clientAVG)
        #     self.logger.write(f"\n-------------Fine tuning round-------------")
        #     self.logger.write("\nEvaluate new clients")
        #     self.evaluate()