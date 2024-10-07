import time
import numpy as np
import torch
import os
from flcore.clients.clientproto import clientProto
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from utils.data_utils import read_client_data
from threading import Thread
from collections import defaultdict
from tqdm import tqdm


class FedProto(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientProto)

        self.logger.write(f"Join ratio / total clients: {self.join_ratio} / {self.num_clients}")
        self.logger.write("Finished creating server and clients.")

        self.global_protos = None
        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes

    def train(self):
        self.selected_clients = self.select_clients()
        for client in self.clients:
            client.get_protos()

        self.receive_protos()
        for k, v in self.global_protos.items():
            print(type(k), k)
            print(type(v), v)
        try:
            torch.save(self.global_protos, os.path.join(self.model_folder_name, "global_protos.pt"))
            print('Successfully saved global prototypes')
        except Exception as e:
            print('Failed to save global prototypes:', e)
        exit()
        for i in range(self.start_round, self.end_round):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            for client in tqdm(self.selected_clients):
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_protos()
            self.send_protos()

            if i%self.eval_gap == 0:
                self.logger.write(f"-------------Round number: {i}-------------")
                self.logger.write("Evaluate heterogeneous models")
                self.evaluate()

            self.Budget.append(time.time() - s_t)
            self.logger.write("The current global round takes {} seconds".format(self.Budget[-1]))

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        self.logger.write("Best accuracy: {}".format(max(self.rs_test_acc)))
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        # print(max(self.rs_test_acc))
        # print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_models()
        
    def send_protos(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_protos(self.global_protos)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        uploaded_protos = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            uploaded_protos.append(client.protos)
            
        self.global_protos = proto_aggregation(uploaded_protos)


# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L221
def proto_aggregation(local_protos_list):
    agg_protos_label = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            agg_protos_label[label].append(local_protos[label])

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0].data

    return agg_protos_label