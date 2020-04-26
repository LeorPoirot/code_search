import time
import os
import datetime
import torch
import random


class CodeRetrievalTrainer(object):
    def __init__(self, model, train_dataloader, all_dict, opt):
        self.opt = opt
        self.model = model
        self.train_dataloader = train_dataloader
        # self.val_dataloader = val_dataloader
        self.dict_meth = all_dict.meth_name
        self.dict_api = all_dict.api_seq
        self.dict_token = all_dict.tokens
        self.dict_comment = all_dict.description
        self.train_dataloader = train_dataloader

    def train(self, criterion, optim, train_epoch, start_time=None):
        if not start_time:
            self.start_time = time.time()
        else:
            self.start_time = start_time
        for epoch in range(train_epoch):
            model_name = os.path.join(self.opt.save_dir, f'model_{epoch}.pt')

            train_loss_averaged = self.train_epoch(epoch, criterion, optim)
            print("epoch: {} train_loss_averaged: {} ".format(epoch, train_loss_averaged))

            # if self.opt.retrieval_train_dataset_split_type == "train":
            #
            #     time0 = datetime.datetime.now()
            #
            #     val_loss_averaged, total_val_sample = self.evaluator.validation(criterion)
            #     print("epoch: {} val_loss_averaged(origin_ranking_loss,no co-attn): {} total_val_sample:{}". \
            #           format(epoch, val_loss_averaged, total_val_sample))
            #
            #     time1 = datetime.datetime.now()
            #
            #     print("time for validation: ", (time1 - time0))
            #
            #     if (epoch + 1) % 1 == 0:
            #         retrieval_pred_file = model_name + "-" + self.opt.data_query_file.split('/')[-1] + \
            #                               "-uv" + str(self.opt.use_val_as_codebase) + "r" + str(
            #             self.opt.remove_wh_word) + "-.re"
            #         self.metric_evaluator.retrieval(pred_file=retrieval_pred_file)
            #         self.metric_evaluator.eval_retrieval_json_result(pred_file=retrieval_pred_file)
            #
            #         time2 = datetime.datetime.now()
            #         print("time for metric: ", (time2 - time1))
            #
            #     print("time for metric: ", (datetime.datetime.now() - time1))
            #
            # else:
            #     print("no validation in train progress, self.opt.retrieval_train_dataset_split_type: ",
            #           self.opt.retrieval_train_dataset_split_type)

            # if len(self.opt.gpus) == 1:

            torch.save(self.model.state_dict(), model_name)
            # elif len(self.opt.gpus) > 1:
            #     torch.save(self.model.module.state_dict(), model_name)
            print("Save model as %s" % model_name)

    def train_epoch(self, epoch, criterion, optim):
        self.model.train()
        total_train_loss = 0
        total_sample = 0

        for i, data_batch in enumerate(self.train_dataloader):
            (meth_data, token_data, api_data,  comment_data), tree_batch = data_batch
            batch_size = comment_data.size()[0]
            total_sample += batch_size
            code_data = (meth_data, token_data, api_data, tree_batch)
            good_comment_data = comment_data
            bad_comment_list = comment_data.tolist()
            random.shuffle(bad_comment_list)
            bad_comment_data = torch.tensor(bad_comment_list)
            code_feat, good_comment_feat, bad_comment_feat = self.model(code_data,
                                                                        good_comment_data,
                                                                        bad_comment_data)
            code_feat = code_feat.cpu()
            good_comment_feat = good_comment_feat.cpu()
            bad_comment_feat = bad_comment_feat.cpu()
            loss = criterion(code_feat, good_comment_feat, bad_comment_feat)
            self.model.zero_grad()
            loss.backward(retain_graph=True)
            optim.step()
            total_train_loss += (loss.item()) * batch_size
            if i % self.opt.log_interval == 0 and i > 0:
                print('Epoch %3d, %6d/%d batches; loss: %14.10f; %s elapsed' % (
                    epoch, i, len(self.train_dataloader), loss,
                    str(datetime.timedelta(seconds=int(time.time() - self.start_time)))))

        return total_train_loss / total_sample
