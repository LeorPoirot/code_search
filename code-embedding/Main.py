from opt import get_opt
import torch
import random
import numpy as np
import os
from utils.util_data import load_dict, load_data, create_code_retrieval_model
from metric.Loss import CosRankingLoss
from model.CodeRetrievalModel import CodeRetrievalModel
from trainer.CodeRetrievalTrainer import CodeRetrievalTrainer
from trainer.CodereRerievalSearcher import CodereRerievalSearcher

def main():
    print("Start... => main.py PID: %s" % (os.getpid()))
    opt = get_opt()

    # torch.manual_seed(opt.seed)
    # random.seed(opt.seed)
    # np.random.seed(opt.seed)
    # print(opt.gpus)
    # if opt.gpus and torch.cuda.is_available():
    #     print("opt.gpus: ", opt.gpus)
    #     gpu_list = [int(k) for k in opt.gpus.split(",")]
    #     print("gpu_list: ", gpu_list)
    #     print("gpu_list[0]: ", gpu_list[0])
    #     print("type(gpu_list[0]): ", type(gpu_list[0]))
    #     os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus

    # all_dict = load_dict()
    if opt.train_mode == "train":
        data_loader, all_dict = load_data(opt)
        print('Loaded dataset sucessfully.')
        model = create_code_retrieval_model(opt, all_dict)
        print('created model..')
        cos_ranking_loss = CosRankingLoss(opt).cuda() if torch.cuda.is_available() else CosRankingLoss(opt)
        optim = torch.optim.Adam(model.parameters(), opt.lr)
        print('created criterion and optim')  # standard
        trainer = CodeRetrievalTrainer(model, data_loader, all_dict, opt)
        # begin to training !!!
        trainer.train(cos_ranking_loss, optim, opt.train_epoch)

    if opt.train_mode == "query":
        data_loader, all_dict, code_body, func_name = load_data(opt, model='n_query')
        model = create_code_retrieval_model(opt, all_dict)
        model.load_state_dict(torch.load('./model/Models/model_0.pt', map_location=lambda storage, loc: storage))
        print('Loaded dataset sucessfully.')
        code_searcher = CodereRerievalSearcher(10, code_body, func_name, all_dict, opt, model, data_loader)
        result = code_searcher.search('set working directory')
        code_searcher.save_result()
        # print(result)


if __name__ == '__main__':
    main()
