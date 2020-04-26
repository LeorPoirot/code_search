import argparse


def get_opt():
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument('--train_mode', default='train')   # train
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--lower', default=True, type=bool, help='lowercase data')
    parser.add_argument('--cos_ranking_loss_margin', type=float, default=0.05, required=False,
                        help='margin for CosRankingLoss')
    # model arguments
    parser.add_argument('--nlayers', type=int, default=1, help='Number of layers in the LSTM encoder/decoder')
    parser.add_argument('--rnn_type', type=str, default='LSTM',
                        help='type of  recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
    parser.add_argument('--embsz', type=int, default=100, help='size of word embeddings')
    parser.add_argument('--nout', type=int, default=400, help='size of word vector')
    parser.add_argument('--nhid', type=int, default=200, help='humber of hidden units per layer')
    parser.add_argument('--init_type', type=str, default="xulu")
    parser.add_argument('--batchsz', type=int, default=64, help='Maximum batch size') #128
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout probability; applied between LSTM stacks.')
    parser.add_argument("--train_epoch", type=int, default=15, help="Epoch to supervised training.")
    parser.add_argument('--log_interval', type=int, default=5, help="Print stats at this interval.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument('--save_dir', required=False, default='./model/Models', help='Directory to save models')
    parser.add_argument('--gpus', default="0", type=str, help="Use CUDA on the listed devices.")
    opt = parser.parse_args()
    return opt
