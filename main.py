import os

import wandb

wandb.init(project="test-project", entity="ytwang-dst")

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ["WANDB_DIR"] = os.path.abspath("./logs")

import argparse
from src.utils import *
from torch.utils.data import DataLoader
from src import train

print(torch.cuda.device_count())
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
# PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:512"

if not torch.cuda.is_available():
    print("ERROR: no available GPU.")
else:
    print("CUDA: OK")

torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='MOSEI Sentiment Analysis')
parser.add_argument('-f', default='', type=str)

# Fixed
parser.add_argument('--model', type=str, default='MulT',
                    help='name of the model to use (Transformer, etc.)')

# Tasks
parser.add_argument('--vonly', action='store_true',
                    help='use the crossmodal fusion into v (default: False)')
parser.add_argument('--aonly', action='store_true',
                    help='use the crossmodal fusion into a (default: False)')
parser.add_argument('--lonly', action='store_true',
                    help='use the crossmodal fusion into l (default: False)')
parser.add_argument('--aligned', action='store_true',
                    help='consider aligned experiment or not (default: False)')
parser.add_argument('--dataset', type=str, default='mosi',  # mosei_senti
                    help='dataset to use (default: mosei_senti)')
parser.add_argument('--data_path', type=str, default='data',
                    help='path for storing the dataset')

# Dropouts
parser.add_argument('--attn_dropout', type=float, default=0.2,  # 0.1
                    help='attention dropout')
parser.add_argument('--attn_dropout_a', type=float, default=0.2,  # 0.0
                    help='attention dropout (for audio)')
parser.add_argument('--attn_dropout_v', type=float, default=0.2,  # 0.0
                    help='attention dropout (for visual)')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')
parser.add_argument('--embed_dropout', type=float, default=0.2,  # 0.25
                    help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.1,
                    help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.1,  # 0.0
                    help='output layer dropout')

# Architecture
parser.add_argument('--nlevels', type=int, default=5,
                    help='number of layers in the network (default: 5)')
parser.add_argument('--num_heads', type=int, default=10,  # 注意要被dim=30可整除
                    help='number of heads for the transformer network (default: 5)')
parser.add_argument('--attn_mask', action='store_false',
                    help='use attention mask for Transformer (default: true)')

# Tuning
parser.add_argument('--batch_size', type=int, default=2, metavar='N',  # 4
                    help='batch size (default: 24)')
parser.add_argument('--clip', type=float, default=0.8,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--num_epochs', type=int, default=150,
                    help='number of epochs (default: 40)')
parser.add_argument('--when', type=int, default=20,
                    help='when to decay learning rate (default: 20)')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='number of chunks per batch (default: 1)')

# Logistics
parser.add_argument('--log_interval', type=int, default=30,
                    help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
parser.add_argument('--name', type=str, default='mult',
                    help='name of the trial (default: "mult")')
args = parser.parse_args()

torch.manual_seed(args.seed)
dataset = str.lower(args.dataset.strip())
valid_partial_mode = args.lonly + args.vonly + args.aonly

if valid_partial_mode == 0:
    args.lonly = args.vonly = args.aonly = True
elif valid_partial_mode != 1:
    raise ValueError("You can only choose one of {l/v/a}only.")

use_cuda = False
# use_cuda = True

output_dim_dict = {
    'mosi': 1,
    'mosei_senti': 1,
    'iemocap': 8
}

criterion_dict = {
    'iemocap': 'CrossEntropyLoss'
}

torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    if args.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
    else:
        torch.cuda.manual_seed(args.seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        use_cuda = True

####################################################################
#
# Load the dataset (aligned or non-aligned)
#
####################################################################

print("Start loading the data....")

train_data = get_data(args, dataset, 'train')
valid_data = get_data(args, dataset, 'valid')
test_data = get_data(args, dataset, 'test')
print(type(train_data))

DEF_DEVICE = 'cuda' if use_cuda else 'cpu'
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device=DEF_DEVICE))
valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device=DEF_DEVICE))
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device=DEF_DEVICE))

print('Finish loading the data....')
if not args.aligned:
    print("### Note: You are running in unaligned mode.")


print("test-data-check:")
# print("len::", len(test_data))    # (4659, 3) [id, start, end]  171;
# print("type:", type(test_data))
# print("meta.shape:", test_data.meta.shape)
# print("meta.len:", len(test_data.meta))
# 从test_set中, 获取的数据sample为: sample_ind: tensor([158,  45,  85,  28, 141,  57])
print("real_sample:")
# print(test_data.meta[56])  # 19
# print(test_data.meta[119]) # 18
# print(test_data.meta[36])  # 15
# print(test_data.meta[44])  # 06
# print(test_data.meta[45])
# print(test_data.meta[85])
# print(test_data.meta[28])
# print(test_data.meta[141])
# print(test_data.meta[57])
# print("ck1", test_data.meta[256][0])
# for i in test_data.meta:
#     if i[0].isnumeric():
#         print(i)
#     if i[0] in ['30762', '30762_14']:
#         print("----", i)

print("real_sample:. done.")

# print('meta:', test_data.meta)
# ck_test = test_data[:args.batch_size]
# print("ck,", ck_test)
input("start:")

# print("id:", test_data.meta[917])       # id: ['245582' '11.326' '16.207']
# print("id:", test_data.meta[903])       # id: ['24504' '32.857' '41.18']
# print("id:", test_data.meta[1214])      # id: ['29751' '7.115' '14.381']
# print("id:", test_data.meta[892])       # id: ['243981' '32.208' '39.185']
# print("id:", test_data.meta[3843])      # id: ['lYwgLa4R5XQ' '31.7192743764' '36.5006802721']
# print("id:", test_data.meta[4488])      # id: ['xobMRm5Vs44' '21.672' '26.813']
#
####################################################################
#
# Hyperparameters
#
####################################################################

hyp_params = args
hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = train_data.get_dim()
hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = train_data.get_seq_len()
hyp_params.layers = args.nlevels
hyp_params.use_cuda = use_cuda
hyp_params.dataset = dataset
hyp_params.when = args.when
hyp_params.batch_chunk = args.batch_chunk
hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)
hyp_params.model = str.upper(args.model.strip())
hyp_params.output_dim = output_dim_dict.get(dataset, 1)
hyp_params.criterion = criterion_dict.get(dataset, 'L1Loss')

print("n_train:", hyp_params.n_train)
print("n_test:", hyp_params.n_test)

if __name__ == '__main__':
    # print(torch.cuda.memory_summary())
    torch.autograd.set_detect_anomaly(True)

    hypar_defaults = dict(

        lr=hyp_params.lr,  #
        optim=hyp_params.optim,  #
        num_epochs=hyp_params.num_epochs,  #

        nlevels=hyp_params.nlevels,  #
        num_heads=hyp_params.num_heads,  #
        batch_size=hyp_params.batch_size,  #

        clip=hyp_params.clip,  #
        attn_dropout=hyp_params.attn_dropout,  #
        out_dropout=hyp_params.out_dropout,  #
        embed_dropout=hyp_params.embed_dropout,  #
    )

    wandb.init(config=hypar_defaults)
    wandb.config = hypar_defaults
    print("dict:", wandb.config)

    test_loss = train.initiate(hyp_params, train_loader, valid_loader, test_loader)
