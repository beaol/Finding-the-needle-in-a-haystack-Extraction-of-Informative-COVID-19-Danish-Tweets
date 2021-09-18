import argparse
import DataProcessing
from DataLoader import DataLoader
import torch
import pytorch_lightning as pl
from CovidNet import CovidNet
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from ray import tune
import os

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', dest='gpu', help='GPU device id to use [0]',
        default=0, type=int)
    parser.add_argument('--seed', dest='seed', help='Set random seed',
        default=0, type=int)
    parser.add_argument('--dropout', dest='dropout', help='Set dropout',
        default=0.5, type=int)
    parser.add_argument('--static', dest='static', help='Set if only the static representation of one-hot-encodings should be used',
        default=0, type=int)
    parser.add_argument('--bert', dest='bert', help='Set if only the bert embeddings should be used',
        default=0, type=int)
    parser.add_argument('--resnet', dest='resnet', help='Set resnet should be used on the data',
        default=0, type=int)
    parser.add_argument('--pre_trained_resnet', dest='pre_trained_resnet', help='Set if resnet should be pretrained',
        default=0, type=int)
    parser.add_argument('--samples', dest='samples', help='Set number of samples',
        default=200, type=int)
    parser.add_argument('--file', dest='file', help='Set which tweet file to use for training/validating',
        default="", type=str)

    args = parser.parse_args()
    return args

#To fix issue where ray cannot run on cluster
os.environ["SLURM_JOB_NAME"] = "bash"

args = parse_args()

file = f"./Data/{args.file}tweets_final_train_eval.txt"

seed = args.seed
static = bool(args.static)
bert = bool(args.bert)
resnet = args.resnet
pre_trained_resnet = bool(args.pre_trained_resnet)
dropout = args.dropout
samples = args.samples

pl.seed_everything(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_deterministic(True)
if torch.cuda.is_available():
    num_processes = 1
    gpus = 1
    device = f"cuda:{args.gpu}"
else:
    num_processes = 1
    gpus = None
    device = "cpu"

if args.file == "all_":
    data_length = 4000
else:
    data_length = 400

x_train, x_test, y_train, y_test, vocab_size = DataProcessing.standard_split(file, bert, data_length, seed)

train_ds = DataLoader(x_train, y_train, bert, resnet, data_length, device=device)
test_ds = DataLoader(x_test, y_test, bert, resnet, data_length, device=device)

def train_model(config):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)
    torch.set_deterministic(True)

    batch_size = config["batch_size"]
    n_filters = config["n_filters"]
    filters = config["filters"]
    lr = config["lr"]
    l2 = config["l2"]
    num_epochs = config["num_epochs"]
    emb_size = config["emb_size"]
    log_path = f"logs/covidnet_std_seed_{seed}_batch_{batch_size}_nfilters_{n_filters}_filters_{filters}_lr_{lr}_epochs_{num_epochs}_embsize_{emb_size}"
    train_dataloader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_ds, batch_size=1)

    covidnet = CovidNet(n_filters, filters, vocab_size, emb_size, len(["I+","I-","U"]), lr, l2, dropout, static, bert, resnet, pre_trained_resnet)
    trainer = pl.Trainer(num_processes=num_processes, gpus=gpus, max_epochs=num_epochs, logger=pl.loggers.TensorBoardLogger(log_path))
    trainer.fit(covidnet.float(), train_dataloader=train_dataloader)
    res = trainer.test(test_dataloaders=test_dataloader)
    rep = res[0]["macro f1 score"]
    print("HERE")
    print(rep)
    tune.report(loss = rep)

config = {
    "batch_size": tune.choice([30,40,50]),
    "n_filters": tune.choice([i for i in range(50,201,25)]),
    "filters": tune.choice([[1],[1,3],[1,2,3],[1,3,5],[2,3,4],[3,4,5],[1,2,3,4],[2,3,4,5],[1,3,5,7]]),
    "lr": 1e-4,
    "l2": 0,
    "num_epochs": 20,
    "emb_size": 400
}

analysis = tune.run(
    train_model,
    resources_per_trial={
        "cpu": 1,
        "gpu": 1
    },
    config=config,
    num_samples=samples,
    local_dir="."
)

print(analysis.get_best_config(metric="loss", mode="max"))