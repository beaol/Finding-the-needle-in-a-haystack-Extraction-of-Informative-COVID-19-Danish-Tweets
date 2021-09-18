import argparse
#import TweetLibrary
import DataProcessing
#import LabelUI
import Bayesian_model
from DataLoader import DataLoader
import torch
import pytorch_lightning as pl
from CovidNet import CovidNet
from torch.utils.tensorboard import SummaryWriter
import numpy as np

datafile_danish_small = "./COVID19dk.json"
datafile_danish_small_nontruncated = "./COVID19dk_nontruncated.json"
datafile_danish_large = "./dksund.json"

def RunLabelling():
    alltweets = DataProcessing.CleanData(datafile_danish_small_nontruncated)
    LabelUI.Labelling(alltweets)

def CreateJSONWithoutTruncatedTweets(filename, tweets):
    TweetLibrary.replace_truncated_tweets(filename, tweets)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', dest='gpu', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
            default=6, type=int)
    parser.add_argument('--lr', dest='lr', help='Learning rate',
            default=0.0046601, type=float)#82: 0.0048630
    parser.add_argument('--l2', dest='l2', help='L2 regularization',
            default=0, type=float)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size for model training',
            default=55, type=int)#82: 56
    parser.add_argument('--model_path', dest='model_path', help='Path where the model should be saved to or loaded from',
            default="./models/version_1_epoch_24", type=str)
    parser.add_argument('--testing', dest='testing', help='Toggle live testing mode',
            default=0, type=int)
    parser.add_argument('--seed', dest='seed', help='Set random seed',
            default=2, type=int)
    parser.add_argument('--dropout', dest='dropout', help='Set dropout',
            default=0.5, type=int)
    parser.add_argument('--n_filters', dest='n_filters', help='Set number of filters',
        default=100, type=int)
    parser.add_argument('--emb_size', dest='emb_size', help='Set embedding size for embedding layer',
        default=400, type=int)
    parser.add_argument('--static', dest='static', help='Set if only the static representation of one-hot-encodings should be used',
        default=1, type=int)
    parser.add_argument('--bert', dest='bert', help='Set if only the bert embeddings should be used',
        default=1, type=int)
    parser.add_argument('--multilingual', dest='multilingual', help='Set if the bert embeddings should be multilingual',
        default=1, type=int)
    parser.add_argument('--file', dest='file', help='Set which tweet file to use for training/validating',
        default="all_", type=str)

    args = parser.parse_args()
    return args

def endtest(model, args_file, bert, seed, tokens_integer_enc_dict, ordered_tokens, max_sentence_length, multilingual):
    file = f"./Data/{args_file}tweets_final_endtest.txt"
    if args_file == "all_":
        data_length = 1000
    else:
        data_length = 100

    indices, y, vocab_size, pretrained_embs, tokens_integer_enc_dict, ordered_tokens, max_sentence_length = DataProcessing.load_data(file, bert, data_length, seed, tokens_integer_enc_dict, ordered_tokens, max_sentence_length, multilingual=multilingual)

    print("Final endtest:")
    model.sorted_test(indices, y, data_length, model.eval())

if __name__ == "__main__":
    args = parse_args()

    # RunLabelling()
    # DataProcessing.load_pretrained_embeddings()
    file = f"./Data/{args.file}tweets_final_train_eval.txt"
    if args.file == "all_":
        data_length = 4000
    else:
        data_length = 400
    # print("Scikit learning model:")
    # Bayesian_model.TrainAndTest(file)
    # print("\nOwn Naive Bayes model:")
    # Own_Bayesian_model.TrainAndTest(file)

    num_epochs = args.num_epochs
    lr = args.lr
    l2 = args.l2
    batch_size = args.batch_size
    testing = bool(args.testing)
    seed = args.seed
    model_path = args.model_path
    dropout = args.dropout
    n_filters = args.n_filters
    emb_size = args.emb_size
    static = bool(args.static)
    bert = bool(args.bert)
    multilingual = bool(args.multilingual)

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

    x_train, x_test, y_train, y_test, vocab_size, pretrained_embeddings, tokens_integer_enc_dict, ordered_tokens, max_sentence_length = DataProcessing.standard_split(file, bert, data_length, seed, multilingual=multilingual)
    print(f"vocab size: {vocab_size}")

    n_iplus = 0
    n_iminus = 0
    n_u = 0
    labels = y_train.copy()
    for idx,label in enumerate(labels):
        if label == 0:
            n_iplus += 1
        elif label == 1:
            n_iminus += 1
        else:
            n_u += 1

    unk_token = vocab_size-1
    if bert:
        static = True
        emb_size = 768
        unk_token = "[UNK]"

    train_ds = DataLoader(x_train, y_train, bert, multilingual, data_length, unk_token, test=False, device=device)
    test_ds = DataLoader(x_test, y_test, bert, multilingual, data_length, unk_token, test=True, device=device)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_ds, batch_size=1)

    filters = [1,3,5,7]
    covidnet = CovidNet(n_filters, filters, vocab_size, emb_size, len(["I+","I-","U"]), lr, l2, dropout, static, bert, multilingual, pretrained_embeddings, n_iplus, n_iminus, n_u, device)
    
    #endtest(covidnet.eval(), args.file, bert, seed, tokens_integer_enc_dict, ordered_tokens, max_sentence_length, multilingual)
    
    trainer = pl.Trainer(num_processes=num_processes, gpus=gpus, max_epochs=num_epochs, logger=pl.loggers.TensorBoardLogger(f"logs/covidnet_std_seed_{seed}"))
    trainer.fit(covidnet.float(), train_dataloader=train_dataloader)
    t = trainer.test(test_dataloaders=test_dataloader)
    print("train data for testing:")
    train_dataloader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=1)
    trainer.test(test_dataloaders=train_dataloader)

    covidnet.sorted_test(x_test, y_test, data_length, covidnet.eval())
    endtest(covidnet.eval(), args.file, bert, seed, tokens_integer_enc_dict, ordered_tokens, max_sentence_length, multilingual)