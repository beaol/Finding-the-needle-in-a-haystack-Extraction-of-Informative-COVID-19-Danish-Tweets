import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import transforms

class CovidNet(pl.LightningModule):
    
    def __init__(self, n_filters, filters, vocab_size, embedding_size, output_dim, lr, l2, dropout, static, bert, multilingual, pretrained_embeddings, n_iplus, n_iminus, n_u, device):
        super(CovidNet, self).__init__()
        self.lr = lr
        self.l2 = l2
        self.static = static
        self.bert = bert
        self.multilingual = multilingual
        self.n_iplus = n_iplus
        self.n_iminus = n_iminus
        self.n_u = n_u
        self.train_i = 0
        self.test_i = 0
        self.dev = device

        if not self.static:
            self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx = 0)
            self.embedding.weight = nn.Parameter(torch.FloatTensor(pretrained_embeddings))

        self.convs_final = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                            out_channels = n_filters, 
                                            kernel_size = (filter_size, embedding_size)) 
                                    for filter_size in filters
                                    ])
        
        self.dropout = nn.Dropout(dropout)
        
        self.fc = nn.Linear(len(filters)*n_filters, output_dim)
        
    def forward(self, input):
        if not self.static:
            embedded = self.embedding(input)
            input = embedded.unsqueeze(1)
        else:
            input = input.unsqueeze(1)

        conved = [F.relu(conv(input)).squeeze(3) for conv in self.convs_final]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        cat = self.dropout(torch.cat(pooled, dim = 1))
            
        return self.fc(cat)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        if not self.static:
            if not self.bert:
                x = x.long()
            else:
                x = x.float()
        else:
            x = x.float()

        out = self(x)
        inverse_sqrt_number_of_samples = [1/(self.n_iplus**0.5), 1/(self.n_iminus**0.5), 1/(self.n_u**0.5)]
        loss = F.cross_entropy(out, y.long(), weight=torch.tensor(inverse_sqrt_number_of_samples).to(self.dev))
        self.logger.experiment.add_scalar('train_loss', loss, self.train_i)
        self.logger.experiment.flush()
        self.train_i += 1
        return loss

    def test_step(self, test_batch, batch_idx):
        softmax = nn.Softmax(dim=1)
        x, y = test_batch
        if not self.static:
            x = x.long()
        else:
            x = x.float()

        out = self(x)
        confidence = softmax(out)
        n_digits = 2
        max_confidence = torch.round(torch.max(confidence) * 10**n_digits) / (10**n_digits)
        y_pred = torch.argmax(confidence)
        if y_pred.item() == y.item():
            self.logger.experiment.add_scalar(f'prediced label {y_pred.item()} correct', max_confidence, self.test_i)
        else:
            self.logger.experiment.add_scalar(f'predicted label {y_pred.item()} wrong (actual {y.item()})', max_confidence, self.test_i)
        
        self.logger.experiment.flush()
        self.test_i += 1

        return {"true":y, "pred": y_pred}

    def test_epoch_end(self, outputs):
        true_vals = [x["true"].item() for x in outputs]
        pred_vals = [x["pred"].item() for x in outputs]

        f1_score_ind = f1_score(true_vals, pred_vals, labels=[0, 1, 2], average=None)
        f1_score_macro = f1_score(true_vals, pred_vals, labels=[0, 1, 2], average="macro")

        return {"progress_bar": {"individual f1 scores": f1_score_ind,"macro f1 score": f1_score_macro}}

    def sorted_test(self, x_test, y_test, data_length, model):
        softmax = nn.Softmax(dim=1)

        sorted_concat_x_y = sorted(zip(y_test, x_test), key=lambda pair: pair[0])
        sorted_x_test = [x for _, x in sorted_concat_x_y]
        sorted_y_test = [y for y, _ in sorted_concat_x_y]
        if self.bert:
            if self.multilingual:
                file_name = f"./Data/BertData{data_length}/bert"
            else:
                file_name = f"./Data/BertDataDanish{data_length}/bert"
        else:
            file_name = f"./Data/OneHotData{data_length}/onehot"
        y_pred = []
        
        for i,x in enumerate(sorted_x_test):
            emb = []
            f_name = file_name + f"{x}.txt"
            with open(f_name, "r", encoding="utf-8") as file:
                for l in file:
                    emb.append(list(map(float, l.split(";"))))
                    
            torch_emb = torch.tensor(emb).unsqueeze(0).to(self.dev)
            if not self.bert:
                input = torch_emb.squeeze(0).long()
            else:
                input = torch_emb.float()

            pred = model(input)
            pred = softmax(pred)
            pred = torch.argmax(pred)
            pred = pred.item()
            #if pred != sorted_y_test[i]:
            print(f"sentence: {x}, label: {sorted_y_test[i]}, pred: {pred}")
            y_pred.append(pred)

        f1_score_ind = f1_score(sorted_y_test, y_pred, labels=[0, 1, 2], average=None)
        f1_score_macro = f1_score(sorted_y_test, y_pred, labels=[0, 1, 2], average="macro")
        print({"individual f1 scores": f1_score_ind,"macro f1 score": f1_score_macro})

        data = pd.DataFrame()
        data['tweets'] = [i for i in range(len(x_test))]
        data['label'] = y_pred

        g = sn.FacetGrid(data)
        g = g.map(plt.scatter, "tweets", "label", edgecolor="w")
        plt.plot(data['tweets'], sorted_y_test, color='r')
        plt.gcf().set_size_inches(20, 5)
        plt.savefig(f'./Data/sorted_predictions_{i}.png', dpi=299)
        plt.clf()

        confusion_matrix_test = confusion_matrix(sorted_y_test, y_pred, labels=[0, 1, 2])
        df_cm = pd.DataFrame(confusion_matrix_test, index = ["Informative+", "Informative-", "Uninformative"],
                    columns = ["Informative+", "Informative-", "Uninformative"])
        plt.figure(figsize = (10,7))
        hm = sn.heatmap(df_cm, annot=True, fmt='g')
        fig = hm.get_figure()
        fig.savefig(f"./Data/CNN_confusion_matrix_{i}.png")
        fig.clf()