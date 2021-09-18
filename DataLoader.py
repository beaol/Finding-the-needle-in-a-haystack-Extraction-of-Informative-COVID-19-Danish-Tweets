import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms
import bert as b

class DataLoader(Dataset):
    def __init__(self, x, y, bert, multilingual, data_length, unk_token, test=False, device="cpu"):
        self.x = x
        self.y = y
        self.unk_token = unk_token
        self.test = test

        self.bert = bert
        self.length = len(y)

        if self.bert:
            tokenizer, model = b.getBertModel(multilingual=multilingual)
            self.tokenizer = tokenizer
            self.model = model
            if multilingual:
                self.file_name = f"./Data/BertData{data_length}/bert"
            else:
                self.file_name = f"./Data/BertDataDanish{data_length}/bert"
        else:
            self.file_name = f"./Data/OneHotData{data_length}/onehot"

        self.device = device

    def __getitem__(self, index):
        f_index = self.x[index]
        f_name = self.file_name + f"{f_index}.txt"
        emb = []
        with open(f_name, "r", encoding="utf-8") as file:
            for l in file:
                emb.append(list(map(float, l.split(";"))))
                if not self.test and not self.bert:
                    for i in range(len(emb[0])):
                        if np.random.randint(1,101) == 1:
                            emb[0][i] = self.unk_token
                
            if self.bert and not self.test:
                for i in range(len(emb)):
                    if np.random.randint(1,101) == 1:
                        contextualized_embeddings = b.getContextualizedEmbeddings(self.unk_token, self.tokenizer, self.model, 0)
                        unk_embedding = contextualized_embeddings.last_hidden_state.tolist()[0][1]
                        emb[i] = unk_embedding

        torch_emb = torch.tensor(emb).to(self.device)
        torch_emb = torch_emb.squeeze(0)

        label = self.y[index]
        return torch_emb, torch.tensor(np.array(label)).to(self.device)

    def __len__(self):
        return self.length