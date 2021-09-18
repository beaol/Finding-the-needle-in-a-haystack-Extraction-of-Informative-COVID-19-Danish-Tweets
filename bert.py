import torch
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModelForPreTraining

def getBertModel(multilingual=True):
    if multilingual:
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        model = BertModel.from_pretrained("bert-base-multilingual-uncased")
    else:
        tokenizer = BertTokenizer.from_pretrained("Maltehb/danish-bert-botxo")
        model = BertModel.from_pretrained("Maltehb/danish-bert-botxo")

    return tokenizer, model

def getContextualizedEmbeddings(text, tokenizer, model, max_length):
    #Tokenize input and remove start and end tokens from sentence
    if max_length > 0:
        encoded_input = tokenizer(text, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    else:
        encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

    #Get contextualized embeddings
    output = model(**encoded_input)
    return output