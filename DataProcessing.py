import re
import json
import random
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
import torch
import bert as b

def RemoveLinks(text, replace):
    return re.sub(r'https?:\/\/.\S+', replace, text)

def RemoveTags(text, replace):
    return re.sub(r'#\S*', replace, text)

def RemoveUsers(text, replace):
    return re.sub(r'@\S*', replace, text)

def RemoveRT(text, replace):
    return re.sub(r'^RT[\s]+', replace, text)

def RemoveDotsAndCommas(text, replace):
    return re.sub(r'[.,]', replace, text)

def RemoveLinksTagsUsers(org_tweet_text):
    #Remove links
    tweet_text = RemoveLinks(org_tweet_text, "")
    #Remove hashtags - like in the article
    tweet_text = RemoveTags(tweet_text, "")
    #Remove user mentions - like in the article
    tweet_text = RemoveUsers(tweet_text, "")

    return tweet_text

def GetIDText(text):
    return text.split(", label: ")[0]

def GetID(text, id_text=False):
    if id_text: text.split(", text: ")[0].split("id: ")[1]
    else: return GetIDText(text).split(", text: ")[0].split("id: ")[1]

def GetText(text, id_text=False):
    if id_text: return text.split(", text: ")[1]
    else: return GetIDText(text).split(", text: ")[1]

def GetLabel(text):
    return text.split(", label: ")[1].strip()

def RemoveEmojis(text):
    regrex_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U0001F1F2-\U0001F1F4"  # Macau flag
        u"\U0001F1E6-\U0001F1FF"  # flags
        u"\U0001F600-\U0001F64F"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U0001F1F2"
        u"\U0001F1F4"
        u"\U0001F620"
        u"\u200d"
        u"\u2640-\u2642"
        "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'',text)

def ReduceSentence(text):
    text = RemoveLinks(text, "_link_")
    text = RemoveUsers(text, "_usertag_")
    text = RemoveDotsAndCommas(text, "")
    #text = RemoveEmojis(text)

    return text.lower()

def CleanData(path):
    tweets = {}
    tweet_texts = {}
    with open(path) as json_file:
        data = json.load(json_file)
        for e in data['tweets']:
            #Remove tweets with metadata set to other language code
            try:
                language = e['metadata']['iso_language_code']
            except KeyError:
                language = "da"
                
            if language == "da":
                try:
                    org_tweet_text = e['text']
                except KeyError:
                    org_tweet_text = e['full_text']
                
                #Remove tweets that has less than 10 words - like in the article
                if len(org_tweet_text.split(" ")) >= 10:
                    #Remove links, hashtags, and user mentions
                    tweet_text = RemoveLinksTagsUsers(org_tweet_text)
                    #Remove retweet tag
                    tweet_text = RemoveRT(tweet_text, "")

                    tweet_text = tweet_text.strip()
                    if tweet_text and tweet_text.lower() not in tweet_texts:
                        tweets[org_tweet_text] = e   #Remove duplicates based on tweet text - like in the article
                        tweet_texts[tweet_text.lower()] = True
    
    print(len(tweets))
    return tweets

def GetExperimentData(textfile):
    with open(textfile, encoding="utf-8") as file:
        informative_plus = []
        informative_minus = []
        uninformative = []
        all_data = []
        for l in file:
            #l_split = l.split(", label: ")
            id_text, label = GetIDText(l), GetLabel(l) #l_split[0], l_split[1].strip()
            if label == "not informative":
                uninformative.append(id_text)
            elif label == "informative+":
                informative_plus.append(id_text)
            elif label == "informative-":
                informative_minus.append(id_text)

    random.shuffle(uninformative)
    number_of_uninformative = 500 - (len(informative_plus)+len(informative_minus))
    i = 0
    j = 0
    while True:
        text = GetText(uninformative[j], id_text=True) #uninformative[j].split(", text: ")[1]
        #Remove links, hashtags, and user mentions
        text = RemoveLinksTagsUsers(text)

        if i < number_of_uninformative/2:
            if bool(re.search(r'\d+', text)):
                all_data.append(uninformative[j])
                i += 1
        elif i < number_of_uninformative:
            all_data.append(uninformative[j])
            i += 1
        else:
            break

        j += 1

    all_data.extend(informative_plus)
    all_data.extend(informative_minus)
    random.shuffle(all_data)

    with open("./Data/tweets_for_annotation.txt", 'w' , encoding="utf-8") as save_file:
        for t in all_data:
            save_file.write(t + f", label: ???\n")

def SaveFinalTrainEvalAndTestData(filename):
    with open(filename, 'r', encoding="utf-8") as file:
        labels = []
        tweets = file.readlines()
        ip = 0
        im = 0
        u = 0
        for l in tweets:
            label = GetLabel(l)
            labels.append(label)
            if label == "I+":
                ip += 1
            elif label == "I-":
                im += 1
            elif label == "U":
                u += 1
        X_strat, X = train_test_split(tweets, test_size=0.8, stratify=labels)
        print(f"total: [{ip}, {im}, {u}]")

        with open("./Data/tweets_final_train_eval.txt", 'w', encoding="utf-8") as file:
            ip = 0
            im = 0
            u = 0

            for l in X:
                file.write(l)

                label = GetLabel(l)
                if label == "I+":
                    ip += 1
                elif label == "I-":
                    im += 1
                elif label == "U":
                    u += 1

            print(f"train eval set: [{ip}, {im}, {u}]")

        with open("./Data/tweets_final_endtest.txt", 'w', encoding="utf-8") as file:
            ip = 0
            im = 0
            u = 0
            
            for l in X_strat:
                file.write(l)

                label = GetLabel(l)
                if label == "I+":
                    ip += 1
                elif label == "I-":
                    im += 1
                elif label == "U":
                    u += 1

            print(f"end test set: [{ip}, {im}, {u}]")

def MergeTrainEvalWithAllLabelledTweets():
    train_eval = {}
    final_doc_train = []
    final_doc_test = []
    uninformative = []

    with open("./Data/tweets_final_train_eval.txt", 'r', encoding="utf-8") as file:
        for l in file:
            id = GetID(l)
            train_eval[id] = l
            final_doc_train.append(l)

    with open("./Data/tweets_final_endtest.txt", 'r', encoding="utf-8") as file:
        for l in file:
            id = GetID(l)
            train_eval[id] = l
            final_doc_test.append(l)

    with open("./Data/all_tweets_labelled.txt", 'r', encoding="utf-8") as file:
        u = 0
        for l in file:
            id = GetID(l)
            text = GetText(l)
            label = GetLabel(l)

            if label == "not informative": 
                label = "U"
            elif label == "informative+":
                label = "I+"
            elif label == "informative-":
                label = "I-"

            if id not in train_eval:
                if label == "U": u += 1
                new_label_tweet = f"id: {id}, text: {text}, label: {label}\n"
                uninformative.append(new_label_tweet)

        print(f"total uninformative: {u}")

    X_train, X_test = train_test_split(uninformative, test_size=0.2)

    for train in X_train:
        final_doc_train.append(train)

    for test in X_test:
        final_doc_test.append(test)

    with open("./Data/all_tweets_final_train_eval.txt", 'w', encoding="utf-8") as file:
        random.shuffle(final_doc_train)
        for l in final_doc_train:
            file.write(l)

    with open("./Data/all_tweets_final_endtest.txt", 'w', encoding="utf-8") as file:
        random.shuffle(final_doc_test)
        for l in final_doc_test:
            file.write(l)

def get_embeddings(file, tokens_integer_enc_dict_load=None, ordered_tokens_load=None, max_sentence_length_load=0, bert=False, multilingual=True):
    padding_token = "__padding__"
    unk_token = "_unk_"
    tokens = []
    y = []
    max_sentence_length = 0
    sentences = []
    with open(file, "r", encoding="utf-8") as file:
        for line in file:
            text = GetText(line)
            label = GetLabel(line)
            if label == "I+":
                label = 0   #[1,0,0]
            elif label == "I-":
                label = 1   #[0,1,0]
            elif label == "U":
                label = 2   #[0,0,1]
            text = ReduceSentence(text)
            sentences.append(text)
            words = text.split(" ")
            max_sentence_length = max(max_sentence_length, len(words))
            tokens.append(words)
            y.append(label)

    #Added "__padding__" to increase vocabulary size
    tokens_flattened = sum([[padding_token]] + tokens + [[unk_token]], [])#[item for sublist in tokens for item in sublist]
    pretrained_embs_weights = {}
    
    if tokens_integer_enc_dict_load is not None:
        max_sentence_length = max_sentence_length_load
        ordered_tokens = ordered_tokens_load
        for i,token in enumerate(tokens_flattened):
            if token not in tokens_integer_enc_dict_load:
                tokens_integer_enc_dict_load[token] = tokens_integer_enc_dict_load[unk_token]

        tokens_integer_enc_dict = tokens_integer_enc_dict_load
    else:
        ordered_tokens = pd.factorize(tokens_flattened)[0]
        pretrained_embs = load_pretrained_embeddings()
        tokens_integer_enc_dict = {}
        
        for i,token in enumerate(tokens_flattened):
            if token not in tokens_integer_enc_dict:
                index = ordered_tokens[i]
                tokens_integer_enc_dict[token] = index
                try:
                    pretrained_emb = pretrained_embs[token]
                except:
                    pretrained_emb = [0 for _ in range(400)]
                    sd = np.sqrt(6.0 / 400)
                    for idx in range(400):
                        x = np.float32(random.uniform(-sd, sd))
                        pretrained_emb[idx] = x
                pretrained_embs_weights[index] = " ".join(str(e) for e in pretrained_emb)

        tokens_integer_enc_dict[padding_token] = 0
        tokens_integer_enc_dict[unk_token] = max(ordered_tokens)

    integer_encoded = np.array(ordered_tokens).reshape(len(ordered_tokens), 1)
    vocab_size = integer_encoded.max()+1
    
    enc = OneHotEncoder(sparse=False)
    enc = enc.fit(integer_encoded)
    one_hot_encodings = []
    for sentence in tokens:
        integer_encoded_sentence = []
        for token in sentence:
            integer_encoded_sentence.append(tokens_integer_enc_dict[token])
        integer_encoded_sentence = np.array(integer_encoded_sentence)
        integer_encoded_sentence = integer_encoded_sentence.reshape(len(integer_encoded_sentence), 1)
        ohe = enc.transform(integer_encoded_sentence)
        
        padding_size = max_sentence_length - len(sentence)
        if padding_size > 0:
            padding = [0] * padding_size
            padding = enc.transform(np.array(padding).reshape(padding_size, 1))
            ohe = np.append(ohe, padding, axis=0)
        one_hot_encodings.append(ohe)
    
    final_ohe = []
    for sen in one_hot_encodings:
        new_ohe = []
        for ohe in sen:
            idx = np.argmax(ohe)
            new_ohe.append(idx)
        final_ohe.append(new_ohe)

    for idx,sen in enumerate(final_ohe):
        with open(f"./Data/OneHotData{len(final_ohe)}/onehot{idx}.txt", "w", encoding="utf-8") as file:
            #for ohe in sen:
            save = ";".join(map(str, sen)) + "\n"
            file.write(save)
    with open(f"./Data/OneHotData{len(final_ohe)}/vocab_size.txt", "w", encoding="utf-8") as file:
        file.write(str(vocab_size))

    with open(f"./Data/pretrained_encodings_{len(final_ohe)}.txt", "w", encoding="utf-8") as file:
        strings = list(pretrained_embs_weights.values())
        file.writelines("\n".join(strings))
    
    if bert:
        tokenizer, model = b.getBertModel(multilingual=multilingual)
        if multilingual:
            folder = "BertData"
        else:
            folder = "BertDataDanish"

        if len(final_ohe) < 1000:
            max_length = 106 if folder == "BertData" else 107
        else:
            max_length = 118 if folder == "BertData" else 134

        # Find padding size for dataset
        # contextualized_embeddings = b.getContextualizedEmbeddings(text=sentences, tokenizer=tokenizer, model=model, max_length=0)
        # print(contextualized_embeddings.last_hidden_state.size())
        # max_sentence_length = contextualized_embeddings.last_hidden_state.size(1)

        for idx,sen in enumerate(sentences):
            with open(f"./Data/{folder}{len(final_ohe)}/bert{idx}.txt", "w", encoding="utf-8") as file:
                contextualized_embeddings = b.getContextualizedEmbeddings(text=sen, tokenizer=tokenizer, model=model, max_length=max_length)
                for word in contextualized_embeddings.last_hidden_state.tolist()[0][1:-1]:
                    save = ";".join(map(str, word)) + "\n"
                    file.write(save)

    with open(f"./Data/labels{len(final_ohe)}.txt", "w", encoding="utf-8") as file:
        for idx,label in enumerate(y):
            file.write(f"{label}\n")

    return tokens_integer_enc_dict, ordered_tokens, max_sentence_length

def load_data(file, bert, length, seed, tokens_integer_enc_dict=None, ordered_tokens=None, max_sentence_length=0, multilingual=True):
    # if tokens_integer_enc_dict is None:
    #     tokens_integer_enc_dict, ordered_tokens, max_sentence_length = get_embeddings(file, bert=bert, multilingual=multilingual)
    # else:
    #     get_embeddings(file, tokens_integer_enc_dict, ordered_tokens, max_sentence_length, bert=bert, multilingual=multilingual)

    #Get the embedding size by loading the first "image" and checking the width of it
    if bert:
        if multilingual:
            with open(f"./Data/BertData{length}/bert{0}.txt", "r", encoding="utf-8") as file:
                vocab_size = len(file.readline().strip().split(";"))
        else:
            with open(f"./Data/BertDataDanish{length}/bert{0}.txt", "r", encoding="utf-8") as file:
                vocab_size = len(file.readline().strip().split(";"))
    else:
        with open(f"./Data/OneHotData{length}/vocab_size.txt", "r", encoding="utf-8") as file:
            vocab_size = int(file.readline().strip())
    
    with open(f"./Data/labels{length}.txt", "r", encoding="utf-8") as file:
        y = list(map(int,file.read().splitlines()))

    with open(f"./Data/pretrained_encodings_{length}.txt", "r", encoding="utf-8") as file:
        pretrained_embs = list(file.readlines())
        pretrained_embs = [list(map(float, emb.split(" "))) for emb in pretrained_embs]

    x = np.array(range(0,len(y)))

    return x, y, vocab_size, pretrained_embs, tokens_integer_enc_dict, ordered_tokens, max_sentence_length

def standard_split(file, bert, length, seed, multilingual=True):
    indices, y, vocab_size, pretrained_embs, tokens_integer_enc_dict, ordered_tokens, max_sentence_length = load_data(file, bert, length, seed, multilingual=multilingual)

    x_train, x_test, y_train, y_test = train_test_split(indices, y, test_size=0.2, random_state=seed, stratify=y)

    return x_train, x_test, y_train, y_test, vocab_size, pretrained_embs, tokens_integer_enc_dict, ordered_tokens, max_sentence_length

def load_pretrained_embeddings():
    embs = {}
    with open("./Data/da.txt", "r", encoding="utf8", errors="ignore") as file:
        for l in file:
            vals = l.strip().split(" ")
            word = vals[0]
            emb = vals[1:]
            embs[word] = emb

    return embs