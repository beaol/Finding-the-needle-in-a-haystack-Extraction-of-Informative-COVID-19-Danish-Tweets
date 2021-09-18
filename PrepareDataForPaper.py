from DataProcessing import GetID, GetText, GetLabel

new_file = []

with open("./Data/all_tweets_labelled.txt", 'r', encoding="utf-8") as file:
    for l in file:
        id = GetID(l)
        label = GetLabel(l)

        if label == "not informative": 
            label = "U"
        elif label == "informative+":
            label = "I+"
        elif label == "informative-":
            label = "I-"

        new_file.append(f"id: {id}, label: {label}\n")

with open("./Data/all_tweets_labelled_paper.txt", 'w', encoding="utf-8") as file:
    for l in new_file:
        file.write(l)