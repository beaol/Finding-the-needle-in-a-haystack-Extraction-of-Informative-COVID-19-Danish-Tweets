import sklearn.metrics

def GetLabelsForExperimentalTweets():
    ids_to_labels = {}

    with open("./Data/all_tweets_labelled.txt", encoding="utf-8") as file:
        for t in file:
            id_label = t.split(", text: ")
            id = id_label[0].split("id: ")[1]
            label = id_label[1].split(", label: ")[1].strip()
            ids_to_labels[id] = label

    new_data = []
    with open("./Data/tweets_for_annotation.txt", encoding="utf-8") as file:
        for t in file:
            id_label = t.split(", text: ")
            id = id_label[0].split("id: ")[1]
            label = ids_to_labels[id]
            
            if label == "not informative":
                label = "U"
            elif label == "informative+":
                label = "I+"
            elif label == "informative-":
                label = "I-"

            new_line = t[:-4] + label + "\n"
            new_data.append(new_line)

    with open("./Data/tweets_for_annotation_labelled.txt", "w", encoding="utf-8") as file:
        for t in new_data:
            file.write(t)

def CalculateCohensKappaScore():
    ids_to_labels1 = {}
    ids_to_labels2 = {}

    with open("./Data/tweets_for_annotation_labelled.txt", encoding="utf-8") as file:
        for t in file:
            id_label = t.split(", text: ")
            id = id_label[0].split("id: ")[1]
            label = id_label[1].split(", label: ")[1].strip()
            ids_to_labels1[id] = label
            
    disagree = 0
    disagree_informativeplus_uninformative = 0

    with open("./Data/tweets_for_annotation - annotated.txt", encoding="utf-8") as file:
        for t in file:
            id_label = t.split(", text: ")
            id = id_label[0].split("id: ")[1]
            label = id_label[1].split(", label: ")[1].strip()
            ids_to_labels2[id] = label

            label1 = ids_to_labels1[id]
            if label1 != label:
                print(f"id: {id}, label1: {label1}, label2: {label}")
                disagree += 1
                if (label == "U" and label1 == "I+") or (label == "I+" and label1 == "U"):
                    disagree_informativeplus_uninformative += 1
                
    print(f"Disagreement on all labels: {disagree}")
    print(f"Disagreement where one of the labels is labelled U and the other is labelled I+: {disagree_informativeplus_uninformative}")
    cohen = sklearn.metrics.cohen_kappa_score(y1=list(ids_to_labels1.values()), y2=list(ids_to_labels2.values()), labels=["U","I+","I-"])
    print(f"Cohen's kappa score: {cohen}")
    return cohen

CalculateCohensKappaScore()