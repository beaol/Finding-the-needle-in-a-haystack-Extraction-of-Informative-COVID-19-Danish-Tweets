from tkinter import * 
from tkinter.ttk import *
from tkinter import messagebox
import json

def Labelling(tweets):
    if(isinstance(tweets, dict)): list_tweets = list(tweets.items())
    else: list_tweets = tweets
    
    all_labelled_tweets = []
    informative_tweets_plus = []
    non_informative_tweets = []
    informative_tweets_minus = []
    # non_informative_tweets_minus = []
    ids = []

    master = Tk() 
    master.title = "Tweet labelling"
    master.geometry("720x480") 
    t = StringVar()
    t.set(list_tweets[0][0])
    count = IntVar()
    count.set(0)
    count_informative_plus = IntVar()
    count_informative_plus.set(0)
    count_informative_minus = IntVar()
    count_informative_minus.set(0)
    count_not_informative = IntVar()
    count_not_informative.set(0)
    label_tweet = Label(master,  textvariable = t, font=("Arial", 25), wraplength=1000) 
    label_tweet.pack(side = TOP, pady = 10) 
    label_count_tweets = Label(master,  textvariable = count)
    label_count_tweets.pack(side = TOP, pady = 10)
    label_count_informative_tweets_plus = Label(master,  textvariable = count_informative_plus)
    label_count_informative_tweets_plus.pack(side = LEFT, pady = 10)
    label_count_informative_tweets_minus = Label(master,  textvariable = count_not_informative)
    label_count_informative_tweets_minus.pack(side = RIGHT, pady = 10)

    def informativePlus(event=None):
        tweet = list_tweets[count.get()][1]
        informative_tweets_plus.append(tweet)
        add_tweet(tweet, "informative+")
        add_id(tweet)
        next_tweet()
        count_informative_plus.set(count_informative_plus.get()+1)

    def not_informative(event=None):
        tweet = list_tweets[count.get()][1]
        non_informative_tweets.append(tweet)
        add_tweet(tweet, "not informative")
        add_id(tweet)
        next_tweet()
        count_not_informative.set(count_not_informative.get()+1)

    def informativeMinus(event=None):
        tweet = list_tweets[count.get()][1]
        informative_tweets_minus.append(tweet)
        add_tweet(tweet, "informative-")
        add_id(tweet)
        next_tweet()
        count_informative_minus.set(count_informative_minus.get()+1)

    # def not_informativeMinus(event=None):
    #     tweet = list_tweets[count.get()][1]
    #     non_informative_tweets_minus.append(tweet)
    #     add_tweet(tweet, "not informative-")
    #     add_id(tweet)
    #     next_tweet()

    def add_tweet(tweet, label):
        all_labelled_tweets.append((tweet, label))
    
    def add_id(tweet):
        ids.append(str(tweet['id']) + "\n")

    def next_tweet(event=None):
        count.set(count.get()+1)
        t.set(list_tweets[count.get()][0])

    def auto_skip(event=None):
        try:
            with open("./Data/ids.txt") as ids:
                curr_id = str(list_tweets[count.get()][1]['id'])
                
                while True:
                    found = False
                    for id in ids:
                        if id.strip() == curr_id:
                            found = True
                            next_tweet()
                            curr_id = str(list_tweets[count.get()][1]['id'])
                    if not found:
                        break
        except IOError:
            print("Failed to open file!")

    def replay_data(event=None):
        with open("./Data/replay_data.txt", 'r') as r_data:
            for action in r_data:
                action = action.strip()
                
                if action == "left":
                    informativePlus()
                elif action == "a":
                    informativeMinus()
                elif action == "right":
                    not_informative()
                # elif action == "d":
                #     not_informativeMinus()

        force_close()

    def on_closing(event=None):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            force_close()

    def force_close():
        save_to_file()
        master.destroy()

    def save_to_file():
        if(len(informative_tweets_plus) > 0 or len(non_informative_tweets) > 0):
            file1 = open("./Data/dk_informative+_tweets.json", 'a')
            file2 = open("./Data/dk_non_informative_tweets.json", 'a')
            file3 = open("./Data/dk_informative-_tweets.json", 'a')
            # file4 = open("./Data/dk_non_informative-_tweets.json", 'a')
            file5 = open("./Data/ids.txt", 'a')
            file6 = open("./Data/all_tweets.json", 'a')
            file7 = open("./Data/all_tweets_labelled.txt", 'a', encoding="utf-8")
            file8 = open("./Data/all_tweets_notlabelled.txt", 'a', encoding="utf-8")
            file9 = open("./Data/replay_data.txt", 'a')
            json.dump(informative_tweets_plus, file1)
            json.dump(non_informative_tweets, file2)
            json.dump(informative_tweets_minus, file3)
            # json.dump(non_informative_tweets_minus, file4)
            json.dump([t for (t,l) in all_labelled_tweets], file6)
            file5.writelines(ids)
            
            label_mappings = {
                'informative+': "left",
                'informative-': "a",
                'not informative': "right",
                # 'not informative-': "d"
            }

            for (t,l) in all_labelled_tweets:
                try:
                    tweet_text = t['text']
                except KeyError:
                    tweet_text = t['full_text']
                tweet_text = tweet_text.replace("\n", "")
                line = f"id: {t['id']}, text: {tweet_text}"
                file7.write(line + f", label: {l}\n")
                file8.write(line+"\n")
                file9.write(label_mappings[l]+"\n")
            file1.close()
            file2.close()
            file3.close()
            # file4.close()
            file5.close()
            file6.close()
            file7.close()
            file8.close()
            file9.close()
            messagebox.showinfo("","Saved informative tweets to: dk_informative_tweets.json and dk_non_informative_tweets.json")
        else:
            messagebox.showinfo("","No file was saved because no informative/non-informative tweets were selected")

    btn_informative_plus = Button(master,  
                text ="Informative+",  
                command = informativePlus)
    btn_informative_plus.pack(side = LEFT, pady = 10) 
    
    btn_notinformative_plus = Button(master,  
                text ="Not Informative",  
                command = not_informative)
    btn_notinformative_plus.pack(side = RIGHT, pady = 10)

    label_count_uninformative_tweets = Label(master,  textvariable = count_informative_minus)
    label_count_uninformative_tweets.pack(side = LEFT, pady = 10)

    btn_informative_minus = Button(master,  
                text ="Informative-",  
                command = informativeMinus)
    btn_informative_minus.pack(side = LEFT, pady = 10) 
    
    # btn_notinformative_minus = Button(master,  
    #             text ="Not Informative-",  
    #             command = not_informativeMinus)
    # btn_notinformative_minus.pack(side = RIGHT, pady = 10) 

    btn_autoskip = Button(master,  
                text ="Start auto skip",  
                command = auto_skip)
    btn_autoskip.pack(side = TOP, pady = 10) 

    master.bind("<Left>", informativePlus)
    master.bind("<Right>", not_informative)
    master.bind("<a>", informativeMinus)
    # master.bind("<d>", not_informativeMinus)
    master.bind("<space>", auto_skip)
    master.bind("<Escape>", on_closing)
    master.bind("r", replay_data)
    master.protocol("WM_DELETE_WINDOW", on_closing)
    mainloop()