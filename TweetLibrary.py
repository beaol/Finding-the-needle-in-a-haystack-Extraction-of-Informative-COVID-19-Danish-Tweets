import json
from twitter import OAuth, Twitter
from twitter.api import TwitterHTTPError

def GetTweetFromId(id):
    credentials = json.load(open("./benjamin_credentials.json"))
    CONSUMER_KEY=credentials["CONSUMER_KEY"]
    CONSUMER_SECRET=credentials["CONSUMER_SECRET"]
    ACCESS_TOKEN=credentials["ACCESS_TOKEN"]
    ACCESS_TOKEN_SECRET=credentials["ACCESS_TOKEN_SECRET"]

    my_authentication = OAuth(ACCESS_TOKEN, ACCESS_TOKEN_SECRET, CONSUMER_KEY, CONSUMER_SECRET)
    twitter = Twitter(auth=my_authentication)

    try:
        tweet = twitter.statuses.show(_id=id, _method='GET', tweet_mode='extended')
        return (tweet, True)
    except TwitterHTTPError:
        return (object(), False)

def replace_truncated_tweets(filename, tweets):
    updated_tweets = {"tweets": []}
    list_tweets = list(tweets.items())
    for tweet in list_tweets:
        t = tweet[1]
        if t['truncated']:
            (full_tweet, bool) = GetTweetFromId(t['id'])
            if bool: updated_tweets["tweets"].append(full_tweet)

            # I decided to remove truncated tweets that could not be found because
            # it is sometimes hard to determine if a tweet is informative or not
            # without seeing the full text

            #else: updated_tweets["tweets"].append(t)
        else:
            updated_tweets["tweets"].append(t)

    file = open(f"./{filename}_nontruncated.json", 'a')
    json.dump(updated_tweets, file)
    file.close()