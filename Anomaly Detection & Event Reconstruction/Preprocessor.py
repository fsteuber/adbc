#! The following code is intented only for illustration and cannot be run as a single script.

import lzma
import logging
import json
from os import listdir
from os.path import join, isfile
import spacy
import re
from datetime import datetime
import redis


def process_files(file_list):
    # set up some NLP tools before processing
    nlp = spacy.load('en_core_web_sm', disable=["parser", "textcat", "custom"])
    redis_cli = redis.Redis(host="localhost", port=6379, db=0)

    # start processing
    for path_to_file in file_list:
        print("Processing file {}".format(path_to_file))
        tweets = []
        texts = []

        with lzma.open(path_to_file, "rb") as file:
            for tweet in file.readlines():
                tweet = json.loads(tweet)

                if not tweet.get("lang") == "en":  # restrict on english tweets to avoid encoding problems
                    continue

                text = re.sub(r'[\s]+', ' ',  # remove multiple space characters
                        re.sub(r'\b\w{1,2}\b', '',  # remove tokens with less than 3 characters
                        re.sub(r'[^A-z ]', '',  # remove non latin characters
                        re.sub(r'[\n\t]', ' ',  # remove linebreaks and tabs
                        re.sub(r'@\w+', '',  # remove @ annotations
                        re.sub(r'https?:\/\/.*[\r\n]*', '', tweet.get("text"))))))).strip()  # remove urls

                if len(text.split(" ")) >= 3:  # tweet needs to have a minimum amount of textual content
                    tweets.append(tweet)  # temporarily save the tweet
                    texts.append(text)  # prepare for batch processing

                # end for tweet in file.readlines()
            # end with lzma.open(tweet_file, "rb") as file:

            # process files
            for tweet, doc in zip(tweets, nlp.pipe(texts, n_process=-1, batch_size=1000)):  # uses all cores
                # Stopword Removal and Lemmatization
                tweet["tokens"] = [n.lemma_ for n in doc if not n.is_stop]

                # Strong Context consists of Nouns, Named Entities and Hashtags
                strong_context = [n.text for n in doc if n.pos_ in ("NOUN", "PROPN")]

                for entity in doc.ents:
                    if entity.label_ in ("PERSON", "ORG", "GPE"):
                        strong_context.append(entity.text)

                strong_context.extend([x.get("text") for x in tweet.get("entities").get("hashtags")])  # hashtags

                # Weak Context consists of remaining tokens
                weak_context = [x for x in tweet.get("tokens") if x not in strong_context]

                # Restrict on Tweets with valuable content
                if len(strong_context) == 0:
                    continue

                # Remove Word Duplicates
                strong_context = list(set(strong_context))
                weak_context = list(set(weak_context))

                # Put Context Tokens into Tweet
                tweet["context_strong"] = strong_context
                tweet["context_weak"] = weak_context

                ts = int(datetime.timestamp(datetime.strptime(tweet.get("created_at"), "%a %b %d %H:%M:%S %z %Y")))
                tid = int(tweet.get("id_str"))

                # Pass Tweet and Tokens to Queues
                for token in strong_context:  # serves as input for convolutional queue
                    convQ.append((token, tid, ts))
                    conv_dict = {'token': token, 'ts': ts, 'tid': tid}
                    redis_cli.rpush("ADBC-ConvolutionalQueue", str(json.dumps(conv_dict)))

                expQ.append((tid, tweet))  # serves as input for expiring queue
                redis_cli.append(str(tid), str(json.dumps(tweet)).encode("utf-8"))
                redis_cli.expire(str(tid), 60)
        # end for path_to_file in file_list:
    # end def process_file()


"""
def run(q_in, q_conv, q_expire):
  
                # check for original tweet. if original_tweet stays None, then t0 = t1
                original_tweet = None
                if tweet.get("retweeted_status") is not None:
                    original_tweet = tweet.get("retweeted_status")
                elif tweet.get("quoted_status") is not None:
                    original_tweet = tweet.get("quoted_status")

                # extract information from t0 for prediction purposes in log scale
                if original_tweet is not None:
                    origin_friends = np.log10(original_tweet.get("user").get("friends") + 1)
                    origin_followers = np.log10(original_tweet.get("user").get("followers") + 1)
                    origin_retweets = np.log10(original_tweet.get("retweet_count") + 1)
                    origin_favorites = np.log10(original_tweet.get("favorite_count") + 1)
"""


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        filename="preprocessor.log")

    path_to_files = "/home/../tweets/"
    files = sorted([join(path_to_files) + f for f in listdir(path_to_files) if isfile(join(path_to_files) + f)])
    print("Files to process: {}".format(files))

    convQ = []
    expQ = []

    process_files(files)

    print("{} tweets in ExpQueue", len(expQ))
    print("{} tokens in ConvQueue", len(convQ))
