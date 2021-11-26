#!/usr/bin/env python
# coding: utf-8

import os
import re
import sys
import time
import pickle
import numpy as np
import pandas as pd
from settings import config
from unshortenit import UnshortenIt
from tensorflow.keras import models, preprocessing


def getUrl(text):
    url_pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    try:
        return " ".join(re.findall(url_pattern, text))
    except:
        pass


def getLong(url):
    u_list = []
    symbol_pattern ='http[s]?://(([a-z0-9\-\_]*){1,63}\.?){1,255}@(([a-z0-9\-\_]*){1,63}\.?){1,255}/\w+'
    unshortener = UnshortenIt()
    urls = str(url).split()
    for url in urls:
        if len(url) <= 50:
            if '@' in url:
                if re.search('\.[a-z]+@', url):
                    try:
                        u = re.match(symbol_pattern, url).group()
                        u = u.split(':')[0] + "://" + u.split('@')[1]
                    except:
                        u_list.append(url)
                    else:
                        try:
                            uri = unshortener.unshorten(u)
                        except:
                            u_list.append(u)
                        else:
                            if len(uri) > 0:
                                u_list.append(uri)
                            else:
                                u_list.append(u)
                else:
                    u_list.append(url)
            else:
                try:
                    uri = unshortener.unshorten(url)
                except:
                    u_list.append(url)
                else:
                    if len(uri) > 0:
                        u_list.append(uri)
                    else:
                        u_list.append(url)
        else:
            u_list.append(url)

    return " ".join(u_list)


def parseText(text):
    urls = getUrl(text)
    if len(urls) != 0:
        for url in urls.split():
            text = re.sub(re.escape('\n'), '', text)
            text = re.sub(re.escape(url), '', text)
        text = re.sub('[#@][a-zA-Z\u4E00-\u9FA5_]+\s*', '', text).replace('&lt;', '').replace('&gt;', ' ').replace('@!', '').replace('@&', '').strip()
        text = text.replace('\n', '')

    return text



class Classifier(object):
    def __init__(self, model_type, text):
        self.model_type = model_type
        self.text = text
        self.tokenizer, _, self.seq_maxlen = self.loadModel(self.model_type)
        self.texts = self.loadData(self.text)

    def loadModel(self, model_type):
        with open(config["token_paths"][model_type], "rb") as f:
            tokenizer, vocab_size, seq_maxlen = pickle.load(f)

        return tokenizer, vocab_size, seq_maxlen

    def loadData(self, text):
        if "url" in self.model_type:
            urls = getUrl(text)
            texts = getLong(urls)
            return urls, texts
        elif 'wubao' in self.model_type:
            text = parseText(text)
            return text

    def predictOne(self, value):
        if "url" in self.model_type:
            if value.startswith("http"):
                url = value.split("://")[1]
                texts = self.tokenizer.texts_to_sequences(url)
        elif 'wubao' in self.model_type:
                texts = self.tokenizer.texts_to_sequences(value)
        text = np.array([text[0] for text in texts]).reshape(1, -1)
        x = preprocessing.sequence.pad_sequences(
            text, maxlen=self.seq_maxlen, padding="post", truncating="post"
        )

        return x.flatten().tolist()

    def loadFile(self, file):
        if os.path.splitext(file)[1][1:] != "csv":
            print("only csv format accepted, please change it!")
        else:
            data = pd.read_csv(file, encoding="utf-8")

        return data
        
    def predictFile(self, file):
        output_path = config["output"]["path"]

        testData = self.loadFile(file)
        if "url" in self.model_type:
            values = testData['url']
        elif 'wubao' in self.model_type:
            values = testData['tweet']
        texts = self.tokenizer.texts_to_sequences(values)
        X = preprocessing.sequence.pad_sequences(
            texts, maxlen=self.seq_maxlen, padding="post", truncating="post"
        )
        y_pred_proba = self.model.predict(X)
        y_pred = y_pred_proba.round()

        testData["pred"] = y_pred
        testData["score"] = [(1 - y)[0] if y <= 0.5 else y[0] for y in y_pred_proba]
        output_file = "{}/{}_{}_{}.csv".format(output_path,
                                               config["output"]["prefix"],
                                               file.rsplit("/", 2)[-1].split(".")[0],
                                               str(time.strftime("%Y-%m-%d",
                                               time.localtime(time.time()))))
        testData.to_csv(output_file, index=False)

        return output_file
