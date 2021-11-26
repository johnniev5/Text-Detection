#!/usr/bin/env python
# coding: utf-8

import sys
import json
import requests
from predictor import Classifier


def getReuslts(model_type, text):
	if model_type == "shorten_urls":
		clf = Classifier("shorten_urls", text)
		urls_list = clf.urls.split()
		if len(urls_list) != 0:
			texts_list = clf.texts.split()
			instances = []
			if len(texts_list) > 1:
				for text in texts_list:
					instances.append(clf.predictOne(text))
			else:
					instances.append(clf.predictOne("".join(clf.texts)))
			data = json.dumps({"instances": instances})
			headers = {"content-type": "application/json"}
			json_response = requests.post('http://172.17.0.1:8501/v1/models/url:predict', data=data, headers=headers)
			predictions = json.loads(json_response.text)['predictions']
			results = []
			if len(predictions) > 1:
				for proba, text, url in zip(predictions, texts_list, urls_list):
					prob = proba[0]
					pred = ''.join(['白' if round(prob) == 0 else '黑'])
					score = [1 - prob if prob <= 0.5 else prob][0]
					results.append({'orginal_' + str(clf._field): url, 'long_' + str(clf._field): text, "pred": str(pred), "score": round(score, 7)})
			else:
				prob = predictions[0][0]
				pred = ''.join(['白' if round(prob) == 0 else '黑'])
				score = [1 - prob if prob <= 0.5 else prob][0]
				results.append({'orginal_' + str(clf._field): "".join(urls_list), 'long_' + str(clf._field): "".join(texts_list), "pred": str(pred), "score": round(score, 7)})
	elif model_type == "wubao_chinese":
		clf = Classifier("wubao_chinese", text)
		if len(clf.texts) != 0:
			data = json.dumps({"instances": [clf.predictOne(clf.texts)]})
			headers = {"content-type": "application/json"}
			json_response = requests.post('http://172.17.0.1:8501/v1/models/wubao:predict', data=data, headers=headers)
			predictions = json.loads(json_response.text)['predictions']
			results = []
			prob = predictions[0][0]
			pred = ''.join(['正常' if round(prob) == 0 else '五毛'])
			score = [1 - prob if prob <= 0.5 else prob][0]
			results.append({'text': text, "pred": str(pred), "score": round(score, 7)})

	print ({"result": results})


if __name__ == "__main__":
	getReuslts(sys.argv[1], sys.argv[2])
