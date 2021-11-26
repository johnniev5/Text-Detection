#!/usr/bin/env python
# coding: utf-8

import sys
from predictor import Classifier

def data2vec(model_type, text):
	clf = Classifier(model_type, text)
	if sys.argv[1] == 'shorten_urls':
		texts_list = clf.texts.split()
		instances = []
		if len(texts_list) > 1:
			for text in texts_list:
				instances.append(clf.predictOne(text))
		else:
			instances.append(clf.predictOne("".join(clf.texts)))
	elif sys.argv[1] == 'wubao_chinese':
		if len(clf.texts) != 0:
			instances = [clf.predictOne(clf.texts)]
	print (instances)


if __name__ == "__main__":
	data2vec(sys.argv[1], sys.argv[2])

