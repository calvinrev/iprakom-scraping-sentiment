import pandas as pd
from transformers import pipeline

class Sentiment():
    def __init__(self):
        self.model = 'w11wo/indonesian-roberta-base-sentiment-classifier'
        self.classifier = pipeline("sentiment-analysis", model=self.model)

    def getSentimentLabel(self, text):
        return self.classifier(text)[0]['label']
    
    def getSentimentScore(self, text):
        return self.classifier(text)[1]['score']