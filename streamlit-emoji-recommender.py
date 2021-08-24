import streamlit as st
import re
import os
import numpy as np
import pandas as pd
import re
from tqdm.auto import tqdm
from datasets import load_dataset
from datasets import load_metric
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoModel
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class EmojiRecommender:
    def __init__(self):
        self.getModel()
    
    def getModel(self):
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
        self.model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    
    def getMeanTokensSentence(self, sentence):
        sentence = sentence.lower()
        sentence = re.sub('[^a-z]+', ' ', sentence)
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(sentence)
        sentence = [w for w in word_tokens if not w.lower() in stop_words]
        sentence = ' '.join(sentence)
        return obj.getMeanTokens([sentence])
    
    def getMeanTokens(self, sentences):
        self.getTokens(sentences)
        self.getEmbedding()
        return self.getMeanValue()
    
    def getTokens(self, sentences):
        self.tokens = {'input_ids': [], 'attention_mask': []}

        for sentence in sentences:
            new_tokens = self.tokenizer.encode_plus(sentence, max_length=128,
                                               truncation=True, padding='max_length',
                                               return_tensors='pt')
            self.tokens['input_ids'].append(new_tokens['input_ids'][0])
            self.tokens['attention_mask'].append(new_tokens['attention_mask'][0])

        self.tokens['input_ids'] = torch.stack(self.tokens['input_ids'])
        self.tokens['attention_mask'] = torch.stack(self.tokens['attention_mask'])
    
    def getEmbedding(self):
        outputs = self.model(**self.tokens)
        self.embeddings = outputs.last_hidden_state
    
    def getMeanValue(self):
        attention_mask = self.tokens['attention_mask']
        mask = attention_mask.unsqueeze(-1).expand(self.embeddings.size()).float()
        masked_embeddings = self.embeddings * mask
        summed = torch.sum(masked_embeddings, 1)
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)
        self.mean_pooled = summed / summed_mask
        self.mean_pooled = self.mean_pooled.detach().numpy()
        return self.mean_pooled

    def getSimilarity(self, sentence_tokens, mean_tokens):
        similarity = cosine_similarity([sentence_tokens],mean_tokens)
        return similarity
    
    def getEmoji(self, sentence):
        all_tokens = torch.load('checkpoint/token-all.pt')
        sentence_token = obj.getMeanTokensSentence(sentence)
        similarity = obj.getSimilarity(sentence_token[0], all_tokens)
        indices = (-similarity[0]).argsort()[:5]
        emoji_df = pd.read_csv("data/emoji-data.csv")
        emoji_list = { 'emoji':[], 'description':[] }
        for j in indices:
            emoji_list['emoji'].append(emoji_df['emoji'][j])
            emoji_list['description'].append(emoji_df['description'][j])
        return emoji_list


obj = EmojiRecommender()

st.title("Emoji Prediction")
st.subheader("Write a comment")
text = st.text_input('Enter text')
st.subheader("Emojis")

if len(text) > 0:
    emoji_list = obj.getEmoji(text)
    output = ' '

    for i in range(len(emoji_list['emoji'])):
        output += emoji_list['emoji'][i]
    st.write(output)

    
    st.subheader("Developer mode:")
    st.write("Predicted emoji list")
    st.write(emoji_list)