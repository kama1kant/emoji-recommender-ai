# Emoji Recommender 🦾 🧠

This library provides smart emoji recommendations on the input text using novel Natural Language Processing methods.

Built by [Kamal Kant](https://kama1kant.com/)

##  How it works
### Dataset
The dataset consists of a list of emojis & their names/descriptions.

### Model
This project use [BERT](https://arxiv.org/abs/1810.04805) which is a state-of-the-art language model for NLP.
Emoji's description is tokenized using BERT & stored in a file.

### Inference
The input text is first normalized by removing the stopwords. The mean of the tokenized values is then taken for the remaining text. The mean tokenized vector of the text is then compared with the tokenized value of each emoji. Using cosine similarity we then find the top five nearest valued emoji.

## How to run
- Download the repository
- Run the following command
```py
streamlit run streamlit-emoji-recommender.py
```

## Libraries used
- Pytorch
- Huggingface Transformers
- BERT
- NLTK
- Pandas
- NumPy

## Contact
If you have any questions or feedback, feel free to reach out to me at <kamalkant.k3@gmail.com>