# Emoji Recommender 🦾 🧠
---

This library provides smart emoji recommendation on the input text using.
Built by [Kamal Kant](https:kama1kant.com/)

---
##  How it works
### Dataset
The dataset consists of list of emojis & their name/description.

### Model
Emoji's description is tokenized using BERT & stored in a file.

### Inference
The input text is first normalized by removing the stopwords. Mean of the tokenized values is then taken for the remaining text. The mean tokenized vector of the text is then compared with the tokenized  value of each emoji. Using cosine similarity we then find the top five nearest valued emoji.

---
## How to run
- Download the repository
- Run the following command
```py
streamlit run streamlit-emoji-recommender.py
```
---
## Libraries used
- Pytorch
- Huggingface Transformers
- BERT
- Pandas
- NumPy
---
## Contact
If you have any questions or feedback, feel free to reach out to me at <kamalkant.k3@gmail.com>