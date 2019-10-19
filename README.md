# NSIETE Project Proposal

## Motivation

We picked a common supervised NLP task namely sentiment analysis. For starters we consider only binary classification of movie review. This is a very common task for NLP, which allows for experimentation with different models.

## Related Work

Yuan, Ye, and You Zhou. "Twitter sentiment analysis with recursive neural networks." CS224D Course Projects (2015).

Zhang, Lei, Shuai Wang, and Bing Liu. "Deep learning for sentiment analysis: A survey." Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery 8.4 (2018): e1253.

## Datasets

We intend to use the following dataset from https://nlp.stanford.edu/sentiment/code.html which contain 25,000 labeled and 25,000 unlabeled higly polarized movie reviews.

## High-Level Solution Proposal

We will try multiple recurrent network architectures LSTM, GRU and simple RNN to predict the sentiment of reviews. In addition we will also use simple word or character level embeddings.

### Possible extensions
- Language models
- Transfer learning (pre trained language models, word embeddings)
- Attention mechanism
- Adversary networks
- Use of Convolutional networks
