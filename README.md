# NSIETE Návrh projektu

## Motivácia

Vybrali sme si veľmi častú úlohu NLP konkrétne analýzu sentimentu. Na začiatok by sme chcely iba binárnu klasifikáciu recenzií filmov. Tým pádom, že to je pomerne jednoduchá úloha je veľa možností na experimentovanie a popridávanie viacerých rozšírení k tejto úlohe.

## Súvisiaca práca

Yuan, Ye, and You Zhou. "Twitter sentiment analysis with recursive neural networks." CS224D Course Projects (2015).

Zhang, Lei, Shuai Wang, and Bing Liu. "Deep learning for sentiment analysis: A survey." Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery 8.4 (2018): e1253.

## Datasets

Rozhodli sme sa použiť nasledujúci dataset https://nlp.stanford.edu/sentiment/code.html ktorá obsahuje 25,000 označkovaných vysoko polarizovaných recenzíí a 25,000 neoznačnených recenzíí.

## Vysoko úrovňové riešenie

Vyskúšame viacero rekurentných sietí ako LSTM, GRU a jednoduché RNN na predikciu sentimentov. Navyše použijeme embedingy slova alebo znakov.

### Možné rozšírenia
- Jazykové modely
- Transfer learning (pretrénované jazykové modely, word embeddings)
- Mechanizmus pozornosti
- Adversary networks
- Konvolučné siete
