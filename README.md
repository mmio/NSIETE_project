# NSIETE Návrh projektu

## Motivácia

Našou témou bude veľmi častá úloha v oblasti NLP (Natural language processing) konkrétnejšie zameraná na analýzu sentimentu. Na začiatok by sme chceli začať s binárnou klasifikáciou používateľských recenzií k jednotlivým filmom z IMDb. Keďže klasifikácia používateľskýchií je pomerne jednoduchá úloha, poskytuje nám široké možnosti experimentovania a priestor na pridávanie viacerých rozšírení k tejto téme. 

Téma sa nám javi ako zaujímava z dôvodu nášho vysokého záujmu o folmovú tvorbu. Keďže táto oblasť je zároveň našim koníčkom, tak práca na tejto téme je pre nás o to zábavnejšia.

## Analýza sentimentu

Analýza sentimentu (známa aj ako prieskum názorov) je oblasťou aktívneho výskumu v oblasti spracovania prirodzeného jazyka. Zameriava sa na identifikáciu, získavanie a organizovanie sentimentov z textov generovaných používateľmi v sociálnych sieťach, blogoch alebo recenziách produktov. Mnoho štúdií v literatúre využíva prístupy strojového učenia na riešenie úloh analýzy sentimentu z rôznych perspektív za posledných 15 rokov. 

## Súvisiaca práca

Yuan, Ye, and You Zhou. "Twitter sentiment analysis with recursive neural networks." CS224D Course Projects (2015).

Zhang, Lei, Shuai Wang, and Bing Liu. "Deep learning for sentiment analysis: A survey." Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery 8.4 (2018): e1253.

Tang, Duyu, Bing Qin, and Ting Liu. "Deep learning for sentiment analysis: successful approaches and future challenges." Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery 5.6 (2015): 292-303.

## Datasets

Na prácu na našej téme sme sa rozhodli použiť nasledujúci dataset https://nlp.stanford.edu/sentiment/code.html . Tento dataset obsahuje 25 000 označkovaných vysoko polarizovaných recenzíí a 25 000 neoznačnených recenzíí. Keďže dát je pomerne veľa, nemal by vzniknúť problém pri trénovaní modelu. 

## Vysoko úrovňové riešenie

Vyskúšame viacero rekurentných sietí ako LSTM, GRU a jednoduché RNN na predikciu sentimentov. Navyše použijeme embedingy slova alebo znakov.

### Možné rozšírenia
- Jazykové modely
- Transfer learning (pretrénované jazykové modely, word embeddings)
- Mechanizmus pozornosti
- Adversary networks
- Konvolučné siete
