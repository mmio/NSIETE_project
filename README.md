# NSIETE Návrh projektu

Dominik Vasko, Adrian Vyskoč

## Motivácia

Našou témou bude veľmi častá úloha v oblasti NLP (Natural language processing), konkrétne analýza sentimentu. Na začiatok by sme chceli natrénovať modely na binárnu klasifikáciou používateľských recenzií k filmom. Keďže klasifikácia recenzií je pomerne jednoduchá úloha, poskytuje nám široké možnosti experimentovania a priestor na pridanie rôznych rozšírení k zadaniu.

Analýza sentimentu (známa aj ako prieskum názorov, opinion mining) je oblasťou aktívneho výskumu v oblasti spracovania prirodzeného jazyka. Zameriava sa na identifikáciu, získavanie a organizovanie sentimentov z textov generovaných používateľmi v sociálnych sieťach, blogoch alebo recenziách produktov vie nám povedať, čo si ľudia myslia o danom produkte. Práve kvôli tomu má široké využitie hlavne pre firmy, ktoré sa snažia prispôsobiť svôj produkt pre zákazníkov alebo sa sústreďujú na user experiance.

Analyzovať tisíce recenzií manuálne je časovo aj finančne náročné. Na pomoc prichádza AI a jeho podoblasť NLP, ktorá sa už viaceré rokov zaoberá analýzou sentimentu. Oproti iným metódam, neurónové siete majú tú vyhodu, že sa dokážu pomerne jednuch prispôsobiť danej doméne. Neorbia im problémy ani skratky alebo slang. Navyše sa nám téma javí ako zaujímavá z dôvodu nášho záujmu o filmy, práca na tejto téme je pre nás o to zábavnejšia.

## Súvisiaca práca

Veľmi jednoduchým riešením zadanie je použítie rekurentných sietí, ktoré sú na túto úlohu najvhodnejšie.
- Yuan, Ye, and You Zhou. "Twitter sentiment analysis with recursive neural networks." CS224D Course Projects (2015).

Existuje však mnoho rozšírení, ktoré je možné aplikovať aj v našej práci napr. mechanizmy pozornosti alebo konvolučné siete.
- Zhang, Lei, Shuai Wang, and Bing Liu. "Deep learning for sentiment analysis: A survey." Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery 8.4 (2018): e1253.
- Tang, Duyu, Bing Qin, and Ting Liu. "Deep learning for sentiment analysis: successful approaches and future challenges." Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery 5.6 (2015): 292-303.

## Datasety

Na prácu na našej téme sme sa rozhodli použiť nasledujúci dataset recenzíí filmov z websídla IMDb https://nlp.stanford.edu/sentiment/code.html . Tento dataset obsahuje 25 000 označkovaných vysoko polarizovaných recenzíí a 25 000 neoznačnených recenzíí. Keďže dát je pomerne veľa, nemal by vzniknúť problém pri trénovaní modelu.

## Vysoko úrovňové riešenie

Vyskúšame viacero architektúr rekurentných sietí ako LSTM, GRU a jednoduché RNN na predikciu sentimentov. Navyše použijeme embedingy slova.

### Možné rozšírenia
- Jazykové modely
- Transfer learning (pretrénované jazykové modely, word embeddings)
- Mechanizmus pozornosti
- Adversary networks
- Konvolučné siete
