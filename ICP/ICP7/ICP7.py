from bs4 import BeautifulSoup
import requests
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import wordpunct_tokenize, pos_tag, ne_chunk
from nltk.util import ngrams

url = "https://en.wikipedia.org/wiki/Python_(programming_language)"

response = requests.get(url)

data = response.text

soup = BeautifulSoup(data, "html.parser")

textout = soup.get_text()
textout2 = textout[1:1000]
f = open('input.txt','w+',encoding="utf-8")
f.write(soup.get_text())
f.close()

#Tokenization
stokens = nltk.sent_tokenize(textout[1:1000])
wtokens = nltk.word_tokenize(textout[1:1000])

for s in stokens:
    print(s)

for t in wtokens:
    print(t)

print("---------------------------------Tokenization Done-------------------------------------------------")
#Stemming

pStemmer = PorterStemmer()

for p in wtokens:
  print(pStemmer.stem(p))

lStemmer = LancasterStemmer()

for l in wtokens:
  print(lStemmer.stem(l))

sStemmer = SnowballStemmer('english')
for s in wtokens:
  print(sStemmer.stem(s))

print("___________________-------Stemming Done________________________----------------------------------")


#Lematization

lemma = WordNetLemmatizer()

print(lemma.lemmatize(textout[1:1000]))

#POS

print("-------------lemma Done_____________________")

textnew = nltk.word_tokenize(textout[1:1000])
print(nltk.pos_tag(textnew))

print("----------------------------------------------------POS Done-------------------------------------------------------------------")

#NER

print(ne_chunk(pos_tag(wordpunct_tokenize(textout2))))

print("------------------------------------------------NER Done_--------------------------------------------------------------")

#Trigram

tgram = ngrams(wtokens,3)

for t in tgram:
    print(t)