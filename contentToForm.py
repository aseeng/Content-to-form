import re
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords as sw
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import string

def __nltk_to_wordnet_tag(tag):

    if tag in ['NN', 'NNS', 'NNP', 'NNPS', 'FW']:
        return wn.NOUN
    elif tag in ['JJ', 'JJR', 'JJS']:
        return wn.ADJ
    elif tag in ['RB', 'RBR', 'RBS']:
        return wn.ADV
    elif tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'MD']:
        return wn.VERB


def getHyponyms(synset):
    hyponyms = set()
    for s in synset.hyponyms():
        for lemma in s.lemma_names():
            hyponyms = hyponyms.union(tokenize(lemma))
        hyponyms = hyponyms.union(tokenize(s.definition()))
    return hyponyms


def getExamples(synset):
    examples = set()
    for example in synset.examples():
        examples = examples.union(tokenize(example))

    return examples

def findCommonChild(relevant, stemm):

    lemmatizer = WordNetLemmatizer()
    search = []
    sol = set()

    for i in stemm:
        find = set()
        count = 0
        rel = ""
        for j in relevant:
            if re.findall("'\D*("+i+")",str(j)):
                count +=1
                find.add(j)

        if count >= 5:
            sol = sol.union(find)

    if len(sol) >= 15:
        return sol

    for word in relevant:
        word_hypon = getHyponyms(word)
        for hypon in word_hypon:
            lem = lemmatizer.lemmatize(hypon)
            for h in wn.synsets(lem, pos='n'):
                search.append(h)

    for i in search:
        relevant.append(i)

    return findCommonChild(relevant, stemm)

def tokenize(sentence):
    lemmatizer = WordNetLemmatizer()
    tokens = set(word for word in word_tokenize(sentence.lower()) if word not in sw.words('english') and word not in string.punctuation)

    for token in tokens:
        tokens = tokens.union(lemmatizer.lemmatize(token))

    tokens = tokens.difference(token for token in tokens if len(token)<2)
    return tokens


def findSense(common, definitions):

    best_sense = ""
    max_overlap = 0

    for i in common:
        signature = set()
        for lemma in i.lemma_names():
            signature = signature.union(tokenize(lemma))

        signature = signature.union(tokenize(i.definition()))
        signature = signature.union(getExamples(i))
        signature = signature.union(getHyponyms(i))

        sentence = set()

        for j in definitions:
            data = j.split()
            sentence.update(data)

        overlap = set(signature).intersection(sentence)

        if len(overlap) > max_overlap:
            max_overlap = len(overlap)
            best_sense = i

    return best_sense