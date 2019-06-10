import contentToForm
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords as sw
from nltk.stem import WordNetLemmatizer
from definitions import DEFINITIONS
from nltk.stem.porter import PorterStemmer

stopwords = sw.words('english')
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

for key, definitions in DEFINITIONS.items():

    content_words = dict()
    stemm_words = set()

    for definition in definitions:

        def_words = nltk.pos_tag(nltk.word_tokenize(definition), lang='eng')
        lemmas = []
        stemm = []
        for token in def_words:

            if token[0] not in stopwords:
                if token[1] in ['FW', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'] and token[0] not in "?!,'()’“”":
                    pos = contentToForm.__nltk_to_wordnet_tag(token[1])

                    stemm.append(stemmer.stem(token[0]))
                    lemmas.append([lemmatizer.lemmatize(token[0], pos = pos),pos])

        for word in lemmas:
            if word[0] in content_words.keys():
                content_words[word[0]] = [word[1], content_words[word[0]][1] + 1]
            else:
                content_words[word[0]] = [word[1], 1]

        for word in stemm:
            stemm_words.add(word)

    relevant = []
    sense = wn.synsets(str(key).lower())
    print(sense)

    for word in content_words.keys():
        if content_words[word][1] > 1:
            relevant.append(wn.synsets(word)[0])

    common = contentToForm.findCommonChild(relevant, stemm_words)

    best_sense = contentToForm.findSense(common, definitions)
    print(best_sense)

    MAX_VALUE = 0

    for i in sense:
        if best_sense.wup_similarity(i) != None:
            value = best_sense.wup_similarity(i)
            if value> MAX_VALUE:
                MAX_VALUE = value
    
    print(MAX_VALUE)

    print()

