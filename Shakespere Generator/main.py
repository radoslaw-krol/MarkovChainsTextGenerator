import re
import warnings

import markovify
import spacy
from nltk.corpus import gutenberg

warnings.filterwarnings('ignore')

# nltk.download('gutenberg')


# print(gutenberg.fileids())

hamlet = gutenberg.raw('shakespeare-hamlet.txt')
macbeth = gutenberg.raw('shakespeare-macbeth.txt')
caesar = gutenberg.raw('shakespeare-caesar.txt')


# print('\nRaw:\n', hamlet[:100])
# print('\nRaw:\n', macbeth[:100])
# print('\nRaw:\n', caesar[:100])

def text_cleaner(text):
    text = re.sub(r'--', ' ', text)
    text = re.sub('[\[].*?[\]]', '', text)
    text = re.sub(r'(\b|\s+\-?|^\-?)(\d+|\d*\.\d+)\b', '', text)
    text = ' '.join(text.split())
    return text


hamlet = re.sub(r'Chapter \d+', '', hamlet)
macbeth = re.sub(r'Chapter \d+', '', macbeth)
caesar = re.sub(r'Chapter \d+', '', caesar)

hamlet = text_cleaner(hamlet)
caesar = text_cleaner(caesar)
macbeth = text_cleaner(macbeth)

nlp = spacy.load("en_core_web_sm")
hamlet_doc = nlp(hamlet)
macbeth_doc = nlp(macbeth)
caesar_doc = nlp(caesar)

hamlet_sents = ' '.join([sent.text for sent in hamlet_doc.sents if len(sent.text) > 1])
macbeth_sents = ' '.join([sent.text for sent in macbeth_doc.sents if len(sent.text) > 1])
caesar_sents = ' '.join([sent.text for sent in caesar_doc.sents if len(sent.text) > 1])
shakespeare_sents = hamlet_sents + macbeth_sents + caesar_sents

# print(shakespeare_sents)

generator_1 = markovify.Text(shakespeare_sents, state_size=3)


# next we will use spacy's part of speech to generate more legible text
class POSifiedText(markovify.Text):
    def word_split(self, sentence):
        return ['::'.join((word.orth_, word.pos_)) for word in nlp(sentence)]

    def word_join(self, words):
        sentence = ' '.join(word.split('::')[0] for word in words)
        return sentence


# Call the class on our text
generator_2 = POSifiedText(shakespeare_sents, state_size=3)

# now we will use the above generator to generate sentences
for i in range(5):
    print(generator_2.make_sentence())
# print 100 characters or less sentences
for i in range(5):
    print(generator_2.make_short_sentence(max_chars=100))
