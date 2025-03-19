import contractions
import inflect
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, words
from nltk import PorterStemmer, LancasterStemmer, download

class TextPreprocessor(object):
    """
    Ajoute contractions au processing de sklearn.
    Doit par contre re-implementer lowercase et strip accent
    """
    def __init__(self, 
                 lower_case=True,
                 remove_punctuation=True,
                 remove_contractions=True):
        self._lower_case = lower_case
        self._remove_punctuation = remove_punctuation
        self._remove_contractions = remove_contractions
    
    def __call__(self, text):
        if self._lower_case:
            text = text.lower()
            
        if self._remove_punctuation:
            text = "".join([c for c in text if c not in string.punctuation])

        if self._remove_contractions:
            text = contractions.fix(text)

        return text

class TextTokenizer(object):
    """
    Ajoute quelques fonctionalite a sklearn (stemming, numbers to word).
    Doit re-implementer les stopwords
    """
    def __init__(self,
                 stop_words=None,
                 restrict_to_words=None,
                 number_to_words=True,
                 stemmer=None):
        self._stop_words = None
        if not stop_words is None:
            self._stop_words = set(stop_words)

        self._restrict_to_words = None
        if not restrict_to_words is None:
            self._restrict_to_words = set(restrict_to_words)
            
        self._stemmer = stemmer
        self._inflect_engine = inflect.engine() if number_to_words else None

    def __call__(self, text):
        words = word_tokenize(text)
        
        if self._stop_words:
            words = [w for w in words if w not in self._stop_words]

        if self._restrict_to_words:
            words = [w for w in words if w in self._restrict_to_words]

        if self._inflect_engine:
            words = self._number_to_words(words)
            
        if self._stemmer:
            words = [self._stemmer.stem(w) for w in words]

        return words

    def _number_to_words(self, words):
        def _internal(word):
            if word.isnumeric():
                return self._inflect_engine.number_to_words(word)
            return word
        return [_internal(w) for w in words]


def get_stopwords(language="english"):
    return stopwords.words(language)

def get_english_corpus_words():
    """
    Liste de mots en anglais
    """
    return words.words()

def stemmer_porter():
    return PorterStemmer()

def stemmer_lancaster():
    return LancasterStemmer()

def import_nltk_dependencies():
    download("punkt")
    download("stopwords")
    download('words')

def fit_transform(data_df,
                  transformer_type, 
                  preprocessor, 
                  tokenizer, 
                  **transformer_type_kwargs):
    model = transformer_type(preprocessor=preprocessor,tokenizer=tokenizer, **transformer_type_kwargs)
    data_fit_transform = model.fit_transform(data_df)
    return model, data_fit_transform
