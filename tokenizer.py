import numpy as np
from numpy import dot, ndarray
from numpy.linalg import norm
from typing import List
from text2digits import text2digits

from stop_words import stop_words
from universal import nlp_global_object


class TokenizerHelper:
    @staticmethod
    def get_average_vector(vector_list: List[ndarray]):
        assert len(vector_list) > 0
        vec = np.zeros(vector_list[0].size)
        count: int = 0
        for item in vector_list:
            vec += item
            count += 1

        avg_vec = vec / count
        return avg_vec

    @staticmethod
    def get_cosine_similarity(vec1: ndarray, vec2: ndarray):
        cos_sim = dot(vec1, vec2) / (norm(vec1) * norm(vec2))
        return cos_sim


class TextPreProcessor:
    def __init__(self):
        self.contractions = {
            "'m": " am", " he's ": " he is ", " she's ": " she is ", " it's ": " it is ", "'re": " are", "'d": " would",
            "'ll": " will", " isn't ": " is not ", " aren't ": " are not ", " wasn't ": " was not ",
            " weren't ": " were not ", " hasn't ": " has not ", " haven't ": " have not ", " hadn't ": " had not ",
            " don't ": " do not ", " doesn't ": " does not ", " didn't ": " did not ", " can't ": " can not ",
            " couldn't ": " could not ", " won't ": " will not ", " wouldn't ": " would not ",
            " shouldn't ": " should not ", " mustn't ": " must not ", " needn't": " need not ",
            " mightn't ": " might not ", " daren't ": " dare not ", " let's ": " let us ", " who's": " who is ",
            " who'd ": " who would ", " who'll ": " who will ", " what's ": " what is ", " what'll ": " what will ",
            " how's ": " how is ", " where's ": " where is ", " when's ": " when is ", " here's ": " here is ",
            " there's ": " there is ", " there'd ": " there would ", " there'll ": " there will ",
            " that's ": " that is ", " hes ": " he is ", " shes ": " she is ", " its ": " it is ", " isnt ": " is not ",
            " arent ": " are not ", " wasnt ": " was not ", " werent ": " were not ", " hasnt ": " has not ",
            " havent ": " have not ", " hadnt ": " had not ", " dont ": " do not ", " doesnt ": " does not ",
            " didnt ": " did not ", " cant ": " can not ", " couldnt ": " could not ", " wont ": " will not ",
            " wouldnt ": " would not ", " shouldnt ": " should not ", " mustnt ": " must not ",
            " neednt ": " need not ",
            " mightnt ": " might not ", " darent ": " dare not ", " lets ": " let us ", " whos": " who is ",
            " whats ": " what is ", " hows ": " how is ", " thats ": " that is "
        }
        self.t2d = text2digits.Text2Digits()

    def _fix_contractions(self, input_text: str):
        for key in self.contractions:
            input_text = input_text.replace(key, self.contractions[key])

        return input_text

    def work(self, input_text):
        input_text = " " + input_text.lower().replace("’", "'") + " "
        input_text = self._fix_contractions(input_text)
        input_text = input_text.translate({ord(c): " " for c in "!@#$%^&*()[]{};:./<>?\|`~-=_+'\""})
        input_text = self.t2d.convert(input_text)
        input_text = ", ".join(input_text.split(","))

        return input_text


class Tokenizer:
    def __init__(self):
        self.nlp = nlp_global_object

    def _get_tokens(self, tags_text: str, lemmatize: bool, get_word_tokens_only: bool):
        word_tokens = []
        full_tokens = []

        pre_processor = TextPreProcessor()
        tags_text = pre_processor.work(tags_text)
        doc = self.nlp(tags_text)

        if lemmatize:
            lemmas = [token.lemma_ for token in doc]
            lemmas = [sub.replace('abs', 'ab') for sub in lemmas]
            lemmatized_text = " ".join(lemmas)
            doc = self.nlp(lemmatized_text)

        for token in doc:
            if token.is_oov:
                continue

            if len(token.text) <= 1 and not token.like_num and token.text != ',':
                continue

            if token.text in stop_words:
                continue

            word_tokens.append(token.text)
            full_tokens.append(token)

        if get_word_tokens_only:
            return word_tokens
        else:
            return full_tokens

    def get_word_tokens(self, tags_text: str):
        return self._get_tokens(tags_text, lemmatize=False, get_word_tokens_only=True)

    def get_lemmatized_word_tokens(self, tags_text: str):
        return self._get_tokens(tags_text, lemmatize=True, get_word_tokens_only=True)

    def get_tokens(self, tags_text: str, lemmatize: bool):
        return self._get_tokens(tags_text, lemmatize=lemmatize, get_word_tokens_only=False)

    def get_document(self, tags_text: str):
        word_tokens = self.get_lemmatized_word_tokens(tags_text)
        if len(word_tokens) == 0:
            return None
        return self.nlp(" ".join(word_tokens))
