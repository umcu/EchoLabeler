# add autoreload
import numpy as np
from typing import Callable, List, Union, Literal
import spacy
from gensim.models import Word2Vec as W2V, keyedvectors as KV
from gensim.models import Doc2Vec, fasttext
import os
import re
from tqdm import tqdm

def report_filter(texts: List[str],
                  Tokenizer: Callable = None,
                  min_tokens: int = 4,
                  min_chars: int = 15,
                  flag_terms: List[str] = None,
                  save_terms: List[str] = None,
                  suspect_terms: List[str] = None):
    '''
    :param texts: texts to filter, List of strings
    :param Tokenizer: Tokenizer to be used for filtering, use standard tokenizer if None
           The tokenizer is expected to ingest a string, and output a list of strings
    :param min_tokens: texts with < min_tokens are filtered out
    :param min_chars: texts with < min_char are filtered out
    :param flag_terms: texts with any of these spans are filtered out
    :param save_terms: texts with these terms should remain
    :param suspect_terms: texts with any of these spans are flagged as suspect for removal
    :return:
    '''
    if Tokenizer is None:
        Tokenizer = lambda x: x.split()

    filtered_texts = []
    suspect_texts = []
    for text in texts:
        Tokens = Tokenizer(text)

        numtokens = len(Tokens)
        numchars = len(text)

        if (numtokens <= min_tokens) or (numchars <= min_chars):
            if any([st in text.lower() for st in save_terms]):
                filtered_texts.append(text)
            else:
                filtered_texts.append(None)
            continue

        if any([ft in text.lower() for ft in flag_terms]):
            filtered_texts.append(None)
            continue

        if (isinstance(suspect_terms, list)) and (any([st in text for st in suspect_terms])):
            filtered_texts.append(text)
            suspect_texts.append(text)
            continue

        filtered_texts.append(text)
    return filtered_texts, suspect_texts

class TextToVectors():
    def __init__(self, 
                 source: Literal['spacy', 'cardio_wv', 'cardio_sb']='cardio_wv',
                 padmethod: Literal['zero', 'random', 'mean']='mean',                  
                 maxlen: int=200, 
                 window: int=3,
                 embedding_path: str=None,
                 use_progress_bar: bool=True
                 ):

        assert source in ['spacy', 'cardio_wv', 'cardio_sb'], f'source {source} not supported'
        assert padmethod in ['zero', 'random', 'mean'], f'padmethod {padmethod} not supported'
        assert maxlen > 0, f'maxlen {maxlen} not supported'
        assert window > 0, f'window {window} not supported'
        assert embedding_path is None or os.path.exists(embedding_path), f'embedding_path {embedding_path} not supported'

        self.source = source
        self.padmethod = padmethod
        self.maxlen = maxlen
        self.window = window
        self.use_progress_bar = use_progress_bar
        
        if source =='cardio_wv':          
            #EMB_PATH = EMB_PATH.replace("/",
            #                            "\\\\")
            self.wvs = fasttext.FastTextKeyedVectors.load(embedding_path)
            self.emb_dim = self.wvs.vector_size # self.wvs.wv[0].shape[0]
            self.vocab_size = self.wvs.vectors_vocab.shape[0]
        elif source == 'spacy':
            self.nlp = spacy.load("nl_core_news_lg", disable=['parser', 'ner'])
            self.emb_dim = self.nlp.vocab.vectors_length
            self.vocab_size = self.nlp.vocab.vectors.n_keys

    def text_to_vectors(self, texts: List[str]=None):
        '''
        # TODO: add option for other embeddings
        # TODO: for padding don't use zero vector but random or mean vector
        :param texts: texts to embed, list of strings
        :param maxlen: max length, in tokens, of the texts
        :return: array of arrays
        '''
        max_trailing_mean_window = 5
        
        if self.source == 'spacy':            
            _docs = self.nlp.pipe(texts)
            # We truncate or pad the document vector to a fixed size
            array_list = []
            for doc in tqdm(_docs, disable=~self.use_progress_bar):
                vectors = [token.vector for token in doc]
                mwv = np.mean(vectors, axis=0)
                
                if len(vectors) > self.maxlen:
                    vectors = vectors[:self.maxlen]
                elif len(vectors) < self.maxlen:
                    if self.padmethod == 'zero':
                        vectors += [np.zeros((self.nlp.vocab.vectors_length,))] * (self.maxlen - len(vectors))
                    elif self.padmethod == 'random':
                        vectors += [np.random.rand(self.nlp.vocab.vectors_length)] * (self.maxlen - len(vectors))
                    elif self.padmethod == 'mean':
                        vectors += [mwv] * (self.maxlen - len(vectors))
                    
                array_list.append(np.array(vectors))
        elif self.source == 'cardio_wv':            
            # go through documents
            array_list = []
            splitter = re.compile(r'[^\w]')
            for txt in tqdm(texts, disable=~self.use_progress_bar):
                words = splitter.split(txt)    
                vectors = []
                for k in range(self.maxlen):
                    try:
                        vectors.append(self.wvs[words[k]])
                    except:
                        trailing_mean_window = min(max_trailing_mean_window,
                                                   len(vectors)-1)
                        mwv = np.mean(vectors[-trailing_mean_window:], axis=0)
                        vectors.append(mwv)
                    
                mwv = np.mean(vectors, axis=0)            
                
                if len(vectors) < self.maxlen:
                    if self.padmethod == 'zero':
                        vectors += [np.zeros((self.emb_dim,))] * (self.maxlen - len(vectors))
                    elif self.padmethod == 'random':
                        vectors += [np.random.rand(self.emb_dim)] * (self.maxlen - len(vectors))
                    elif self.padmethod == 'mean':
                        vectors += [mwv] * (self.maxlen - len(vectors)) 
                
                array_list.append(np.array(vectors))    
                # if word not in vocab, take closest one according to Levenhstein or mean of surrounding tokens
        elif self.source == 'cardio_sb':
            # TODO: add sentence BERT, with sliding window.
            pass
        return np.array(array_list)
    