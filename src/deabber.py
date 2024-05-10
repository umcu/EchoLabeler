import benedict
from tqdm import tqdm
import re
from gensim.models import Word2Vec, Doc2Vec, FastText
from sentence_transformers import SentenceTransformer, util as st_util
from typing import Dict, List, Tuple, Union, Callable
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
import torch
from scipy.spatial.distance import cosine
import numpy as np

from gensim.models.callbacks import CallbackAny2Vec

class deabber():
    '''
        Deabbreviation of clinical abbreviations
    '''
    def __init__(self,
                 model_type: str='sbert',
                 model: str='FremyCompany/BioLORD-2023-M',
                 abbreviations: Union[Dict[str, List[str]], str] = None,
                 min_sim: float=0.5,
                 top_k: int=7):
        self.model_type = model_type
        self.model_name = model
        self.tokenizer = None
        self.min_similarity = min_sim
        self.top_k = top_k
        if isinstance(abbreviations,str):
            self.abbreviations = benedict.benedict.from_yaml(abbreviations)
        else:
            self.abbreviations = abbreviations

        print("Initializing model...")
        self._initialise()


    def Likelihood(self, probeString: str):
        """Get log-likelihood of a string using a pre-trained model such as BERT or Llama"""
        return 0

    def _initialise(self):
        if self.model_type == 'sbert':
            self.model = SentenceTransformer(self.model_name)
        elif self.model_type == 'fasttext':
            self.model = FastText.load(self.model_name)
        elif self.model_type == 'transformer':
            self.model = FastText.load(self.model_name)
            self.mlm = pipeline("fill-mask",
                                  model=self.model_name,
                                  tokenizer=self.tokenizer)
    @staticmethod
    def get_context_window(TokenList: List[str],
                    CharnumListSpan: List[Tuple],
                    TokenRadius: int,
                    SPAN: Tuple[int, int]):
        start_list, end_list = [t[0] for t in CharnumListSpan], [t[1] for t in CharnumListSpan]
        start_abbr = [i for i, x in enumerate(end_list) if x >= SPAN[0]][0]
        end_abbr = [i for i, x in enumerate(start_list) if x >= SPAN[1]][0]

        MATCH_TOKEN_SPAN = (start_abbr, end_abbr)
        CONTEXT_TOKEN_SPAN = (max(0, start_abbr-TokenRadius), min(len(TokenList), end_abbr+TokenRadius))
        ContextTokenList = TokenList[CONTEXT_TOKEN_SPAN[0]:CONTEXT_TOKEN_SPAN[1]+1]
        return MATCH_TOKEN_SPAN, ContextTokenList


    def SenSim(self, probeString: str,
               OriginalString: str,
               Model: str=None):
        """Get semantic similarity between two strings using a pre-trained model"""

        if (self.model_type == 'sbert') | (Model == 'sbert'):
            assert isinstance(self.model, SentenceTransformer), "SBERT_MODEL must be a SentenceTransformer model"
            v1 = self.model.encode(probeString)
            v2 = self.model.encode(OriginalString)
            return st_util.pytorch_cos_sim(v1, v2)
        elif (self.model_type == "fastText") | (Model == 'fastText'):
            assert isinstance(self.model, FastText), "FASTTEXT_MODEL must be a FastText model"
            # TODO: should be fastText per N tokens
            v1 = self.model.wv[probeString]
            v2 = self.model.wv[OriginalString]
            return st_util.pytorch_cos_sim(v1, v2)
        else:
            raise NotImplementedError

    def get_most_likely_replacement(self, ReplaceLoc: int,
                                    ContextTokenList: List[str],
                                    Replacements: List[str]):
        """Get most likely replacement for an abbreviation"""

        OriginalToken = ContextTokenList[ReplaceLoc]
        OriginalString = " ".join(ContextTokenList)
        likelihood = []
        similarities = []
        repllist = []

        if self.model_type == 'transformer':
            probeString = " ".join(ContextTokenList[:ReplaceLoc]) + " <mask> " + " ".join(
                ContextTokenList[ReplaceLoc:])
            fill_in = self.mlm(probeString, top_k=self.top_k)
            masks = [(o['token_str'].strip(), o['score']) for o in fill_in]
            for replacement in Replacements:
                _similarities = []
                for alt, score in masks:
                    _similarities.append(self.SenSim(replacement, alt, Model="fastText"))
                agg_sim = np.mean(_similarities)
                similarities.append(agg_sim)
        else:
            for replacement in Replacements:
                probeString = " ".join(ContextTokenList[:ReplaceLoc]) + " " + replacement + " " + " ".join(
                    ContextTokenList[ReplaceLoc:])
                likelihood.append(self.Likelihood(probeString))
                similarities.append(self.SenSim(probeString, OriginalString))
                repllist.append(replacement)

        if len(likelihood) > 0:
            if max(similarities) > self.min_similarity:
                repl = repllist[similarities.index(max(similarities))]
            else:
                repl = OriginalToken
        else:
            repl = OriginalToken

        return repl

    def deabb(self,
                docs: List[str],
                abbreviations: Dict[str,List[str]]=None,
                TokenRadius: int = 3)->List[str]:
        """Deabbreviate text

        Args:
            TEXT_echo (List[str]): Text to deabbreviate
            abbreviations (Dict): Dictionary of abbreviations

        Returns:
            List[str]: Deabbreviated text
        """

        if abbreviations is None:
            abbreviations = self.abbreviations
            assert (len(abbreviations) > 0), "No abbreviations are given, please give as a parameter or set as attribute"

        TEXT_deabbreviated = []
        for line in tqdm(docs):
            # TODO: remove leading and trailing \b
            # line = re.sub(pattern=r'^\b(.*)\b$', repl='\1', string=line)

            # add character number range to each token, make a list of tuples
            # CharnumList = [len(token) for token in TokenList]
            # CharnumListSpan = [(sum(CharnumList[:i]), sum(CharnumList[:i+1])) for i in range(len(CharnumList))]
            for key in abbreviations:
                new_key = "".join([f"{c.upper()}\\.?" for c in key])
                re_abbr_ = re.compile(rf"{new_key}")
                re_abbr = re.compile(rf"\b{new_key}\b")
                if len(abbreviations[key]) == 1:
                    if re_abbr.search(line) is not None:
                        line = re_abbr.sub(repl=abbreviations[key][0], string=line)
                        # ideally you would do this iteratively, and check for some minimum semantic similarity..
                else:
                    # we go through each match object and check the match
                    # between context with the abbrevation and the context with the de-abbrevation
                    # TODO: add option to use encoder-based LLMs to fill in the log-likehood for the de-abbrevation
                    if re_abbr.search(line) is not None:
                        TokenList = re.split(pattern=r'\b', string=line)
                        Replacements = abbreviations[key]
                        NewTokens = []
                        for i, token in enumerate(TokenList):
                            if re_abbr_.match(token) is not None:
                                min_r = max(0, i - TokenRadius)
                                max_r = min(len(TokenList), i + TokenRadius) + 1

                                NewTokens.append(
                                    self.get_most_likely_replacement(
                                        ReplaceLoc=min_r,
                                        ContextTokenList=TokenList[i - min_r:i + max_r],
                                        Replacements=Replacements
                                    )
                                )
                            else:
                                NewTokens.append(token)
                        line = "".join(NewTokens)
                    else:
                        continue
            TEXT_deabbreviated.append(line)
        return TEXT_deabbreviated
