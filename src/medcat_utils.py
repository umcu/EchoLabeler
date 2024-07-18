import re
import random
import pandas as pd
from typing import Callable

FLAG_TERMS = ['uitslag zie medische status', 'zie status', 'zie verslag status', 'slecht echovenster', 'echo overwegen', 'ge echo',
              'geen echovenster', 'geen beoordeelbaar echo', 'tee 190', 'hdf 36mnd', 'geen beoordeelbare echo', 'verslag op ic', 'psyheartanalyse']
MIN_TOKENS = 2
def create_dict_with_conf(ent):
    result_dict = {}
    for k, v in ent._.meta_anns.items():
        result_dict[k] = v['value']
        result_dict[f'conf_{k}'] = v['confidence']
    return result_dict

def _bad_doc(txt):
    if any(f in txt for f in FLAG_TERMS):
        return True
    if len(re.split(r'\W', txt)) < MIN_TOKENS:
        return True
    return False


def _remap_labs(label, classname):
    if classname in ['rv_syst_func', 'lv_syst_func']:
        if label == 'rv_syst_func_normal':
            return 'rv_sys_func_normal'
        elif label == 'lv_syst_func_normal':
            return 'lv_sys_func_normal'
        elif label == 'lv_sys_func_unchanged':
            return 'lv_sys_func_unknown'
        elif label == 'lv_sys_func_improved':
            return 'lv_sys_func_unknown'
        else:
            return label
    else:
        return label


def get_medcat_json_per_class(filename: str = None, HashSet: set = False, ClassName: str = None, ClassMap: dict = None):
    # to add negatives: we need to know, per span, if it is not labeled per class
    # if it is labeled for a class, and not for others, then it gets a 'nolabel' value
    if ClassName == 'merged':
        assert isinstance(ClassMap, dict), "ClassMap must be a dict to map labels to classes if ClassName is 'merged'"
    texts = pd.read_json(filename, lines=True)
    output = {"projects": [{
        "name": ClassName if ClassName != 'merged' else 'merged',
        "id": 42,
        "cuis": "",
        "tuis": "",
        "documents": None
    }]}

    documents = []
    for i, _row in enumerate(texts.iterrows()):
        row = _row[1]
        if (row["_input_hash"] in HashSet) | (HashSet is False):
            txt = row['text']
            if _bad_doc(txt):
                continue
            id = i
            input_hash = row["_input_hash"]
            task_hash = row["_task_hash"]
            annotations = []
            for j, ann in enumerate(row["spans"]):
                res = {}
                res['user'] = 'BVE'
                res['cui'] = 123
                res['id'] = j
                res['start'] = ann['start']
                res['end'] = ann['end']
                res['value'] = txt[ann['start']:ann['end']]
                res['validated'] = True
                res['correct'] = True
                res['deleted'] = False
                res['alternative'] = False
                res['killed'] = False
                res["meta_anns"] = {
                    ClassName: {
                        "name": ClassName if ClassName != 'merged' else ClassMap[ann['label']],
                        "validated": True,
                        "accuracy": 1.0,
                        "value": _remap_labs(ann['label'], ClassName)
                    }
                }
                annotations.append(res)
            doc = {
                'id': id,
                'text': txt,
                'input_hash': input_hash,
                'task_hash': task_hash,
                'annotations': annotations
            }
            documents.append(doc)
    output['projects'][0]['documents'] = documents

    return output
def _get_token_split(txt: str, split_by='\W'):
    splitter = re.compile(split_by)
    toks = []
    tbnds = []
    lb = 0

    for match in splitter.finditer(txt):
        start, end = match.span()
        if lb != start:
            toks.append(txt[lb:start])
            tbnds.append((lb, start - 1))
        lb = end

    if lb < len(txt):
        toks.append(txt[lb:])
        tbnds.append((lb, len(txt) - 1))

    return toks, tbnds

def _within_any_bounds(tup, bounds):
    for bound in bounds:
        if bound[0] <= tup[0] <= bound[1] or bound[0] <= tup[1] <= bound[1]:
            return True
    return False

def _get_token_groups(bnds):
    '''
    :param bnds: list of character ranges
    :return: sequence of numbers describing the # of the consequence
    '''

    # if difference between lower-bound and previous upper-bound is >2
    # then we go the next sequence

    seqnumbers = []
    seqdict = {}
    seqnum = 0
    i_b = 0  # index of the first token in the current sequence  # start of the sequence in the token list (bnds)
    old_bound = (0, 0)
    for i, bnd in enumerate(bnds):
        if (bnd[0] - old_bound[1]) > 2:
            seqdict[seqnum] = {'start_token': i_b, 'end_token': i - 1, 'size': i - 1 - i_b}
            seqnum += 1
            i_b = i
        old_bound = bnd
        seqnumbers.append(seqnum)

    return seqnumbers, seqdict


def _make_nolabel_class_dict(className: str = None,
                            start: int = None,
                            end: int = None,
                            val: str = None,
                            j: int = None):
    res = {}
    res['user'] = 'BVE'
    res['cui'] = 123
    res['id'] = j
    res['start'] = start
    res['end'] = end
    res['value'] = val
    res['validated'] = True
    res['correct'] = True
    res['deleted'] = False
    res['alternative'] = False
    res['killed'] = False
    res["meta_anns"] = {
        className: {
            "name": className,
            "validated": True,
            "accuracy": 1.0,
            "value": f"{className}_nolabel"
        }
    }
    return res

def update_medcat_json_per_class_with_negatives(MedCATJSON: dict = None,
                                                ClassName: str = None,
                                                MinTokens: int = 3,
                                                MaxTokens: int = 7,
                                                MedCATModel: Callable = None
                                                ):
    # We assume that the MedCATJSON holds the information for one class
    # Per document we collect the span that are labeled for this class
    # Now each span NOT labeled is a potential negative class, we
    # select for each labeled span an unlabeled span with a minimum of N tokens
    # if a document contains NO labeled spans we assume that every span is a 'nolabel' span

    assert ClassName == MedCATJSON['projects'][0][
        'name'], f"Class name mismatch? : {ClassName} / {MedCATJSON['projects'][0]['name']}"

    output = {"projects": [{
        "name": ClassName,
        "id": 42,
        "cuis": "",
        "tuis": "",
        "documents": None
    }]}

    Docs = MedCATJSON['projects'][0]['documents']
    NewDocs = []
    for i, doc in enumerate(Docs):
        annotations = doc["annotations"]
        text = doc['text']
        toks, tbnds = _get_token_split(text)

        # get estimated medical entities
        ment_bnds = []
        ment_ntoks = []
        if MedCATModel is not None:
            for _ent in MedCATModel(text):
                ment_bnds.append(_ent.start_char, _ent.end_char)
                ment_ntoks.append(len(_get_token_split(_ent.text)[0]))
        # collect labeled spans
        LabeledSpans = []
        for ann in doc["annotations"]:
            LabeledSpans.append((ann['start'], ann['end']))
        NumLabSpans = len(LabeledSpans)
        if NumLabSpans > 0:
            LabeledSpans = list(set(LabeledSpans))
            PotentialNegativeSpans = [tbnd for tbnd in tbnds
                                      if not _within_any_bounds(tbnd, LabeledSpans)]
            SpanSeqs, SpanSeqsDict = _get_token_groups(PotentialNegativeSpans)

            _k = 0
            # TODO: include medical entities with less than min tokens, then add more tokens
            if len(ment_bnds) > 0:
                for (char_bnd, ntoks) in enumerate(random.sample(zip(ment_bnds, ment_ntoks), len(ment_bnds))):
                    if ntoks > MinTokens:
                        _start, _end = char_bnd[0], char_bnd[1]
                        # check if there are no LabeledSpans in _start,_end
                        if not _within_any_bounds((_start, _end), LabeledSpans):
                            _value = text[_start:_end + 1]
                            res = _make_nolabel_class_dict(ClassName, _start, _end, _value)
                            annotations.append(res)
                            _k += 1
                            if _k >= NumLabSpans:
                                break

            if _k < NumLabSpans:
                for spanNum in random.sample(list(SpanSeqsDict.keys()), len(SpanSeqsDict)):
                    spanChars = SpanSeqsDict[spanNum]
                    if spanChars['size'] > MinTokens:
                        _MaxTokens = min(spanChars['size'], MaxTokens)
                        NumTokens = random.randint(MinTokens, _MaxTokens)
                        start_span = PotentialNegativeSpans[spanChars['start_token']]
                        end_span = PotentialNegativeSpans[spanChars['start_token'] + NumTokens]
                        _start, _end = start_span[0], end_span[1]
                        _value = text[_start:_end + 1]
                        res = _make_nolabel_class_dict(ClassName, _start, _end, _value)
                        annotations.append(res)

                        _k += 1
                        if _k >= NumLabSpans:
                            break
        else:
            _k = 0
            if len(ment_bnds) > 0:
                # we can now use the medical entities directly if they have within Min/Max number
                # of tokens
                # TODO: include medical entities with less than min tokens, then add more tokens
                for (char_bnd, ntoks) in enumerate(random.sample(zip(ment_bnds, ment_ntoks), len(ment_bnds))):
                    if ntoks > MinTokens:
                        _start, _end = char_bnd[0], char_bnd[1]
                        _value = text[_start:_end + 1]
                        res = _make_nolabel_class_dict(ClassName, _start, _end, _value)
                        annotations.append(res)

                        _k += 1
                        if _k >= NumLabSpans:
                            break
            try:
                if (_k < NumLabSpans) | (NumLabSpans == 0):
                    _MaxTokens = min(len(tbnds), MaxTokens)
                    TokNum = random.randint(MinTokens, _MaxTokens)
                    _start, _end = (tbnds[0][0], tbnds[TokNum - 1][1])
                    _value = text[_start:_end + 1]
                    res = _make_nolabel_class_dict(ClassName, _start, _end, _value)
                    annotations.append(res)
            except Exception as e:
                print(f"Error: for |{text}|")
                raise (IndexError, "index error")

        doc['annotations'] = annotations
        NewDocs.append(doc)
    output['projects'][0]['documents'] = NewDocs
    return output