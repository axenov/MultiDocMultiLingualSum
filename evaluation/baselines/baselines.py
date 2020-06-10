from baselines.random import Random
from baselines.lead import Lead
from baselines.lexrank import LexRank
from baselines.textrank import TextRank
from baselines.tfidf import TFIDF
from baselines.rouge_oracle import RougeOracle
from baselines.bert2bert import Bert2Bert
from baselines.bart import Bart
from baselines.t5 import T5
from baselines.t5_with_title import T5WithTitle
from baselines.combine import Combine


def use(baseline_class, **init_kwargs):
    if baseline_class == "Random":
        return Random(**init_kwargs)
    if baseline_class == "Lead":
        return Lead(**init_kwargs)
    if baseline_class == "LexRank":
        return LexRank(**init_kwargs)
    if baseline_class == "TextRank":
        return TextRank(**init_kwargs)
    if baseline_class == "TFIDF":
        return TFIDF(**init_kwargs)
    if baseline_class == "RougeOracle":
        return RougeOracle(**init_kwargs)
    if baseline_class == "Bert2Bert":
        return Bert2Bert(**init_kwargs)
    if baseline_class == "Bart":
        return Bart(**init_kwargs)
    if baseline_class == "T5":
        return T5(**init_kwargs)
    if baseline_class == "T5 with title":
        return T5WithTitle(**init_kwargs)
    if baseline_class == "Combine":
        return Combine(**init_kwargs)
    raise ValueError("Baseline baseline_class not correct")
