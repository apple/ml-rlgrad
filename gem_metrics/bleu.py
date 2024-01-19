#!/usr/bin/env python3

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#


from .metric import ReferencedMetric
from .texts import Predictions, References

from typing import Dict
import sacrebleu
from itertools import zip_longest


class BLEU(ReferencedMetric):
    """BLEU uncased BLEU from SacreBLEU."""

    def support_caching(self):
        # BLEU is corpus-level, so individual examples can't be aggregated.
        return False

    def compute(self, cache, predictions: Predictions, references: References) -> Dict:
        ref_streams = list(zip_longest(*references.untokenized))
        bleu = sacrebleu.corpus_bleu(
            predictions.untokenized, ref_streams, lowercase=True
        )
        return {"bleu": round(bleu.score, 5)}
