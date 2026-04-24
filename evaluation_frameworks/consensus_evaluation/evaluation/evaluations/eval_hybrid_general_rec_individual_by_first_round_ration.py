"""
**Compatibility entrypoint — same experiment class as ``eval_hybrid_general_rec_individual``.**

English: historically this module swept ``first_round_ration``; that belongs in
``tune_hybrid_general_rec_individual``. The primary **eval** with a fixed ratio is
``eval_hybrid_general_rec_individual`` (see ``DEFAULT_SIGMOID_PARAMS`` / ``first_r_ration`` per ``W``).

Czech: zbytek popisu viz výše — tento soubor jen přesměruje ``autorun`` na stejnou třídu.
"""

from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.config import autorun
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_hybrid_general_rec_individual import (
    EvalHybridGeneralRecIndividual,
)

if __name__ == "__main__":
    autorun(EvalHybridGeneralRecIndividual)
