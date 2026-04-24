"""
Evaluators: glue between experiment scripts and the consensus simulation.

- ``evaluation_runner.Runner`` — loads hold-out context, builds EASer/LightFM pieces, runs many groups
  through a mediator *factory* with optional population-mood biases and NDCG aggregation.
- ``consensus_mediator_factories`` — fluent builders that, given a user group, return a configured
  ``ConsensusMediator*`` plus metadata (async / sync / hybrid paths).
- ``consensus_evaluator.ConsensusAgentBasedEvaluator`` — drives round-by-round interaction: asks the
  mediator for lists, uses ``UserVoteSimulator`` to imitate users, tracks first-match rounds and NDCG.
- ``evaluation_data_interpreter`` — small CLI-friendly pretty-printer for one run's result dict.
- ``consensus_evaluation_agents/`` — simulated users (preference model + voting rules).

Experiment modules (``eval_*.py`` / ``tune_*.py``) typically construct a ``Runner``, attach a factory
from ``consensus_mediator_factories``, then call ``ConsensusAgentBasedEvaluator`` inside ``Runner.run``.
"""
