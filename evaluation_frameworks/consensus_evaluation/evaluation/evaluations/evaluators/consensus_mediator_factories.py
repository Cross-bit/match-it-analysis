"""
**Mediator factories** for experiments: map ``(user_ids,) -> (mediator, meta_dict)``.

Builders (async / sync / hybrid) mirror the construction steps in production mediators — recommender
engine, priority function, redistribution unit, threshold policy, mediator class — so ``Runner.run``
can swap entire algorithm stacks without branching on experiment name.

Return ``meta`` is stored in evaluation metadata for tables (method name, component classes, etc.).
"""

from typing import Any, Dict


class AsyncMediatorFactoryBuilder:
    """Fluent builder for async mediators (per-user engine + redistribution + threshold + mediator)."""

    def __init__(self):
        self._mk_recommender = None
        self._mk_priority = None
        self._mk_threshold_static = None
        self._mk_threshold_sigmoid = None
        self._mk_redistrib = None
        self._mk_mediator = None

    def with_recommender_engine(self, f):
        self._mk_recommender = f
        return self

    def with_priority_function(self, f):
        self._mk_priority = f
        return self

    # alias – obě nastavují továrnu na threshold policy
    def with_threshold_policy(self, f):
        self._mk_threshold_static = f
        return self

    def with_sigmoid_policy(self, f):
        self._mk_threshold_sigmoid = f
        return self

    def with_redistribution(self, f):
        self._mk_redistrib = f
        return self

    def with_mediator(self, f):
        self._mk_mediator = f
        return self

    def build(self):
        def factory(group):
            rec = self._mk_recommender(group)
            pr  = self._mk_priority(group)
            ru  = self._mk_redistrib(group, pr)

            if self._mk_threshold_sigmoid:
                th = self._mk_threshold_sigmoid(ru)
            elif self._mk_threshold_static:
                th = self._mk_threshold_static()

            med = self._mk_mediator(group, rec, ru, th)

            meta = {
                "eval_method": "async",
                "recommender": rec.__class__.__name__,
                "priority_function": getattr(pr, "get_metadata", lambda: pr.__class__.__name__)(),
                "threshold_policy": getattr(th, "get_metadata", lambda: th.__class__.__name__)(),
            }
            return med, meta

        return factory


class SyncMediatorFactoryBuilderSync:
    """Fluent builder for sync mediators (group EASer aggregate + ``ConsensusMediatorSyncApproach``)."""

    def __init__(self):
        self._mk_group_algorithm = None
        self._mk_group_recommender = None
        self._mk_mediator = None

    def with_group_algorithm(self, f):
        self._mk_group_algorithm = f
        return self

    def with_group_recommender_engine(self, f):
        # f: (group, easer_group) -> group_recommender
        self._mk_group_recommender = f
        return self

    def with_mediator(self, f):
        # f: (group, group_recommender) -> mediator
        self._mk_mediator = f
        return self

    def build(self):
        def factory(group):
            eg  = self._mk_group_algorithm()
            gre = self._mk_group_recommender(group, eg)
            med = self._mk_mediator(group, gre)
            meta = {
                "eval_method": "sync",
                "recommender": gre.__class__.__name__,
            }
            return med, meta
        return factory


class HybridMediatorFactoryBuilder:
    """Fluent builder for ``ConsensusMediatorHybridApproach`` (per-user async branch + group sync branch)."""

    def __init__(self):
        # general (per-user) větev (async-like)
        self._mk_general_recommender = None
        self._mk_priority = None
        self._mk_threshold_static = None
        self._mk_threshold_sigmoid = None
        self._mk_redistrib = None

        # group (sync-like) větev
        self._mk_group_algorithm = None
        self._mk_group_recommender = None

        # mediator
        self._mk_mediator = None

    # --- general (per-user) větev ---

    def with_general_recommender_engine(self, f):
        # f: (group) -> general_recommender
        self._mk_general_recommender = f
        return self

    def with_priority_function(self, f):
        # f: (group, ...) -> priority_function
        self._mk_priority = f
        return self

    # aliasy pro threshold policy (stejné chování jako u async):
    def with_threshold_policy(self, f):
        # f: () -> threshold_policy
        self._mk_threshold_static = f
        return self

    def with_sigmoid_policy(self, f):
        # f: (redistribution_unit) -> threshold_policy
        self._mk_threshold_sigmoid = f
        return self

    def with_redistribution(self, f):
        # f: (group, priority_fn) -> RedistributionUnit
        self._mk_redistrib = f
        return self

    # --- group (sync) větev ---

    def with_group_algorithm(self, f):
        # f: () -> group_algorithm
        self._mk_group_algorithm = f
        return self

    def with_group_recommender_engine(self, f):
        # f: (group, group_algorithm) -> group_recommender
        self._mk_group_recommender = f
        return self

    # --- mediator ---

    def with_mediator(self, f):
        """
        f: (group, general_recommender, group_recommender, redistribution_unit, threshold_policy) -> mediator
        Pozn.: v lambdě si můžeš předat i window_size a first_round_ratio přes defaultní parametry.
        """
        self._mk_mediator = f
        return self

    def build(self):
        # rychlá validace povinných větví
        missing = []
        if not self._mk_general_recommender:
            missing.append("general_recommender")
        if not self._mk_group_recommender:
            missing.append("group_recommender")
        if not self._mk_mediator:
            missing.append("mediator")
        if not self._mk_redistrib:
            missing.append("redistribution")
        if not (self._mk_threshold_sigmoid or self._mk_threshold_static):
            missing.append("threshold_policy (sigmoid nebo static)")
        if missing:
            raise ValueError("HybridMediatorFactoryBuilder: chybí části: " + ", ".join(missing))

        def factory(group):
            # group větev (sync-like)
            ga = self._mk_group_algorithm() if self._mk_group_algorithm else None
            gre = self._mk_group_recommender(group, ga) if ga is not None else self._mk_group_recommender(group, None)

            # general větev (async-like)
            gen = self._mk_general_recommender(group)
            pr  = self._mk_priority(group) if self._mk_priority else None
            ru  = self._mk_redistrib(group, pr)

            if self._mk_threshold_sigmoid:
                th = self._mk_threshold_sigmoid(ru)
            else:
                th = self._mk_threshold_static()

            # mediator: předáme vše; lambda si může vyzobnout users_ids a hyperparametry
            med = self._mk_mediator(group, gen, gre, ru, th)

            # metadata – sjednocený styl
            meta: Dict[str, Any] = {
                "eval_method": "hybrid",
                "recommender_general": getattr(gen, "__class__", type(gen)).__name__,
                "recommender_group": getattr(gre, "__class__", type(gre)).__name__,
                "priority_function": getattr(pr, "get_metadata", lambda: getattr(pr, "__class__", type(pr)).__name__)() if pr is not None else None,
                "threshold_policy": getattr(th, "get_metadata", lambda: getattr(th, "__class__", type(th)).__name__)(),
            }
            return med, meta

        return factory


class HybridMediatorUpdatableFactoryBuilder:
    """
    """

    def __init__(self):
        # general (per-user) větev (async-like)
        self._mk_general_recommender = None
        self._mk_updatable_recommender_engine = None
        self._mk_priority = None
        self._mk_threshold_static = None
        self._mk_threshold_sigmoid = None
        self._mk_redistrib = None

        # group (sync-like) větev
        self._mk_group_algorithm = None
        self._mk_group_updatable_model = None
        self._mk_group_recommender_for_first_round = None

        # mediator
        self._mk_mediator = None

    # --- general (per-user) větev ---

    def with_general_recommender_engine(self, f):
        # f: (group) -> general_recommender
        self._mk_general_recommender = f
        return self

    def with_updatable_group_rec_model(self, f):
        self._mk_group_updatable_model = f
        return self

    def with_general_recommender_engine_updatable(self, f):
        self._mk_updatable_recommender_engine = f
        return self

    def with_priority_function(self, f):
        # f: (group, ...) -> priority_function
        self._mk_priority = f
        return self

    # aliasy pro threshold policy (stejné chování jako u async):
    def with_threshold_policy(self, f):
        # f: () -> threshold_policy
        self._mk_threshold_static = f
        return self

    def with_sigmoid_policy(self, f):
        # f: (redistribution_unit) -> threshold_policy
        self._mk_threshold_sigmoid = f
        return self

    def with_redistribution(self, f):
        # f: (group, priority_fn) -> RedistributionUnit
        self._mk_redistrib = f
        return self


    # --- group (sync) větev ---
    def with_group_updatable_algorithm(self, f):
        self._mk_group_updatable_model = f
        return self

    def with_group_algorithm(self, f):
        # f: () -> group_algorithm
        self._mk_group_algorithm = f
        return self

    def with_group_recommender_engine(self, f):
        # f: (group, group_algorithm) -> group_recommender
        self._mk_group_recommender_for_first_round = f
        return self

    # --- mediator ---

    def with_mediator(self, f):
        """
        f: (group, general_recommender, group_recommender, redistribution_unit, threshold_policy) -> mediator
        Pozn.: v lambdě si můžeš předat i window_size a first_round_ratio přes defaultní parametry.
        """
        self._mk_mediator = f
        return self

    def build(self):
        # rychlá validace povinných větví
        missing = []
        if not self._mk_group_updatable_model:
            missing.append("updatable_group_model")
        if not self._mk_group_recommender_for_first_round:
            missing.append("group_recommender")
        if not self._mk_mediator:
            missing.append("mediator")
        if not self._mk_redistrib:
            missing.append("redistribution")
        if not (self._mk_threshold_sigmoid or self._mk_threshold_static):
            missing.append("threshold_policy (sigmoid nebo static)")
        if missing:
            raise ValueError("HybridMediatorFactoryBuilder: chybí části: " + ", ".join(missing))

        def factory(group):
            # group branch (sync-like)
            ga = self._mk_group_algorithm() if self._mk_group_algorithm else None
            gre = self._mk_group_recommender_for_first_round(group, ga) if ga is not None else self._mk_group_recommender_for_first_round(group, None)

            gr_updatable_model = self._mk_group_updatable_model()
            gr_updatable = self._mk_updatable_recommender_engine(group, gr_updatable_model)

            # general branch (async-like)
            pr  = self._mk_priority(group) if self._mk_priority else None
            ru  = self._mk_redistrib(group, pr)

            if self._mk_threshold_sigmoid:
                th = self._mk_threshold_sigmoid(ru)
            else:
                th = self._mk_threshold_static()

            # mediator: we put everything
            med = self._mk_mediator(group, gr_updatable, gre, ru, th)

            # metadata
            meta: Dict[str, Any] = {
                "eval_method": "hybrid",
                "updatable_recommender": getattr(gr_updatable, "__class__", type(gr_updatable)).__name__,
                "recommender_group": getattr(gre, "__class__", type(gre)).__name__,
                "priority_function": getattr(pr, "get_metadata", lambda: getattr(pr, "__class__", type(pr)).__name__)() if pr is not None else None,
                "threshold_policy": getattr(th, "get_metadata", lambda: getattr(th, "__class__", type(th)).__name__)(),
            }
            return med, meta

        return factory