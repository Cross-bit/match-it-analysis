"""
Společná konfigurace metrik pro LaTeX tabulky typu „RFC srovnání“.

``--rfc-metric`` v print skriptech vybíre mezi:
  - rounds_to_consensus — průměrné kolo první shody (klíč v cache ``average``; u async volitelně RFC$_{adj.}$).
  - mean_rank_at_consensus — průměr pozic (1-based) dohodnuté položky v osobním okně
    (klíč ``first_consensus_rank_across_groups``; bez odečítání 1 u async).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, List

RFC_METRIC_CHOICES: Final[List[str]] = [
    "rounds_to_consensus",
    "mean_rank_at_consensus",
    "cards_seen_until_consensus",
]

RFC_METRIC_ARG_HELP: Final[str] = (
    "rounds_to_consensus — průměrné kolo první skupinové shody (výchozí; pole ``average`` v cache). "
    "mean_rank_at_consensus — průměrná 1-based pozice shodné položky v uživatelském výpisu v kole shody "
    "(pole ``first_consensus_rank_across_groups``; nižší = blíž začátku seznamu). "
    "cards_seen_until_consensus — průměrná globální 1-based pozice od začátku session "
    "(kolik karet uživatel viděl do první shody; pole ``first_consensus_global_position_across_groups``)."
)


@dataclass(frozen=True)
class RfcTableMetricSpec:
    storage_key: str
    latex_caption_cs: str
    latex_caption_short: str
    """U async řádků (slug s „A“) odečíst 1 od hodnoty — jen u metriky „kola do shody“."""
    subtract_one_for_async_slug: bool


def resolve_rfc_metric(mode: str) -> RfcTableMetricSpec:
    m = (mode or "rounds_to_consensus").strip()
    if m == "rounds_to_consensus":
        return RfcTableMetricSpec(
            storage_key="average",
            latex_caption_cs="průměrný počet kol do první skupinové shody",
            latex_caption_short="RFC (kola do shody)",
            subtract_one_for_async_slug=False,
        )
    if m == "mean_rank_at_consensus":
        return RfcTableMetricSpec(
            storage_key="first_consensus_rank_across_groups",
            latex_caption_cs="průměrná pozice shodné položky v osobním doporučení při první shodě (1 = nejlepší)",
            latex_caption_short="RFC — průměrná rank pozice při shodě",
            subtract_one_for_async_slug=False,
        )
    if m == "cards_seen_until_consensus":
        return RfcTableMetricSpec(
            storage_key="first_consensus_global_position_across_groups",
            latex_caption_cs=(
                "průměrná globální 1-based pozice první shody od začátku session "
                "(kolik karet uživatel viděl do shody)"
            ),
            latex_caption_short="RFC — karty do první shody (globální pozice)",
            subtract_one_for_async_slug=False,
        )
    raise ValueError(
        f"Neznámý --rfc-metric={mode!r}. Povolené: {', '.join(RFC_METRIC_CHOICES)}"
    )


def add_rfc_metric_arg(parser, *, default: str = "rounds_to_consensus") -> None:
    parser.add_argument(
        "--rfc-metric",
        choices=list(RFC_METRIC_CHOICES),
        default=default,
        metavar="MODE",
        help=RFC_METRIC_ARG_HELP,
    )
