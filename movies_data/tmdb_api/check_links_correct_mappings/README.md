# MovieLens ↔ TMDB — kontrola a oprava souboru `links.csv`

## Účel

Cílem tohoto podprojektu je sjednotit mapování mezi MovieLens `movieId` a TMDB `tmdbId` před finálním nebo produkčním použitím datasetu (filtrování, dotazy na TMDB API, další kroky pipeline). Oficiální `links.csv` z MovieLens může obsahovat prázdná ID, duplicitní `tmdbId` mezi řádky nebo zastaralá ID bez platného záznamu na TMDB — při načítání metadat pak vznikají chyby, přesměrování nebo prázdné odpovědi API. Podprojekt má za úkol tyto chybné záznamy identifikovat a pokud možno nahradit platnými záznamy.

Výstupem je `out/links_repaired.csv`; soubor lze předat například do `production_filtering/filter_dataset.py` parametrem `--links-filter` (nebo ekvivalentní cestou).

## Požadavky

- Proměnná prostředí `TMDB_API_KEY` u skriptů volajících TMDB HTTP API.

## Rozložení adresáře

| Složka | Určení |
|--------|--------|
| `data/` | Vstupy zkopírované z MovieLens (`links_original.csv`, `movies.csv`). Velké soubory obvykle neverzovat. |
| `out/` | Mezivýstupy a výsledný `links_repaired.csv`. |
| Kořen | Skripty Python, modul `paths.py`, tato dokumentace. |

## Soubory a jejich role

| Soubor | Umístění | Popis |
|--------|----------|--------|
| `links_original.csv` | `data/` | Kopie výchozího MovieLens `links.csv`. |
| `movies.csv` | `data/` | MovieLens `movies.csv` (tituly, rok) — vstup pro kontrolu oprav. |
| `links.csv` | `out/` | Vyčištěné odkazy po `remove_empty_ids.py`. |
| `missing_en_tmdb_ids.csv` | `out/` | Identifikátory s problémem při ověření v TMDB (`validated_tmdb_ids.py`). |
| `repaired.csv` | `out/` | Tabulka přemapování staré → nové `tmdbId` (`find_missing_tmdb_ids.py`). |
| `repaired_filtered.csv` | `out/` | Přemapování po filtrování kolizí (`filter_repaired_ids.py`). |
| `unresolved.csv` | `out/` | Záznamy bez spolehlivého řešení. |
| `links_repaired.csv` | `out/` | Výsledný soubor odkazů (`resolve_missing_tmdb_ids.py`). |

Výchozí cesty jsou v `paths.py` (`DATA_DIR`, `OUT_DIR`); lze je změnit argumenty příkazové řádky (`--links`, `--out`, …).

## Doporučené pořadí spuštění

Spouštět z adresáře `check_links_correct_mappings/`, případně uvádět absolutní cesty.

1. `remove_empty_ids.py` — načíst `data/links_original.csv`, zapsat `out/links.csv`. Volitelně `--dedupe` — odstranit duplicitní `tmdbId` (ponechat první výskyt).
2. `validated_tmdb_ids.py` — ověřit `out/links.csv` vůči TMDB → `out/missing_en_tmdb_ids.csv`.
3. `find_missing_tmdb_ids.py` — doplnit / opravit problematická ID → `out/repaired.csv`, `out/unresolved.csv`.
4. `filter_repaired_ids.py` — zkontrolovat opravy proti titulům → `out/repaired_filtered.csv`.
5. `resolve_missing_tmdb_ids.py` — sloučit opravy do `out/links_repaired.csv`.

Soubory `remove_duplicates*.py` v repozitáři nejsou zahrnuty; duplicity řešit parametrem `remove_empty_ids.py --dedupe`.

## Migrace starého rozložení

Při přechodu z umístění CSV přímo v kořeni tohoto adresáře přesunout vstupy do `data/`, výstupy do `out/`, názvy souborů zachovat.
