# Příprava dat pro group / consensus evaluaci

Tento adresář obsahuje **logiku přípravy dat**. Evaluační skripty v `evaluations/` ji většinou **nepouštějí přímo** – načítají hotové pickle z `analysis/cache/` (nebo je nechají dopočítat přes `load_or_build_pickle`).

## Dvě fáze (důležité)

### 1) „Těžká“ příprava (jednou, ručně / Makefile)

**Vstup:** `python -m evaluation_frameworks.consensus_evaluation.evaluation.evaluation_preparation.eval_dataset_preparation`  
(nebo Makefile target `consensus-eval-dataset-gen` – zkontroluj, že `--group-size` je skutečně *velikost skupiny*, ne počet skupin.)

**Co to dělá (soubor `eval_dataset_preparation.py`):**

- Načte MovieLens (32M), vyfiltruje CSR (`load_filtered_dataset`).
- Natrénuje / načte **LightFM**, vytáhne **embeddingy uživatelů** (`model_train_load.train_or_load_lightfm_model`, `EmbeddingExtractor`).
- Vygeneruje syntetické **skupiny** (similar / outlier / random / divergent pro velikost 3, nebo random pro větší skupiny) a uloží je jako pickle (`groups-*.pkl`).
- Rozseká skupiny na **train / val / test** split (`GroupsEvaluationSetsSplitter` → `group-eval-*.pkl`).

Bez této fáze chybí soubory `groups-*.pkl` / `group-eval-*.pkl`, které očekává `evaluation_context_factory.py`.

### 2) Ground truth a filtrovaná matice (často „lazy“, při prvním běhu evaluace)

**Kde:** `ground_truth_filtering.py` – funkce typu `prepare_group_eval_data2_test_split` (a související helpery).

**Kdo to volá:** `evaluations/evaluation_context_factory.py` uvnitř `build_context_holdout` / `build_context_large_holdout` – obvykle zabaleno v `load_or_build_pickle` s názvem závislým na typu splitu, počtu skupin, `min_common`, případně `group_size`.

**Co to dělá:**

- Pro vybrané skupiny spočítá **společné (a per-user) ground truth** položky, případně **hold-out** část hodnocení.
- Vyrobit **filtrovanou CSR** pro trénink simulace / modelů tak, aby testové interakce neunikly do tréninku (záleží na konkrétní funkci).

Výsledek se zase cachuje jako pickle v `cache/` – při opakovaných bězích je to hlavně **načtení z disku**, ne přepočet.

## `model_train_load.py`

Sdílené **natrénování nebo načtení** modelů:

- `train_or_load_lightfm_model` – používá se při generování skupin (embeddingy).
- `train_or_load_easer_model` – typicky při běhu evaluace (`Runner`), ne při `eval_dataset_preparation`.

## Shrnutí „co spustit kdy“

| Kdy | Co |
|-----|-----|
| Poprvé / po změně parametrů skupin | `eval_dataset_preparation` (Makefile `consensus-eval-dataset-gen`) |
| Běžná evaluace / tuning | `python -m …evaluations…` – sama dožene chybějící GT pickle přes factory + `load_or_build_pickle` |
| Ladění algoritmu | Stejné eval skripty; bottleneck je často počet kombinací gridu a velikost eval setu, ne tento adresář |

## Úklid kódu

Refaktor uvnitř tohoto adresáře (přejmenování funkcí, tenčí wrapper) **nerozbije** zbytek aplikace, pokud zůstanou:

- **stejné názvy pickle** (nebo se aktualizuje `evaluation_context_factory` a případně se smaže stará cache),  
- **stejné signatury** funkcí, které volá `evaluation_context_factory`.

Čistší varianta je např. jeden modul `prepare_datasets.py`, který jen volá existující funkce z `eval_dataset_preparation` – čistě organizační změna.
