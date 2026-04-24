# Kde jsou uložené výsledky evaluací (`autorun`, `safe_eval_res`, `load_eval_res`)

## Kořen

- Globální cache složka: `analysis/cache/` (Python: `utils.config.CACHE_FILES_DIR`).
- Výsledky consensus eval skriptů: **`cache/cons_evaluations/`**

## Aktuální hierarchie (čitelné názvy parametrů)

Každá úroveň nese **prefix**, aby v seznamu složek bylo hned vidět, co je co.

### Bez `group_size` (běžný experiment)

```text
cache/cons_evaluations/
  w_<W>/                       # okno; W z --window-size / DEFAULT_W_SIZE
    split_<train|validation|test>/
      <název_modulu>.py/       # soubor experimentu (evaluation_name)
        1.pkl
        2.pkl
```

Příklad: `.../cons_evaluations/w_10/split_validation/eval_async_static_....py/3.pkl`

### S `group_size` (large-group)

Mezi `w_*` a `split_*` je navíc **`group_<n>`** (nahrazuje staré nejasné `large/<n>`):

```text
cache/cons_evaluations/
  w_<W_fixed>/
    group_<group_size>/
      split_<train|validation|test>/
        <název_modulu>.py/
          N.pkl
```

| Segment | Parametr v kódu |
|---------|------------------|
| `w_*` | `str(experiment.w_size)` |
| `group_*` | jen pokud `experiment.group_size is not None` |
| `split_*` | `experiment.eval_type` |
| `*.py` | `experiment.evaluation_name` |
| `N.pkl` | pořadí běhů; další zápis = `max(N)+1`; `--num` vybere konkrétní |

## Legacy layout (jen načítání)

Starší běhy mohou být pod:

```text
cache/cons_evaluations/<W_fixed>/[large/<group_size>/]<split>/<název_modulu>.py/N.pkl
```

`load_eval_res` zkouší **nejprve** novou strukturu, **pak** legacy, takže stará cache zmizet nemusí — nové zápisy jdou už jen do pojmenovaných složek.

## Python: cesta ke složce s pickle

```python
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.config import evaluation_results_dir

# nový layout (stejný jako ukládá safe_eval_res)
p = evaluation_results_dir(
    window_size="10",
    eval_type="validation",
    evaluation_name="eval_async_static_policy_simple_priority_function_group_rec.py",
    group_size=None,
)
# legacy cesta:
p_old = evaluation_results_dir(..., layout="legacy")
```

## CLI (`autorun`)

- **`--window-size`** — consensus window `w` (v kódu `w_size`, cache segment `w_<W>`).
- **`--groups-count`** — kolik skupin projít evaluátorem (odpovídá `Runner.run(..., evaluation_size=…)`). Starý alias: **`--group-eval-size`**.
- **`--group-size`** — jen u *large-group* experimentů: počet uživatelů ve skupině (kardinalita), ne zaměnit s `--groups-count`.

Režimy:

- **`auto`** — načte cache (labeled → legacy), jinak přepočet a uložení do **labeled**.
- **`load`** — jen načtení; chyba, pokud ani jedna varianta neexistuje.
- **`compute`** — přepočet a nový `N.pkl` pod **labeled**.

---

Ostatní soubory v `cache/` mimo `cons_evaluations/` jsou typicky příprava dat / GT — nejsou to výstupy těchto eval skriptů.
