#!/usr/bin/env bash
# Minimal logger do stats.md v Git repu (nebo libovolné složce)
# Použití:
#   ./eval_log.sh FILE STATUS [-c CODE] [-s START_ISO] [-e END_ISO] [-d DURATION_SEC] [-n "POZNÁMKA"] [-r REPO_DIR] [-o STATS_FILE] [-g]
# Příklady:
#   ./eval_log.sh eval_foo.json success -s 2025-08-29T10:00:00 -e 2025-08-29T10:07:42 -n "baseline"
#   ./eval_log.sh eval_bar.json fail -c 137 -d 55.2 -n "OOM"
#   ./eval_log.sh eval_baz.json running

set -euo pipefail

FILE="${1:-}"; shift || true
STATUS="${1:-}"; shift || true

# defaulty
CODE=""
START=""
END=""
DUR=""
NOTES=""
REPO_DIR="$HOME/projects/bc/text2/analysis/log_repo"
STATS_FILE="stats.md"
DO_GIT=false

# parse flags
while getopts ":c:s:e:d:n:r:o:g" opt; do
    case "$opt" in
        c) CODE="$OPTARG" ;;
        s) START="$OPTARG" ;;
        e) END="$OPTARG" ;;
        d) DUR="$OPTARG" ;;
        n) NOTES="$OPTARG" ;;
        r) REPO_DIR="$OPTARG" ;;
        o) STATS_FILE="$OPTARG" ;;
        g) DO_GIT=true ;;
        \?) echo "Neznámý přepínač -$OPTARG" >&2; exit 2 ;;
        :)  echo "Přepínač -$OPTARG vyžaduje hodnotu" >&2; exit 2 ;;
    esac
done

if [[ -z "$FILE" || -z "$STATUS" ]]; then
  echo "Použití: $0 FILE STATUS [-c CODE] [-s START] [-e END] [-d DURATION_SEC] [-n NOTES] [-r REPO_DIR] [-o STATS_FILE] [-g]" >&2
  exit 2
fi

# mapování na emoji (velmi jednoduše)
lower_status="$(echo "$STATUS" | tr '[:upper:]' '[:lower:]')"
case "$lower_status" in
  success|ok|passed|pass) EMOJI="✅" ;;
  fail|failed|error)      EMOJI="❌" ;;
  *)                      EMOJI="🚧" ;;
esac

# jednoduchý výpočet duration, pokud není a máme start+end
# (očekává ISO 8601; 'date' to ve většině shellů vezme)
if [[ -z "$DUR" && -n "$START" && -n "$END" ]]; then
  if START_S=$(date -d "$START" +%s 2>/dev/null) && END_S=$(date -d "$END" +%s 2>/dev/null); then
    DUR=$(( END_S - START_S ))
  else
    DUR=""
  fi
fi

# hezký formát duration mm:ss nebo hh:mm:ss
fmt_duration() {
  local s="${1:-}"; [[ -z "$s" ]] && { echo ""; return; }
  s=$(printf "%.0f" "$s")
  local h=$(( s/3600 )); local m=$(( (s%3600)/60 )); local sec=$(( s%60 ))
  if (( h > 0 )); then printf "%02d:%02d:%02d" "$h" "$m" "$sec"; else printf "%02d:%02d" "$m" "$sec"; fi
}

LOGGED_AT="$(date -Iseconds)"
DUR_HUMAN="$(fmt_duration "${DUR:-}")"

mkdir -p "$REPO_DIR"
PATH_MD="$REPO_DIR/$STATS_FILE"
touch "$PATH_MD"

# založ hlavičku, pokud chybí
if ! grep -qE '^\| *File *\|' "$PATH_MD"; then
  {
    echo "# Evaluation Stats"
    echo
    echo "| File | Status | Code | Started | Ended | Duration | Logged at | Notes |"
    echo "|:-----|:------:|-----:|:--------|:------|:---------|:----------|:------|"
  } >> "$PATH_MD"
fi

# přidej nový řádek na konec (nejjednodušší a spolehlivé)
# (pokud chceš nejnovější nahoře, můžu pak přidat „prepend“ verzi)
printf "| %s | %s %s | %s | %s | %s | %s | %s | %s |\n" \
  "$FILE" "$EMOJI" "$lower_status" \
  "${CODE}" \
  "${START}" \
  "${END}" \
  "${DUR_HUMAN}" \
  "${LOGGED_AT}" \
  "$(echo "$NOTES" | tr '|' '/')" >> "$PATH_MD"

# volitelný git add/commit/push (-g)
if $DO_GIT && [[ -d "$REPO_DIR/.git" ]]; then
  (cd "$REPO_DIR" && git add "$STATS_FILE" && git commit -m "log: $FILE $lower_status" && git push) || \
    echo "Upozornění: git push selhal (pokračuju)."
fi

echo "Zapsáno do $PATH_MD"