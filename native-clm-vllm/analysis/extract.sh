# Usage: bash extract.sh ROOT LABEL  -> TSV: label actor tag episode reward terminated_by
root=$1 label=$2
cd "$root" || exit 1
for s in Qwen3.5-*/*/*/score.json cells/*/*/*/*/score.json; do
  [ -f "$s" ] || continue
  d=$(dirname "$s")
  r=$(grep -oE '"reward": [0-9.eE-]+' "$s" | head -1 | awk '{print $2}')
  t=$(grep -oE '"terminated_by": "[a-z_]+"' "$d/episode.json" 2>/dev/null | head -1 | cut -d'"' -f4)
  case "$d" in cells/*) set -- $(echo "$d" | tr '/' ' '); a=$3 c=$4 e=$5 ;; *) set -- $(echo "$d" | tr '/' ' '); a=$1 c=$2 e=$3 ;; esac
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$a" "$c" "$e" "$r" "$t"
done
