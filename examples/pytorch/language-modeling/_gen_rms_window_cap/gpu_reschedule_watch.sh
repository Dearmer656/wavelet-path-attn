#!/bin/bash
# Watch for idle GPU nodes; if our regular (non-lang01) quota isn't maxed at 16
# and a pending job of ours requests a GPU type with zero idle nodes anywhere
# while a DIFFERENT type has idle capacity, retarget that pending job's GRES
# in-place (scontrol update, no cancel/resubmit needed) to the idle type so it
# can actually get scheduled. Reports every action taken.
set -u

QUOTA_CAP=16
CHECK_INTERVAL=120
PREFERENCE_ORDER="6000 a100 a6000 3090"

gpu_type_free_count() {
  # $1 = gres type substring (e.g. a6000, a100, 6000, 3090). Sums free GPUs
  # (Total-Used) across idle/mixed nodes advertising that gres type.
  sinfo -N -o "%N %G" --noheader 2>/dev/null | grep -i "gpu:${1}" | while read -r node gres; do
    total=$(echo "$gres" | grep -oiE "gpu:${1}:[0-9]+" | grep -oE "[0-9]+$")
    used=$(squeue -h -o "%N %b" -t RUNNING -w "$node" 2>/dev/null | awk -v t="$1" '
      $0 ~ ("gpu:" t) { n=split($0,a,"gpu:"); split(a[2],g,/[:,]/); for(i=1;i<=length(g);i++) if (g[i] ~ /^[0-9]+$/) {print g[i]; break} }' | awk '{s+=$1} END{print s+0}')
    [ -z "$total" ] && total=0
    echo $(( total - used ))
  done | awk '{s+=$1} END{print s+0}'
}

my_regular_gpu_count() {
  squeue -u "$USER" -h -o "%N %b" -t RUNNING 2>/dev/null | awk '
    $1 != "lang01" {
      n = split($0, parts, "gpu:")
      if (n > 1) {
        split(parts[2], g, /[:,]/)
        for (i=1;i<=length(g);i++) if (g[i] ~ /^[0-9]+$/) { print g[i]; break }
      }
    }' | awk '{s+=$1} END{print s+0}'
}

echo "$(date '+%F %T') gpu_reschedule_watch started, polling every ${CHECK_INTERVAL}s, cap=${QUOTA_CAP}"

while true; do
  used=$(my_regular_gpu_count)
  if [ "$used" -lt "$QUOTA_CAP" ]; then
    room=$(( QUOTA_CAP - used ))
    pending_jobs=$(squeue -u "$USER" -h -t PENDING -o "%i|%b|%r" 2>/dev/null)
    if [ -n "$pending_jobs" ]; then
      while IFS='|' read -r jid greq reason; do
        [ -z "$jid" ] && continue
        [[ "$reason" == *Dependency* ]] && continue
        req_type=$(echo "$greq" | grep -oE "gpu:[a-zA-Z0-9]+" | head -1 | cut -d: -f2)
        req_count=$(echo "$greq" | grep -oE "gpu:[a-zA-Z0-9]+:[0-9]+" | head -1 | grep -oE "[0-9]+$")
        [ -z "$req_type" ] || [ -z "$req_count" ] && continue
        [ "$req_count" -gt "$room" ] && continue
        req_free=$(gpu_type_free_count "$req_type")
        if [ "$req_free" -ge "$req_count" ]; then
          continue  # requested type already has enough idle capacity somewhere; SLURM will place it
        fi
        for alt in $PREFERENCE_ORDER; do
          [ "$alt" = "$req_type" ] && continue
          alt_free=$(gpu_type_free_count "$alt")
          if [ "$alt_free" -ge "$req_count" ]; then
            old_greq="$greq"
            if scontrol update JobId="$jid" Gres="gpu:${alt}:${req_count}" 2>/tmp/gpu_reschedule_err_${jid}.txt; then
              echo "$(date '+%F %T') RETARGETED job $jid: ${old_greq} -> gpu:${alt}:${req_count} (${alt} had ${alt_free} free, ${req_type} had ${req_free})"
            else
              echo "$(date '+%F %T') RETARGET FAILED job $jid: $(cat /tmp/gpu_reschedule_err_${jid}.txt 2>/dev/null)"
            fi
            break
          fi
        done
      done <<< "$pending_jobs"
    fi
  fi
  sleep "$CHECK_INTERVAL"
done
