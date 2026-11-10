#!/bin/bash
# guard.sh LIMIT_GB LOG cmd...   -- runs cmd, kills it if its resident memory exceeds LIMIT_GB
lim=$1; log=$2; shift 2
"$@" > "$log" 2>&1 &
pid=$!
peak=0
while kill -0 $pid 2>/dev/null; do
    rss=$(awk '/VmRSS/ {print int($2/1048576)}' /proc/$pid/status 2>/dev/null)
    [ -n "$rss" ] && [ "$rss" -gt "$peak" ] && peak=$rss
    if [ -n "$rss" ] && [ "$rss" -ge "$lim" ]; then
        echo "guard: RSS ${rss} GB >= ${lim} GB, killing $pid" >> "$log"; kill $pid; break
    fi
    sleep 3
done
wait $pid 2>/dev/null
echo "guard: peak RSS about ${peak} GB" >> "$log"
