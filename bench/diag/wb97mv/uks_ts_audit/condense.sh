#!/bin/bash
# condense.sh LOG > OUT: the lines of a ts_workflow_profile.py or probe log worth keeping (its ">>>" lines, the tables
# of parts, the CPHF cycles, the integral statistics, the guard's lines, the first 60 lines of a cProfile table), with
# the workspace's path, scratch paths and the host name taken out
ws="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
{
grep -E '^>>> \[|^ +[0-9.]+ s +[0-9]+ call|^CPHF (cycle|:)|^CPHF response with|^hess_jk_energy:|^guard:|^two-level SCF:|^reference [0-9]|^    (two-electron second derivatives.*done|CPU time for)|^(one|two) density|^tol 1e|^long-range K second|^[0-9]+ atoms|^    rss|^(C-PCM|IEF-PCM|SS\(V\)PE|SMD): E = |^second derivatives|^gradient, ' "$1"
sed -n '/function calls/,$p' "$1" | awk 'NR<=66' | grep -v '^>>> '
} | sed -e "s#$ws/##g" -e 's#scratch/audit-snap[0-9]/pyscf/pyscf/##g' -e 's#/tmp/[^ ]*/audit/##g' \
        -e 's#/[^ ]*/lib/python3.12/site-packages/##g' -e 's#/[^ ]*/lib/python3.12/##g' -e "s/$(hostname)/laptop/g"
