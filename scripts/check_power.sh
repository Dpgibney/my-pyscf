#!/usr/bin/env bash
# Print the machine state that affects timings; exit 1 if it is not benchmark-ready.
# Benchmark-ready = on AC, performance profile, frequency cap lifted. To get there:
#   powerprofilesctl set performance; sudo cpupower frequency-set -u 5.1GHz
cpu=/sys/devices/system/cpu
ac=$(cat /sys/class/power_supply/{AC,ADP}*/online 2>/dev/null | head -1)
profile=$(powerprofilesctl get 2>/dev/null || cat /sys/firmware/acpi/platform_profile)
cap0=$(cat $cpu/cpu0/cpufreq/scaling_max_freq) max0=$(cat $cpu/cpu0/cpufreq/cpuinfo_max_freq)
echo "ac=$ac profile=$profile cpu0_scaling_max=${cap0} cpu0_hw_max=${max0} epp=$(cat $cpu/cpu0/cpufreq/energy_performance_preference)"
ok=1
[[ $ac == 1 ]] || { echo "  not on AC power"; ok=0; }
[[ $profile == performance ]] || { echo "  power profile is '$profile' (want performance)"; ok=0; }
(( cap0 >= max0 )) || { echo "  frequency capped at $((cap0/1000)) MHz (hw max $((max0/1000)) MHz)"; ok=0; }
(( ok )) && echo "benchmark-ready" || { echo "NOT benchmark-ready"; exit 1; }
