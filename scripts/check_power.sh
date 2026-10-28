#!/usr/bin/env bash
# Print the machine state that affects timings; exit 1 if it is not fit for benchmarking.
# Fit = on AC with the performance profile. A frequency cap is reported as mode=capped rather than failing:
# at setup this HP laptop's firmware clamped every core to its 2.0 GHz base clock even on AC (cpupower
# could not lift it), and uniform fixed clocks are a stable benchmarking mode. Uncapped would need e.g.
#   powerprofilesctl set performance; sudo cpupower -c all frequency-set -u 5.1GHz
cpu=/sys/devices/system/cpu
ac=$(cat /sys/class/power_supply/{AC,ADP}*/online 2>/dev/null | head -1)
profile=$(powerprofilesctl get 2>/dev/null || cat /sys/firmware/acpi/platform_profile)
cap0=$(cat $cpu/cpu0/cpufreq/scaling_max_freq) max0=$(cat $cpu/cpu0/cpufreq/cpuinfo_max_freq)
mode=$( (( cap0 >= max0 )) && echo full || echo "capped@$((cap0/1000))MHz")
echo "ac=$ac profile=$profile mode=$mode cpu0_hw_max=$((max0/1000))MHz epp=$(cat $cpu/cpu0/cpufreq/energy_performance_preference)"
ok=1
[[ $ac == 1 ]] || { echo "  not on AC power"; ok=0; }
[[ $profile == performance ]] || { echo "  power profile is '$profile' (want performance)"; ok=0; }
(( ok )) && echo "benchmark-ready ($mode)" || { echo "NOT benchmark-ready"; exit 1; }
