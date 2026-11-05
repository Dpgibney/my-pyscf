"""The memory of a group of processes (this user's, whose executable lies under one of --exe-prefix), sampled every
--interval s until --pid exits: the sum of their RSS and the sum of their PSS (shared pages split between the
processes that map them, so the shared memory of MPI ranks counts once). Prints a line for each new PSS peak and the
peaks at the end. With --limit-gb, kills the group by PID (SIGTERM, then SIGKILL) once the PSS sum passes it.
Usage: rss_watch.py --pid PID --exe-prefix DIR [--exe-prefix DIR ...] [--interval 2] [--limit-gb 13.5]"""
import argparse
import os
import signal
import time

ap = argparse.ArgumentParser()
ap.add_argument('--pid', type=int, required=True)
ap.add_argument('--exe-prefix', action='append', required=True)
ap.add_argument('--interval', type=float, default=2.)
ap.add_argument('--limit-gb', type=float, default=None)
args = ap.parse_args()
uid = os.getuid()
prefixes = tuple(os.path.realpath(p) for p in args.exe_prefix)


def group():
    for d in os.listdir('/proc'):
        if not d.isdigit():
            continue
        try:
            if os.stat(f'/proc/{d}').st_uid == uid and os.readlink(f'/proc/{d}/exe').startswith(prefixes):
                yield int(d)
        except OSError:
            continue


def mem_kb(pid):
    rss = pss = 0
    try:
        with open(f'/proc/{pid}/smaps_rollup') as f:
            for line in f:
                if line.startswith('Rss:'):
                    rss = int(line.split()[1])
                elif line.startswith('Pss:'):
                    pss = int(line.split()[1])
    except OSError:
        pass
    return rss, pss


t0 = time.time()
peak_pss = peak_rss = 0.
peak_n = 0
while os.path.exists(f'/proc/{args.pid}'):
    pids = list(group())
    rss = pss = 0
    for p in pids:
        r, s = mem_kb(p)
        rss += r
        pss += s
    rss, pss = rss / 1024**2, pss / 1024**2
    peak_rss = max(peak_rss, rss)
    if pss > peak_pss + 0.05:
        print(f'[{time.time() - t0:8.0f} s] PSS {pss:6.2f} GB, RSS {rss:6.2f} GB in {len(pids)} processes', flush=True)
    peak_pss = max(peak_pss, pss)
    peak_n = max(peak_n, len(pids))
    if args.limit_gb and pss > args.limit_gb:
        print(f'WATCHDOG: PSS {pss:.2f} GB > {args.limit_gb} GB, killing {pids}', flush=True)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for p in pids:
                try:
                    os.kill(p, sig)
                except OSError:
                    pass
            time.sleep(5)
        break
    time.sleep(args.interval)
print(f'peak PSS {peak_pss:.2f} GB, peak RSS sum {peak_rss:.2f} GB, up to {peak_n} processes, '
      f'{time.time() - t0:.0f} s', flush=True)
