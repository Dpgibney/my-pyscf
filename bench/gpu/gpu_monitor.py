"""GPU state during a benchmark (node1): `with GpuMonitor() as mon: ...` samples the clocks, P-state, power,
temperature and throttle reasons with nvidia-smi every PERIOD ms while the block runs; mon.summary then holds
min/median/max SM and memory clocks, the P-states and throttle-reason masks seen, median power and max
temperature. Timings from different sessions or clock states do not compare (node1 ran the same build 20% faster
after a reboot), so every benchmark records this next to its times."""
import statistics
import subprocess
import tempfile

FIELDS = 'clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu,clocks_event_reasons.active'


class GpuMonitor:
    def __init__(self, period_ms=250):
        self.period_ms = period_ms
        self.summary = {}

    def __enter__(self):
        self.out = tempfile.TemporaryFile(mode='w+')
        self.proc = subprocess.Popen(['nvidia-smi', f'--query-gpu={FIELDS}', '--format=csv,noheader,nounits',
                                      f'--loop-ms={self.period_ms}'], stdout=self.out, stderr=subprocess.DEVNULL)
        return self

    def __exit__(self, *exc):
        self.proc.terminate()
        self.proc.wait()
        self.out.seek(0)
        rows = [[x.strip() for x in line.split(',')] for line in self.out.read().splitlines() if line.count(',') == 5]
        self.out.close()
        if rows:
            sm = [float(r[0]) for r in rows]
            mem = [float(r[1]) for r in rows]
            self.summary = {
                'samples': len(rows),
                'sm_mhz': [min(sm), statistics.median(sm), max(sm)],
                'mem_mhz': [min(mem), statistics.median(mem), max(mem)],
                'pstates': sorted({r[2] for r in rows}),
                'power_w': statistics.median(float(r[3]) for r in rows),
                'temp_c_max': max(float(r[4]) for r in rows),
                'reasons': sorted({r[5] for r in rows}),
            }
        return False
