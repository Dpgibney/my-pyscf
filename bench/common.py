"""Shared helpers for run_bench.py / validate.py: configs, pinned worker environment, provenance."""
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

BENCH = Path(__file__).resolve().parent
WS = BENCH.parent
SCRATCH = WS / 'scratch'
RESULTS = BENCH / 'results'
REPOS = ['.', 'pyscf', 'pyscf-base', 'libcint', 'qcint', 'libxc', 'OpenBLAS']
sys.path.insert(0, str(BENCH))
from worker import MARK  # noqa: E402


@dataclass
class Config:
    label: str
    tree: str                      # PySCF tree directory in the workspace, or 'wheel'
    env: dict = field(default_factory=dict)

    @classmethod
    def parse(cls, spec):
        '''LABEL=TREE[:VAR=VAL,VAR=VAL...]'''
        label, rest = spec.split('=', 1)
        tree, _, envs = rest.partition(':')
        env = dict(kv.split('=', 1) for kv in envs.split(',') if kv)
        if tree != 'wheel' and not (WS / tree / 'pyscf' / '__init__.py').exists():
            raise SystemExit(f'unknown PySCF tree: {tree}')
        return cls(label, tree, env)

    def python(self):
        return str(WS / ('.venv-wheel' if self.tree == 'wheel' else '.venv') / 'bin' / 'python')

    def variant(self):
        if self.tree == 'wheel':
            return 'pip wheel'
        f = WS / self.tree / 'pyscf' / 'lib' / 'deps' / 'VARIANT'
        return f.read_text().replace('\n', ' ').strip() if f.exists() else 'unbuilt'


def worker_env(cfg, threads):
    '''Environment with every knob that changes timings pinned; nothing inherited from the shell.'''
    drop = ('OMP_', 'OPENBLAS_', 'GOTO', 'MKL_', 'KMP_', 'PYSCF_', 'CONDA', 'LD_')
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(drop) and k not in ('PYTHONPATH', 'VIRTUAL_ENV', 'PYTHONHOME')}
    SCRATCH.mkdir(exist_ok=True)
    env.update(OMP_NUM_THREADS=str(threads), PYSCF_MAX_MEMORY='4000', PYSCF_TMPDIR=str(SCRATCH),
               PYSCF_CONFIG_FILE=str(BENCH / 'pyscf_conf.py'), PYTHONNOUSERSITE='1')
    if cfg.tree != 'wheel':
        env['PYTHONPATH'] = str(WS / cfg.tree)
    env.update(cfg.env)
    return env


def run_script(script, args, cfg, threads, timeout=7200):
    '''Run a bench/ script in the configuration's pinned environment; raise with stderr on failure.'''
    cmd = [cfg.python(), str(BENCH / script)] + list(args)
    p = subprocess.run(cmd, env=worker_env(cfg, threads), cwd=SCRATCH, capture_output=True,
                       text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f'{script} {args} [{cfg.label}] failed (rc={p.returncode}):\n{p.stderr[-4000:]}')
    return p


def run_worker(case, cfg, threads, tight=False, timeout=7200):
    p = run_script('worker.py', [case] + (['--tight'] if tight else []), cfg, threads, timeout)
    for line in p.stdout.splitlines():
        if line.startswith(MARK):
            rec = json.loads(line[len(MARK):])
            check_libs(rec, cfg)
            return rec
    raise RuntimeError(f'{case} [{cfg.label}] failed (rc={p.returncode}):\n{p.stderr[-4000:]}')


def check_libs(rec, cfg):
    '''A source tree must load cint/xc/BLAS from our installs, never from the system.'''
    if cfg.tree == 'wheel':
        return
    for path in rec['libs']:
        name = os.path.basename(path)
        if name.startswith(('libopenblas', 'libcint', 'libxc.', 'libblas')) and \
                not path.startswith(str(WS / 'install')):
            raise RuntimeError(f'{cfg.label}: {name} loaded from {path}, not from {WS / "install"}')


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def machine_state():
    cpu = Path('/sys/devices/system/cpu')
    ready = subprocess.run([str(WS / 'scripts' / 'check_power.sh')], capture_output=True, text=True)
    return {
        'ac': _read('/sys/class/power_supply/AC/online') or _read('/sys/class/power_supply/ADP1/online'),
        'profile': _read('/sys/firmware/acpi/platform_profile'),
        'scaling_max_freq': {c: _read(cpu / f'cpu{c}/cpufreq/scaling_max_freq') for c in (0, 8)},
        'epp': _read(cpu / 'cpu0/cpufreq/energy_performance_preference'),
        'loadavg': _read('/proc/loadavg'),
        'benchmark_ready': ready.returncode == 0,
    }


def repo_shas():
    out = {}
    for repo in REPOS:
        d = WS / repo
        if not (d / '.git').exists():
            continue
        sha = subprocess.run(['git', '-C', str(d), 'rev-parse', '--short=10', 'HEAD'],
                             capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', str(d), 'status', '--porcelain', '--untracked-files=no'],
                               capture_output=True, text=True).stdout.strip()
        out['workspace' if repo == '.' else repo] = sha + ('-dirty' if dirty else '')
    return out
