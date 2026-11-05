"""Print the physical cores and the memory (MB, 10^6 bytes) that this process may use, for pyscf-env.sh's defaults
and cpu_setup.sh's checks.
- Cores: those with a hardware thread in the CPU affinity mask, which a Slurm job's allocation (task/affinity, or
  task/cgroup's cpuset), taskset and numactl restrict; at most $SLURM_CPUS_PER_TASK (else $SLURM_CPUS_ON_NODE),
  for clusters that do not bind jobs to their CPUs.
- Memory: the least of the RAM, the cgroup memory limits from this process's cgroup up (a Slurm job's memory when
  Slurm constrains RAM, containers, systemd scopes), and Slurm's --mem or --mem-per-cpu.
"""
import os


def cores():
    phys = set()
    for c in os.sched_getaffinity(0):
        try:
            with open(f'/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list') as f:
                phys.add(f.read().strip())
        except OSError:
            phys.add(str(c))
    n = len(phys)
    slurm = os.environ.get('SLURM_CPUS_PER_TASK') or os.environ.get('SLURM_CPUS_ON_NODE') or ''
    if slurm.isdigit() and 0 < int(slurm) < n:
        n = int(slurm)
    return n


def memory_mb():
    with open('/proc/meminfo') as f:
        limits = [int(line.split()[1]) * 1024 for line in f if line.startswith('MemTotal:')]
    with open('/proc/self/cgroup') as f:
        for line in f:
            hierarchy, controllers, path = line.rstrip('\n').split(':', 2)
            if hierarchy == '0':                            # cgroup v2
                root, names = '/sys/fs/cgroup', ('memory.max', 'memory.high')
            elif 'memory' in controllers.split(','):        # cgroup v1
                root, names = '/sys/fs/cgroup/memory', ('memory.limit_in_bytes',)
            else:
                continue
            d = os.path.normpath(root + path)
            while d.startswith(root):
                for name in names:
                    try:
                        with open(os.path.join(d, name)) as g:
                            v = g.read().strip()
                    except OSError:
                        continue
                    if v.isdigit() and int(v) > 0:
                        limits.append(int(v))
                d = os.path.dirname(d)
    env = os.environ
    mib = 1 << 20                                           # Slurm's MB
    per_node, per_cpu, ncpu = (env.get(k, '') for k in ('SLURM_MEM_PER_NODE', 'SLURM_MEM_PER_CPU',
                                                         'SLURM_CPUS_ON_NODE'))
    if per_node.isdigit() and int(per_node) > 0:            # --mem=0 is all of the node's memory
        limits.append(int(per_node) * mib)
    elif per_cpu.isdigit() and ncpu.isdigit() and int(per_cpu) > 0:
        limits.append(int(per_cpu) * int(ncpu) * mib)
    return min(limits) // 10**6


print(cores(), memory_mb())
