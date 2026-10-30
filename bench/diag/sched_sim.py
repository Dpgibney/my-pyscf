"""Simulated makespan (relative to perfect balance) of the schedules considered in It18 for kernels that sum
partial matrices over groups of grid points, on 4 fast + 6 slow threads (speed ratio 1.38 or 1.6), with
random, spatially correlated group costs: static(1), dynamic,4, fixed slots igroup % S each with a matrix, and
streams of 4-group slots (largest-remaining first). Part 2: streams with a stickiness margin m (stay on the
previous stream unless another has more than m slots more left), with stream switches per slot.
Usage: sched_sim.py
"""
import heapq, numpy as np
rng = np.random.default_rng(0)
def run(ngroup, speed_fast, nsim=20):
    speeds = np.array([speed_fast]*4 + [1.0]*6)
    res = {}
    for sim in range(nsim):
        cost = rng.lognormal(0, 0.5, ngroup)          # per-group work
        cost = np.convolve(cost, np.ones(25)/25, 'same') * rng.lognormal(0, 0.3, ngroup)  # spatially correlated
        W = cost.sum(); ideal = W / speeds.sum()
        def greedy(chunks):
            # chunks taken in order by the first free thread
            free = [(0.0, t) for t in range(10)]; heapq.heapify(free); end = 0
            for c in chunks:
                t0, t = heapq.heappop(free); t1 = t0 + c / speeds[t]; end = max(end, t1); heapq.heappush(free, (t1, t))
            return end
        r = {}
        # static(1) round robin
        r['static(1)'] = max(cost[t::10].sum() / speeds[t] for t in range(10))
        # dynamic,4 (non-deterministic reference)
        r['dynamic,4'] = greedy([cost[i:i+4].sum() for i in range(0, ngroup, 4)])
        # S equal interleaved slots (igroup % S), dynamic
        for S in (40, 64):
            r[f'slots igroup%{S}'] = greedy([cost[s::S].sum() for s in range(S)])
        # streams: nbuf contiguous streams of slots of 4 groups; a thread takes the next slot of the unlocked
        # stream with most slots left (one slot in flight per stream)
        for nbuf in (12, 20):
            slots = [cost[i:i+4].sum() for i in range(0, ngroup, 4)]
            ns = len(slots); bounds = [ns * b // nbuf for b in range(nbuf + 1)]
            nxt = bounds[:-1].copy(); end_ = bounds[1:]
            busy_until = [0.0] * nbuf
            free = [(0.0, t) for t in range(10)]; heapq.heapify(free); end = 0
            while True:
                t0, t = heapq.heappop(free)
                cand = [b for b in range(nbuf) if nxt[b] < end_[b]]
                if not cand: break
                avail = [b for b in cand if busy_until[b] <= t0]
                if not avail:   # wait for the earliest stream with work to be released
                    t0 = min(busy_until[b] for b in cand); avail = [b for b in cand if busy_until[b] <= t0]
                b = max(avail, key=lambda b: end_[b] - nxt[b])
                c = slots[nxt[b]]; nxt[b] += 1
                t1 = t0 + c / speeds[t]; busy_until[b] = t1; end = max(end, t1)
                heapq.heappush(free, (t1, t))
            r[f'streams nbuf={nbuf}'] = end
        for k, v in r.items():
            res.setdefault(k, []).append(v / ideal)
    return {k: (np.mean(v), np.max(v)) for k, v in res.items()}
for ngroup, sf in ((1282, 1.6), (2633, 1.6), (2633, 1.38), (6200, 1.6)):
    print(f'ngroup={ngroup} fast/slow={sf}: ' + ', '.join(f'{k} {m:.3f} (max {x:.3f})' for k, (m, x) in run(ngroup, sf).items()))

print('--- stickiness margin')
def sim(ngroup, sf, nbuf, m, nsim=20):
    speeds = np.array([sf]*4 + [1.0]*6)
    out_ms, out_sw = [], []
    for _ in range(nsim):
        cost = rng.lognormal(0, 0.5, ngroup)
        cost = np.convolve(cost, np.ones(25)/25, 'same') * rng.lognormal(0, 0.3, ngroup)
        ideal = cost.sum() / speeds.sum()
        slots = [cost[i:i+4].sum() for i in range(0, ngroup, 4)]
        ns = len(slots); bounds = [ns * b // nbuf for b in range(nbuf + 1)]
        nxt = bounds[:-1].copy(); end_ = bounds[1:]
        busy = [0.0] * nbuf; cur = [-1] * 10
        free = [(0.0, t) for t in range(10)]; heapq.heapify(free); end = 0; switches = 0; taken = 0
        while True:
            t0, t = heapq.heappop(free)
            cand = [b for b in range(nbuf) if nxt[b] < end_[b]]
            if not cand: break
            avail = [b for b in cand if busy[b] <= t0]
            if not avail:
                t0 = min(busy[b] for b in cand); avail = [b for b in cand if busy[b] <= t0]
            best = max(avail, key=lambda b: end_[b] - nxt[b])
            c = cur[t]
            if c in avail and (end_[c] - nxt[c]) >= (end_[best] - nxt[best]) - m:
                b = c
            else:
                b = best
            switches += b != cur[t]; cur[t] = b; taken += 1
            t1 = t0 + slots[nxt[b]] / speeds[t]; nxt[b] += 1; busy[b] = t1; end = max(end, t1)
            heapq.heappush(free, (t1, t))
        out_ms.append(end / ideal); out_sw.append(switches / taken)
    return np.mean(out_ms), np.max(out_ms), np.mean(out_sw)
for ngroup in (600, 2633):
    for m in (0, 1, 2, 4, 8, 1000):
        a, b, s = sim(ngroup, 1.6, 20, m)
        print(f'ngroup={ngroup:5d} nbuf=20 margin={m:4d}: makespan {a:.3f} (max {b:.3f}), switches per slot {s:.2f}')
