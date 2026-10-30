"""Attribute libc memset/memmove leaf samples (and other leaves) to their nearest non-libc caller."""
import collections, re, sys
fn = sys.argv[1]
# libc offsets: memset 0x1b1200-0x1b13c0 region; memmove 0x1b0900-0x1b0e00 region (from objdump)
def classify(frame):
    m = re.match(r'(0x[0-9a-f]+) \(.*libc\.so\.6', frame)
    if not m:
        return None
    return int(m.group(1), 16)
total = 0
leafkind = collections.Counter()
parents = collections.defaultdict(collections.Counter)
base = None
lines = open(fn).read().splitlines()
# find libc base from __libc_start_call_main frame (offset 0x2a1ca.. we use the most common bottom libc frame)
bottoms = collections.Counter()
for line in lines:
    stack, _, c = line.rpartition(' ')
    fr = stack.split(';')
    for f in fr[:3]:
        a = classify(f)
        if a: bottoms[a] += int(c)
start_main = bottoms.most_common(1)[0][0]
base = start_main - (start_main & 0xfffff) if False else None
# the earlier native profile showed __libc_start_call_main at base+0x2a601 (0x...82a601), so base = addr - 0x2a601
base = start_main - 0x2a601
def kind(addr):
    off = addr - base
    if 0x1b1200 <= off <= 0x1b13c0: return 'memset'
    if 0x1b0900 <= off <= 0x1b0e40: return 'memmove'
    return 'libc_other@%x' % off
for line in lines:
    stack, _, c = line.rpartition(' ')
    c = int(c); total += c
    fr = stack.split(';')
    a = classify(fr[-1])
    if a is None:
        continue
    k = kind(a)
    leafkind[k] += c
    # nearest non-libc parent(s)
    par = [f for f in fr[:-1] if classify(f) is None]
    chain = ' <- '.join(p.split(' (')[0] + '(' + p.split('(')[-1].split(':')[0].split('/')[-1] for p in reversed(par[-4:]))
    parents[k][chain] += c
print(fn.split('/')[-1], 'total samples', total, 'libc base', hex(base))
for k, n in leafkind.most_common(8):
    print(f'{k}: {n} samples = {100*n/total:.1f}%')
    for ch, m in parents[k].most_common(8):
        print(f'    {100*m/total:5.2f}%  {ch}')
