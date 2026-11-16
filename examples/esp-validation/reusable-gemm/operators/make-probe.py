#!/usr/bin/env python3
"""Generate a timeline probe for dma64_tb.sv from a Bambu forward_kernel.v.

usage: make-probe.py <forward_kernel.v> <tb.sv in> <tb.sv out> [<probe.json out>]

Appends to the testbench, before endmodule, an always block that prints one
line per event, with the testbench's cycle counter:
  EV <cycle> TOP start|done
  EV <cycle> RD <index> <length>        DMA read request accepted
  EV <cycle> WR <index> <length>        DMA write request accepted
  EV <cycle> NODE <n> start|done        task node start/done pulses
  EV <cycle> FIFO <id> w|r <data>       FIFO handshakes between nodes
and, per node, cycles with the node's controller out of its idle state are
counted by window (ACT lines) when +probe_activity is given.
"""
import json
import re
import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text()


def module(name):
    m = re.search(r'^module %s\(.*?^endmodule' % re.escape(name), src, re.S | re.M)
    if not m:
        raise SystemExit(f'no module {name}')
    return m.group(0)


def instances(body):
    """(module, instance name, {port: expr}) of each instantiation."""
    out = []
    for m in re.finditer(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(#\s*\((?:[^;]*?)\))?\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\);', body, re.S | re.M):
        mod, inst, ports = m.group(1), m.group(3), m.group(4)
        if mod in ('module', 'input', 'output', 'wire', 'reg', 'assign', 'parameter', 'localparam'):
            continue
        conn = dict(re.findall(r'\.([A-Za-z0-9_]+)\(([^()]*?)\)', ports))
        out.append((mod, inst, conn))
    return out


# Path from forward_kernel down to datapath_forward_kernel.
def find_path(mod, target, depth=0):
    if depth > 4:
        return None
    for sub, inst, _ in instances(module(mod)):
        if sub == target:
            return [inst]
        if sub.startswith(('PBI_forward_kernel', 'forward_kernel', '_forward_kernel')) and 'node' not in sub:
            rest = find_path(sub, target, depth + 1)
            if rest is not None:
                return [inst] + rest
    return None


path = find_path('forward_kernel', 'datapath_forward_kernel')
if path is None:
    raise SystemExit('datapath_forward_kernel not found under forward_kernel')
prefix = 'dut.u_core.' + '.'.join(path) + '.'
dp = module('datapath_forward_kernel')
insts = instances(dp)
fifos, nodes = [], []
for mod, inst, conn in insts:
    if mod == 'dataflow_fifo':
        fifos.append({'instance': inst, 'write': conn['write'], 'read': conn['read'],
                      'din': conn['din'], 'full_n': conn['full_n'], 'empty_n': conn['empty_n']})
    m = re.match(r'PBI_forward_kernel_node(\d+)$', mod)
    if m:
        nodes.append({'node': int(m.group(1)), 'instance': inst,
                      'start': conn['start_port'], 'done': conn['done_port']})
depth = {}
for m in re.finditer(r'dataflow_fifo\s*#\((.*?)\)\s*([A-Za-z0-9_]+)\s*\(', dp, re.S):
    d = re.search(r'FIFO_DEPTH\((\d+)\)', m.group(1))
    depth[m.group(2)] = int(d.group(1)) if d else 1
for f in fifos:
    f['depth'] = depth.get(f['instance'], 1)
    ident = re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', f['din'].strip())
    f['id'] = re.sub(r'FO0_din$', '', f['din']).replace('DF_bambu_', '') if ident else f['instance']
# Which node writes / reads each FIFO: the node instance connected to the wire.
for mod, inst, conn in insts:
    m = re.match(r'PBI_forward_kernel_node(\d+)$', mod)
    if not m:
        continue
    vals = set(v.strip() for v in conn.values())
    for f in fifos:
        if f['write'] in vals:
            f['writer'] = int(m.group(1))
        if f['read'] in vals:
            f['reader'] = int(m.group(1))

lines = ['\n// ---- timeline probe (make-probe.py) ----', 'reg probe_start_q=0, probe_done_q=0;']
for n in nodes:
    lines.append(f'reg probe_n{n["node"]}_s=0, probe_n{n["node"]}_d=0;')
lines.append('always @(posedge clk) if(rst) begin')
lines.append('  probe_start_q<=start; probe_done_q<=done;')
lines.append('  if(start && !probe_start_q) $display("EV %0d TOP start",cycles);')
lines.append('  if(done && !probe_done_q) $display("EV %0d TOP done",cycles);')
lines.append('  if(rv&&rr) $display("EV %0d RD %0d %0d",cycles,ri,rl);')
lines.append('  if(wv&&wr) $display("EV %0d WR %0d %0d",cycles,wi,wl);')
for n in nodes:
    s, d = prefix + n['start'], prefix + n['done']
    i = n['node']
    lines.append(f'  probe_n{i}_s<={s}; probe_n{i}_d<={d};')
    lines.append(f'  if({s} && !probe_n{i}_s) $display("EV %0d NODE {i} start",cycles);')
    lines.append(f'  if({d} && !probe_n{i}_d) $display("EV %0d NODE {i} done",cycles);')
def sig(e):
    e = e.strip()
    return prefix + e if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', e) else e
for f in fifos:
    w, r = sig(f['write']), sig(f['read'])
    lines.append(f'  if({w} && {sig(f["full_n"])}) $display("EV %0d FIFO {f["id"]} w %0d",cycles,{sig(f["din"])});')
    lines.append(f'  if({r} && {sig(f["empty_n"])}) $display("EV %0d FIFO {f["id"]} r",cycles);')
lines.append('end')
tb = Path(sys.argv[2]).read_text()
assert tb.count('endmodule') == 1
Path(sys.argv[3]).write_text(tb.replace('endmodule', '\n'.join(lines) + '\nendmodule'))
info = {'prefix': prefix, 'nodes': nodes, 'fifos': fifos}
if len(sys.argv) > 4:
    Path(sys.argv[4]).write_text(json.dumps(info, indent=2) + '\n')
print(json.dumps({'prefix': prefix, 'nodes': [n['node'] for n in nodes],
                  'fifos': [(f['id'], f.get('writer'), f.get('reader'), f['depth']) for f in fifos]}))
