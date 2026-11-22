"""Actual SASS PC ledger and index-register evaluation for frozen H001 cubins.

No substring-only lineage verdict: each instruction has source, TTGIR locations,
PTX location, register definitions/uses; each SHFL index is evaluated for all
128 threads from actual SASS integer producers. Unresolved proofs stay pending.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re

import numpy as np

from .evidence_resources import pin

REG = re.compile(r'\b(?:UR\d+|R\d+|UP\d+|P\d+|URZ|RZ|UPT|PT)\b')


def parse(path):
    rows = {}
    file, line = None, None
    for text in path.read_text().splitlines():
        loc = re.search(r'//## File "([^"]+)", line (\d+)', text)
        if loc:
            file, line = loc[1], int(loc[2])
        m = re.search(r'/\*([0-9a-f]+)\*/\s*(.*?)\s*;', text)
        if not m:
            continue
        code = m[2]
        pred = None
        if code.startswith('@'):
            pred, code = code.split(None, 1)
        op, operands = (code.split(None, 1)+[''])[:2]
        args = [x.strip() for x in operands.split(',')]
        rows[int(m[1],16)] = dict(pc='0x'+m[1], opcode=op, args=args,
                                  predicate=pred, file=file, line=line,
                                  instruction=m[2])
    return rows


def definitions(row):
    op, args = row['opcode'], row['args']
    if op.startswith(('STG','BRA','EXIT','NOP')):
        return [], 0
    count = 1
    if 'SHFL' in op or 'SETP' in op or op.startswith('PLOP3'):
        count = 2
    elif op.startswith(('LOP3', 'ULOP3')) and args[0].startswith(('P','UP')):
        count = 2
    elif op.startswith(('LEA','ULEA','UIADD3','UIMAD')) and len(args)>1 and args[1].startswith(('P','UP')):
        count = 2
    regs = [x for arg in args[:count] for x in REG.findall(arg) if x not in ('RZ','URZ','PT','UPT')]
    if ('.64' in op or '.WIDE' in op) and regs:
        first = regs[0]
        if re.fullmatch(r'(?:R|UR)\d+', first):
            regs.append(re.sub(r'\d+$',lambda m:str(int(m[0])+1),first))
    if '.128' in op and regs:
        first=regs[0]
        if re.fullmatch(r'(?:R|UR)\d+',first):
            regs += [re.sub(r'\d+$',lambda m:str(int(m[0])+i),first) for i in (1,2,3)]
    return regs, count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--attempt', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', executable=pin(__file__), entries=[],
                  scope='Static source/TTGIR/PTX/SASS proof; independent review and dynamic counters pending')
    resources = json.loads((args.attempt/'resources.json').read_text())
    for entry in resources['resources']:
        name = entry['entry']
        ptx = (args.attempt/(name+'.ptx')).read_text().splitlines()
        # ptxas can merge identical value/source operands. Match their reaching
        # definitions, and compare surviving PTX/SASS index vectors.
        pv = {}
        pidx = {}
        pdefs = {}
        pkeys = {}
        def po(text):
            if text=='%tid.x':
                return np.arange(128,dtype=np.int64)
            if text.startswith('%'):
                if text not in pv:
                    raise ValueError(text)
                return pv[text]
            return np.full(128,int(text,0),dtype=np.int64)
        for ln,text in enumerate(ptx,1):
            m = re.match(r'\s*([a-z][\w.]*)\s+([^;]+);',text)
            if not m:
                continue
            op = m[1]; z=[x.strip() for x in m[2].split(',')]
            try:
                result=None
                if op=='shfl.sync.idx.b32':
                    pidx[ln]=po(z[2]).tolist()
                    pkeys[ln]=(z[1],pdefs.get(z[1]),z[2],pdefs.get(z[2]),z[3],z[4])
                elif op.startswith(('mov.','cvt.u64.u32')):
                    result=po(z[1])
                elif op.startswith('bfe.'):
                    result=(po(z[1])>>po(z[2])) & ((1<<int(z[3],0))-1)
                elif op.startswith(('and.','or.','xor.','add.','mul.','shr.','shl.')):
                    x,y=po(z[1]),po(z[2])
                    result=x&y if op.startswith('and.') else x|y if op.startswith('or.') else x^y if op.startswith('xor.') else x+y if op.startswith('add.') else x*y if op.startswith('mul.') else x>>y if op.startswith('shr.') else x<<y
                elif op.startswith('setp.'):
                    x,y=po(z[1]),po(z[2]);result=(x<y if '.lt.' in op else x==y if '.eq.' in op else x!=y if '.ne.' in op else x>=y).astype(np.int64)
                elif op=='not.pred':
                    result=1-po(z[1])
                elif op.startswith('selp.'):
                    result=np.where(po(z[3]),po(z[1]),po(z[2]))
                if result is not None:
                    pv[z[0]]=result & 0xffffffff
                elif op!='shfl.sync.idx.b32':
                    pv.pop(z[0],None)
            except (ValueError,IndexError,TypeError):
                pv.pop(z[0],None)
            if z[0].startswith('%'):
                pdefs[z[0]]=ln
        ttgir = (args.attempt/(name+'.ttgir')).read_text().splitlines()
        source = Path(entry['source']['path']).read_text().splitlines()
        a = parse(args.attempt/(name+'.sass-source'))
        b = parse(args.attempt/(name+'.sass-ptx'))
        # Map every TTGIR operation to real inlining locations, transitively.
        locations = {}
        for text in ttgir:
            m = re.match(r'(#loc\w*) = (.*)', text)
            if m:
                locations[m[1]] = m[2]
        def source_lines(loc, visited=None):
            visited = set() if visited is None else visited
            if loc in visited:
                return set()
            visited.add(loc)
            text = locations.get(loc, '')
            values = {int(x) for x in re.findall(r'"[^"\n]+":(\d+):\d+', text)}
            for child in re.findall(r'#loc\w*', text):
                values |= source_lines(child, visited)
            return values
        ir_map = {}
        for number, text in enumerate(ttgir, 1):
            if text.startswith('#'):
                continue
            for loc in re.findall(r'#loc\w*',text):
                for line in source_lines(loc):
                    ir_map.setdefault(line,[]).append(number)
        # Reaching producer PCs, including predicate-preserved prior values.
        last = {}
        nodes = {}
        values = {}
        unresolved = []
        ledger = []
        shuffles = []
        def operand(text):
            text = re.sub(r'\.reuse|\.64', '', text)
            invert = text.startswith('!')
            text = text.lstrip('!')
            if text in ('RZ','URZ'):
                value = np.zeros(128,dtype=np.int64)
            elif text in ('PT','UPT'):
                value = np.ones(128,dtype=np.int64)
            elif re.fullmatch(r'(?:R|UR|P|UP)\d+',text):
                value = values.get(text)
                if value is None:
                    raise ValueError('unknown '+text)
            else:
                value = np.full(128,int(text,0),dtype=np.int64)
            return 1-value if invert else value
        for pc, row in a.items():
            d, count = definitions(row)
            inputs=row['args'][count:]
            uses = set(REG.findall(','.join(inputs)))
            pairs=re.findall(r'\b((?:UR|R)\d+)\.64',','.join(inputs))
            if '.64' in row['opcode']:
                pairs += [x for x in uses if re.fullmatch(r'(?:UR|R)\d+',x)]
            if '.WIDE' in row['opcode'] and len(inputs)>=3:
                pairs += REG.findall(inputs[2])
            for reg in pairs:
                if re.fullmatch(r'(?:UR|R)\d+',reg):
                    uses.add(re.sub(r'\d+$',lambda m:str(int(m[0])+1),reg))
            if row['predicate']:
                uses |= set(REG.findall(row['predicate']))
                uses |= set(d)
            edges = {reg: sorted(last.get(reg,[])) for reg in sorted(uses)
                     if reg not in ('RZ','URZ','PT','UPT')}
            node = dict(**row, defines=d, uses=edges,
                        ptx_line=b[pc]['line'], ptx=ptx[b[pc]['line']-1] if b[pc]['line'] else None,
                        source_text=source[row['line']-1] if row['line'] and row['file']==entry['source']['path'] else None,
                        ttgir_lines=ir_map.get(row['line'],[]))
            nodes[pc] = node
            op, ar = row['opcode'], row['args']
            # Exact concrete integer algebra only. Numeric data are deliberately
            # opaque; no emulator result is used as an operator NUM claim.
            try:
                result = None
                args_in = ar[count:]
                v = lambda i: operand(args_in[i])
                if op == 'S2R' and ar[1] == 'SR_TID.X':
                    result = np.arange(128,dtype=np.int64)
                elif op.startswith(('MOV','SEL')):
                    result = v(0) if op.startswith('MOV') else np.where(v(2),v(0),v(1))
                elif op.startswith('IADD') and '.64' not in op:
                    result = v(0)+v(1)
                elif op.startswith('IMAD') and '.WIDE' not in op and '.HI' not in op:
                    result = v(0)*v(1)+v(2)
                elif op.startswith('SGXT.U32'):
                    result = v(0) & ((1 << int(args_in[1],0))-1)
                elif op.startswith('SHF.R'):
                    result = v(2) >> (v(1)&31)
                    if np.any(v(0)):
                        raise ValueError('nonzero low funnel source')
                elif op.startswith('LOP3'):
                    x,y,z,lut = v(0),v(1),v(2),int(args_in[3],0)
                    result = np.zeros(128,dtype=np.int64)
                    for bit in range(8):
                        if lut & (1<<bit):
                            result |= (x if bit&4 else ~x)&(y if bit&2 else ~y)&(z if bit&1 else ~z)
                    result &= 0xffffffff
                    if count==2 and d and d[0].startswith('P'):
                        values[d[0]] = (result!=0).astype(np.int64)
                elif op.startswith('ISETP'):
                    x,y = v(0),v(1)
                    result = (x>=y if '.GE.' in op else x<y if '.LT.' in op else x==y).astype(np.int64)
                    result = (result & v(2)) if '.AND' in op else result | v(2)
                if result is not None and d:
                    target = d[-1] if count==2 and op.startswith('LOP3') else d[0]
                    if row['predicate']:
                        condition = operand(row['predicate'][1:])
                        old = values.get(target)
                        if old is None:
                            raise ValueError('predicated undefined '+target)
                        result = np.where(condition,result,old)
                    values[target] = result & 0xffffffff
                elif d:
                    for reg in d:
                        values.pop(reg,None)
            except (ValueError, IndexError, TypeError) as e:
                for reg in d:
                    values.pop(reg,None)
            if op=='SHFL.IDX':
                item = dict(pc=row['pc'], instruction=row['instruction'],
                            source_line=row['line'], ttgir_lines=node['ttgir_lines'],
                            ptx_line=node['ptx_line'], ptx=node['ptx'],
                            producer_pcs=edges, unpredicated=row['predicate'] is None)
                try:
                    ix = operand(ar[3])
                    item['source_lane_by_thread'] = ix.tolist()
                    item['all_sources_in_0_31'] = bool(np.all((ix>=0)&(ix<32)))
                    item['source_physical_thread_by_destination'] = (np.arange(128)//32*32+ix).tolist()
                    item['same_warp'] = bool(np.all((np.arange(128)//32*32+ix)//32==np.arange(128)//32))
                    item['PTX_SASS_source_index_equal'] = ix.tolist()==pidx.get(node['ptx_line'])
                except ValueError as e:
                    item.update(index_proof='pending',reason=str(e))
                    unresolved.append(item)
                shuffles.append(item)
            # Linear reaching definitions cover the prologue and first K256
            # iteration. Loop-carried edges are added below as explicit unions.
            for reg in d:
                last[reg] = set(last.get(reg,[]))|{pc} if row['predicate'] else {pc}
            ledger.append(node)
        # Add producer edges from loop backedge. Resolve control labels exactly.
        labels = {}
        label = None
        for text in (args.attempt/(name+'.sass-source')).read_text().splitlines():
            if re.match(r'^\.L_x_\w+:$',text):
                label = text[:-1]
            m = re.search(r'/\*([0-9a-f]+)\*/',text)
            if m and label:
                labels[label]=int(m[1],16);label=None
        loops = []
        for pc,n in nodes.items():
            if n['opcode'].startswith('BRA'):
                target = re.search(r'\((\.L_x_\w+)\)',n['instruction'])
                if target and labels[target[1]] < pc:
                    start = labels[target[1]]
                    loop = [x for x in ledger if start<=int(x['pc'],16)<=pc]
                    loop_defs = {}
                    for x in loop:
                        for reg in x['defines']:
                            loop_defs[reg]=int(x['pc'],16)
                    seen=set()
                    for x in loop:
                        for reg in x['uses']:
                            if reg not in seen and reg in loop_defs:
                                x['uses'][reg] = sorted(set(x['uses'][reg])|{loop_defs[reg]})
                        seen.update(x['defines'])
                    loops.append(dict(begin_pc=hex(start),backedge_pc=hex(pc),instruction=n['instruction'],
                                      uniform=n['opcode']=='BRA.U'))
        banned = [n['pc'] for n in ledger if n['opcode'].startswith(('LDS','STS','BAR','LDL','STL','LDSM','LDGSTS'))]
        numeric_bad = [i for i,text in enumerate(ttgir,1) if re.search(r'tensor<4xf(?:16|32|64)',text)
                       or 'threadsPerWarp = [4, 8]' in text or 'warpsPerCTA = [1, 4]' in text]
        numeric_types=set(re.findall(r'tensor<([^<>]*(?:f32|f16|bf16)[^<>]*)>', '\n'.join(ttgir)))
        # ISA data-movement census: all GPR/UR arithmetic remains in the current
        # thread/warp. R2UR is warp-uniform header/scale/address extraction and
        # never transports data between warps. Keep every such PC in the ledger.
        allowed_prefix=('LDC','LDG','STG','S2R','S2UR','CS2R','R2UR','SHFL.IDX','MOV','UMOV',
                        'LOP3','ULOP3','IADD','UIADD3','IMAD','LEA','ULEA','ISETP','UISETP',
                        'SGXT','USGXT','SHF','USHF','SEL','FSEL','FADD','FMUL','UFMUL',
                        'FFMA','HFMA2','HADD2','I2FP','UI2FP','F2FP','FSETP','MUFU',
                        'PRMT','UPRMT','PLOP3','UIMAD','USEL','F2I','BRA','EXIT','NOP')
        unknown_opcodes=sorted(set(n['opcode'] for n in ledger if not n['opcode'].startswith(allowed_prefix)))
        shuffle_ptx = [i+1 for i,text in enumerate(ptx) if 'shfl.sync.idx.b32' in text]
        mapped = [s['ptx_line'] for s in shuffles]
        eliminated=[]
        for ln in shuffle_ptx:
            if ln in mapped:
                continue
            equivalent=[s['pc'] for s in shuffles if pkeys.get(s['ptx_line'])==pkeys.get(ln) and ln in pkeys]
            eliminated.append(dict(ptx_line=ln,ptx=ptx[ln-1],index=pidx.get(ln),
                                   reaching_definition_key=pkeys.get(ln),equivalent_SASS_PCs=equivalent,
                                   proven_same_operands=bool(equivalent)))
        checks = dict(TTGIR_no_numeric_layout_escape=not numeric_bad,
                      TTGIR_all_numeric_tiles_E32=bool(numeric_types) and numeric_types<={'4x32xf32, #blocked','4x32xbf16, #blocked','4x32xf16, #blocked'},
                      all_shuffle_PTX_PC_or_same_operands=len(set(mapped))==len(mapped) and set(mapped)<=set(shuffle_ptx) and all(x['proven_same_operands'] for x in eliminated),
                      all_SASS_shuffle_matches_PTX_idx=not unresolved and all(s['PTX_SASS_source_index_equal'] for s in shuffles),
                      all_shuffle_source_TTGIR_locations=all(s['ttgir_lines'] for s in shuffles),
                      every_SASS_shuffle_unpredicated=all(s['unpredicated'] for s in shuffles),
                      every_SASS_idx_in_range=not unresolved and all(s['all_sources_in_0_31'] for s in shuffles),
                      every_SASS_idx_same_warp=not unresolved and all(s['same_warp'] for s in shuffles),
                      zero_shared_local_barrier_PCs=not banned,
                      SASS_no_unclassified_transfer_opcode=not unknown_opcodes,
                      K_loop_uniform=bool(loops) and all(x['uniform'] for x in loops),
                      no_exit_before_last_shuffle=not any(n['opcode']=='EXIT' and int(n['pc'],16)<max(int(s['pc'],16) for s in shuffles) for n in ledger))
        path = args.out/(name+'.pc-ledger.json')
        path.write_text(json.dumps(dict(entry=name,checks=checks,loops=loops,shuffles=shuffles,
                                       merged_duplicate_operand_shuffles=eliminated,
                                       numeric_types=sorted(numeric_types),unknown_opcodes=unknown_opcodes,
                                       unresolved=unresolved,violating_PCs=banned,
                                       numeric_layout_violating_lines=numeric_bad,instructions=ledger),indent=2)+'\n')
        report['entries'].append(dict(entry=name,checks=checks,ledger=pin(path),
                                     shuffle_PCs=len(shuffles),unresolved=len(unresolved)))
        print(name,len(shuffles),'unresolved',len(unresolved),checks,flush=True)
    report['status'] = 'self-checks-passed-review-pending' if all(all(e['checks'].values()) for e in report['entries']) else 'proof-incomplete-or-violated-H001-static-stop'
    (args.out/'pc-proof.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
