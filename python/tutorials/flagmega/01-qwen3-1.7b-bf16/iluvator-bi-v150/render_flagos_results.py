"""Revalidate paired raw reports and render aggregate latency comparisons."""
import argparse
import json
from pathlib import Path
import statistics

from benchmark_flagos import require_agreement, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--figure-dir', type=Path)
    args = parser.parse_args()
    reports = {variant: [json.loads(p.read_text()) for p in sorted(args.results.glob(f'round*.{variant}.json'))]
               for variant in ('native', 'flagmega')}
    if not reports['native'] or len(reports['native']) != len(reports['flagmega']):
        raise ValueError('Require complete paired native/FlagMega rounds')
    first = reports['native'][0]
    expected = {s['scenario']:s['runs'][0] for s in first['scenarios']}
    rows = {}
    source_hashes = set()
    for variant, values in reports.items():
        for report in values:
            if not report['performance_valid']: raise ValueError('Failed correctness run cannot be plotted')
            for key in ('checkpoint_files', 'compiler_sha256', 'visible_devices', 'decode_tokens',
                        'block_size', 'num_blocks', 'batch', 'cuda_graph', 'prefix_caching',
                        'vllm_source', 'flaggems_source', 'plugin_source'):
                if report[key] != first[key]: raise ValueError(f'Incomparable reports at {key}')
            if variant == 'flagmega':
                if not report['native_token_agreement']: raise ValueError('Missing native token verification')
                source_hashes.add(report['executor_stats'][0]['source_sha256'])
            if {s['scenario'] for s in report['scenarios']} != set(expected):
                raise ValueError('Scenario sets differ')
            for scenario in report['scenarios']:
                for run in scenario['warmup_runs'] + scenario['runs']:
                    require_agreement(run, expected[scenario['scenario']], scenario['scenario'])
                key = (variant, scenario['prompt_length'])
                bucket = rows.setdefault(key, {'decode':[], 'e2e':[], 'ttft':[], 'rounds':[]})
                bucket['rounds'].append({name:scenario[name]['median'] for name in ('decode_ms','e2e_ms','ttft_ms')})
                bucket['decode'].extend(x for r in scenario['runs'] for x in r['decode_step_ms'])
                bucket['e2e'].extend(r['e2e_ms'] for r in scenario['runs'])
                bucket['ttft'].extend(r['ttft_ms'] for r in scenario['runs'])
    if len(source_hashes) != 1: raise ValueError('Different compiled sources cannot share a final label')
    summaries=[]
    for length in sorted(s['prompt_length'] for s in first['scenarios']):
        row={'prompt_length':length}
        for variant in reports:
            bucket=rows[variant,length]
            row[variant]={name:summarize(bucket[name]) for name in ('decode','e2e','ttft')}
            row[variant]['rounds']=bucket['rounds']
        row['decode_speedup']=row['native']['decode']['median']/row['flagmega']['decode']['median']
        row['request_speedup']=row['native']['e2e']['median']/row['flagmega']['e2e']['median']
        summaries.append(row)
    summary={'accepted':True, 'rounds':len(reports['native']), 'source_sha256':next(iter(source_hashes)),
             'native_prefill':True, 'scenarios':summaries}
    (args.results/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    figure_dir=args.figure_dir or args.results
    figure_dir.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none', 'font.family':'DejaVu Sans'})
    for metric,title,ylabel in [('decode','Qwen3-1.7B BF16 / BI-V150: decode','Median latency (ms/token)'),
                                 ('e2e','Native prefill + 64 generated tokens','Median request latency (ms)')]:
        fig,ax=plt.subplots(figsize=(8,4))
        x=list(range(len(summaries)));width=.36
        for offset,variant,label,color in [(-.5,'native','FlagOS vLLM','#6a7b8b'),(.5,'flagmega','FlagMega decode','#197a62')]:
            bars=ax.bar([i+offset*width for i in x],[r[variant][metric]['median'] for r in summaries],width,label=label,color=color)
            ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
        ax.set_xticks(x,[str(r['prompt_length']) for r in summaries]);ax.set_xlabel('Prompt tokens')
        ax.set_ylabel(ylabel);ax.set_title(title);ax.legend(frameon=False)
        ax.spines[['top','right']].set_visible(False);ax.margins(y=.18);fig.tight_layout()
        fig.savefig(figure_dir/f'flagos_{metric}.svg');plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
