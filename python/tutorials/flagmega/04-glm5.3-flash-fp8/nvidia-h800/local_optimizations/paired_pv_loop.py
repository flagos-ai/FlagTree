"""Keep the K64 recurrence while separating complete pairs from the tail."""
from triton.flagmega.errors import CodegenError

SCHEDULE = 'ordered_K64_pairs_and_scalar_tail_v1'
MIN_COUNT = 1024


def transform(consumer):
    bound = '{% if call.index_traversal == "compact" %}_fm_dsa_count{% else %}{{ call.topk_width }}{% endif %}'
    begin = '            for _fm_dsa_start in tl.range(0, ' + bound + ', 64, num_stages=5):\n'
    end = '            _fm_dsa_inverse = '
    if consumer.count(begin) != 1 or consumer.count(end) != 1:
        raise CodegenError('Ordered PV loop boundaries changed')
    start, stop = consumer.index(begin), consumer.index(end)
    original = consumer[start:stop]
    def indent(text):
        return ''.join('    '+line if line.strip() and not line.startswith('{%') else line
                       for line in text.splitlines(keepends=True))
    paired = original.replace(begin, '            for _fm_dsa_start in tl.range(0, _fm_dsa_bulk_end, 64, num_stages=5, loop_unroll_factor=2):\n')
    tail = original.replace(begin, '            for _fm_dsa_start in tl.range(_fm_dsa_bulk_end, _fm_dsa_count, 64, num_stages=1):\n')
    replacement = ('            if _fm_dsa_count >= 1024:\n'
                   '                _fm_dsa_bulk_end = _fm_dsa_count // 128 * 128\n'
                   + indent(paired) + indent(tail) + '            else:\n' + indent(original))
    return consumer[:start] + replacement + consumer[stop:]


def validate(call):
    p = call.get('parameters', {})
    if (p.get('latent_pv_loop_schedule') != SCHEDULE
            or p.get('latent_pv_pair_min_count') != MIN_COUNT
            or p.get('latent_long_column_tile') != 32):
        raise CodegenError('Paired PV requires the complete ordered K64/column32 contract')
