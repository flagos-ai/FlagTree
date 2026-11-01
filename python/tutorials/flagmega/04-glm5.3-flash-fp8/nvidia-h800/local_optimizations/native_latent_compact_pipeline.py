"""Keep compact MLA tiles dense and retain complete online64 arithmetic."""
from triton.flagmega.errors import CodegenError


def transform(source):
    replacements = (
        ('_fm_dsa_qk_k = tl.arange(0, 128)', '_fm_dsa_qk_k = tl.arange(0, 256)'),
        ('for _fm_dsa_k_start in tl.range(0, {{ call.head_dim }}, 128):',
         'for _fm_dsa_k_start in tl.range(0, {{ call.head_dim }}, 256):'),
        ('_fm_dsa_pv_k = tl.arange(0, 32)', '_fm_dsa_pv_k = tl.arange(0, 64)'),
        ('for _fm_dsa_pv_start in tl.range(0, 64, 32, num_stages=1):',
         'for _fm_dsa_pv_start in tl.range(0, 64, 64, num_stages=1):'),
        ('_fm_dsa_pv_p = tl.gather(_fm_dsa_rounded_p, tl.broadcast_to(_fm_dsa_pv_token[None, :], (16, 32)), 1)',
         '_fm_dsa_pv_p = _fm_dsa_rounded_p'),
        ('for _fm_dsa_start in tl.range(0, {% if call.index_traversal == "compact" %}_fm_dsa_count{% else %}{{ call.topk_width }}{% endif %}, 64):',
         'for _fm_dsa_start in tl.range(0, {% if call.index_traversal == "compact" %}_fm_dsa_count{% else %}{{ call.topk_width }}{% endif %}, 64, num_stages=2):'),
    )
    for before, after in replacements:
        if source.count(before) != 1:
            raise CodegenError('Native compact MLA fragment source contract changed')
        source = source.replace(before, after)
    # Compact stores exactly count valid tokens, densely in slots [0, count).
    # Each iteration starts below count, so its first slot is always valid.
    # Empty inputs execute zero iterations; the final tile retains its masks.
    first = '                if tl.sum(_fm_dsa_valid.to(tl.int32), 0) > 0:\n'
    last = '                    _fm_dsa_max = _fm_dsa_updated_max\n'
    if source.count(first) != 1 or source.count(last) != 1:
        raise CodegenError('Native compact MLA dense-tile proof requires its declared loop')
    start = source.index(first)
    end = source.index(last, start) + len(last)
    body = source[start + len(first):end]
    body = ''.join(line[4:] if line.startswith('                    ') else line
                   for line in body.splitlines(keepends=True))
    return source[:start] + body + source[end:]
