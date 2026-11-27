"""Bracket device timestamps by host CLOCK_MONOTONIC and retain affine-fit uncertainty."""
import math
import numpy as np
import triton
import triton.language as tl
from reference_identity import file_pin, save_json
from timing_isolation import monotonic_ns, clock_identity


@triton.jit
def _stamp(pointer):
    now = tl.inline_asm_elementwise('mov.u64 $0, %globaltimer;', constraints='=l',
                                    args=[], dtype=tl.uint64, is_pure=False, pack=1)
    tl.store(pointer, now)


class ClockCorrelation:
    def __init__(self, context):
        self.c = context
        self.pointer = context.torch.zeros((), dtype=context.torch.uint64, device=context.device)
        _stamp[(1,)](self.pointer, num_warps=1)
        context.stream.synchronize()
        self.anchors = []

    def bracket(self, phase, repeats=12):
        for repeat in range(repeats):
            self.c.stream.synchronize()
            start = monotonic_ns()
            _stamp[(1,)](self.pointer, num_warps=1)
            self.c.stream.synchronize()
            stop = monotonic_ns()
            stamp = int(self.pointer.item())
            self.anchors.append(dict(phase=phase, repeat=repeat, host_low_ns=start,
                                     host_high_ns=stop, device_ns=stamp))

    def save(self, output, spans, window):
        """Intersect all bracket constraints; project every endpoint over feasible maps.

        Device and host origins are subtracted as integers before float arithmetic.
        The map is affine over this capture; errors are bracket bounds conditional
        on that model, not a confidence interval or a retroactive historical mapping.
        """
        origin_d = self.anchors[0]['device_ns']
        origin_h = self.anchors[0]['host_low_ns']
        save_json(output/'host-device-anchors.json', dict(host_clock=clock_identity(),
            anchors=self.anchors, device_origin_ns=origin_d, host_origin_ns=origin_h,
            mechanism='Host start -> GPU globaltimer store -> stream synchronize -> host stop; D2H of anchor after stop'))
        anchors = [(r['device_ns']-origin_d, r['host_low_ns']-origin_h,
                    r['host_high_ns']-origin_h) for r in self.anchors]
        # Conservative initial scale bounds agree with the separately measured
        # CUDA-event device-scale control. Actual brackets tighten the range.
        bmax = max(abs(low)+abs(x)*1.02+abs(high) for x, low, high in anchors)+1
        polygon = [(0.98, -bmax), (1.02, -bmax), (1.02, bmax), (0.98, bmax)]
        def clip(points, ca, cb, limit):
            result = []
            for previous, current in zip(points, points[1:]+points[:1]):
                v0 = ca*previous[0]+cb*previous[1]-limit
                v1 = ca*current[0]+cb*current[1]-limit
                if (v0 <= 0) != (v1 <= 0):
                    fraction = v0/(v0-v1)
                    result.append(tuple(p+fraction*(q-p) for p, q in zip(previous, current)))
                if v1 <= 0:
                    result.append(current)
            return result
        for x, low, high in anchors:
            polygon = clip(polygon, x, 1, high)
            polygon = clip(polygon, -x, -1, -low)
            if not polygon:
                raise RuntimeError('Host/device brackets admit no affine map; preserve raw and leave correlation pending')
        vertices = np.array(polygon)
        endpoint_d = np.array([[s['start_ns']-origin_d, s['end_ns']-origin_d] for s in spans], dtype=np.int64)
        assert min(r['device_ns'] for r in self.anchors) <= window[0]
        assert max(r['device_ns'] for r in self.anchors) >= window[1]
        projected = endpoint_d[..., None]*vertices[:, 0]+vertices[:, 1]
        low = np.floor(projected.min(axis=-1)).astype(np.int64)+origin_h
        high = np.ceil(projected.max(axis=-1)).astype(np.int64)+origin_h
        midpoint = low+(high-low)//2
        raw = np.stack([endpoint_d+origin_d, low, high, midpoint], axis=-1)
        np.save(output/'host-device-points.npy', raw)
        points = [dict(span_id=s['id'], device_start_ns=s['start_ns'], device_end_ns=s['end_ns'],
                       host_start_bounds_ns=list(map(int, low_high)), host_end_bounds_ns=list(map(int, end_high)),
                       host_start_midpoint_ns=int(midpoint[i,0]), host_end_midpoint_ns=int(midpoint[i,1]))
                  for i, (s, low_high, end_high) in enumerate(zip(spans, zip(low[:,0],high[:,0]), zip(low[:,1],high[:,1])))]
        save_json(output/'host-device-points.json', points)
        report = dict(schema='flagmega.host-device-correlation/v1', status='measured-unreviewed',
            host_clock=clock_identity(), device_clock='PTX globaltimer, same execution',
            model='Affine over bracketed capture; feasible polygon from all host execution brackets',
            assumptions='Affine clocks over this execution and separately CUDA-event-validated nominal ns scale; bounds conditional on this model',
            anchor_count=len(anchors), span_count=len(spans), endpoint_count=2*len(spans),
            slope_bounds=[float(vertices[:,0].min()),float(vertices[:,0].max())],
            affine_vertices_slope_offset_ns=polygon, device_origin_ns=origin_d, host_origin_ns=origin_h,
            max_endpoint_bound_width_ns=int((high-low).max()),
            point_layout='int64[span,start_or_end,device_ns/host_low_ns/host_high_ns/host_midpoint_ns]',
            anchors=file_pin(output/'host-device-anchors.json'), points=file_pin(output/'host-device-points.npy'),
            point_ids=file_pin(output/'host-device-points.json'), harness=file_pin(output/'harness/clock_correlation.py'),
            historical_trace='Previous capture remains device-only; this mapping belongs solely to this new capture')
        save_json(output/'host-device-correlation.json', report)
        return report
