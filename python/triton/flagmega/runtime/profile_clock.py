"""A22 bounded integer checksum oracle and predeclared clock fit; no device jobs."""
from __future__ import annotations
import math
MASK=(1<<32)-1
LADDER=[1<<k for k in range(12,27)]
def checksum(iterations,seed):
    if type(iterations) is not int or not 0<=iterations<=1<<26 or type(seed) is not int or not 0<=seed<=MASK:raise ValueError('Bounded U32 controls required.')
    a,c=1664525,1013904223;pa,pc=1,0;n=iterations
    while n:
        if n&1:pa,pc=(a*pa)&MASK,(a*pc+c)&MASK
        a,c=(a*a)&MASK,(a*c+c)&MASK;n>>=1
    return (pa*seed+pc)&MASK

def fit_clock(samples,*,data_origin):
    if data_origin not in ('actual_execution','synthetic_fixture'):raise ValueError('Honest calibration origin required.')
    points=[];lengths={}
    for s in samples:
        if s['iterations'] not in LADDER or s.get('event_status')!=0 or s.get('guard_pass') is not True:raise ValueError('Invalid attempted length/event/guard.')
        start,end=int(s['start_tick']),int(s['end_tick'])
        if str(start)!=s['start_tick'] or str(end)!=s['end_tick'] or not 0<=start<end<(1<<64) or int(s['checksum'])!=checksum(s['iterations'],s['seed']):raise ValueError('Invalid counter/checksum.')
        y=s['event_elapsed_ms']*1e6;x=end-start
        if not math.isfinite(y) or y<=0:raise ValueError('Invalid event interval.')
        points.append((x,y,s['iterations']))
        if y>=1e6:lengths[s['iterations']]=lengths.get(s['iterations'],0)+1
    if len(lengths)<3 or min(lengths.values())<3 or not any(y<1e6 for x,y,n in points):raise ValueError('Need short control and3long lengths x3 repetitions.')
    xm=sum(x for x,y,n in points)/len(points);ym=sum(y for x,y,n in points)/len(points)
    denom=sum((x-xm)**2 for x,y,n in points)
    if denom<=0:raise ValueError('No independent duration range.')
    beta=sum((x-xm)*(y-ym) for x,y,n in points)/denom;intercept=ym-beta*xm
    residuals=[abs(y-(intercept+beta*x)) for x,y,n in points]
    criteria=.98<=beta<=1.02 and -2000<=intercept<=100000 and all(r<=max(2000,.01*y) for r,(x,y,n) in zip(residuals,points) if y>=1e6)
    return {'data_origin':data_origin,'criteria_satisfied':criteria,'beta_ns_per_tick':beta,'intercept_ns':intercept,'residuals_ns':residuals,
            'ns_per_tick':1.0 if criteria and data_origin=='actual_execution' else None,'receipt_issued':False,'attempted_samples':samples}
