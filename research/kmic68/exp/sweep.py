import sys, time, statistics, itertools
sys.path.insert(0,'.')
import mlx.core as mx
from gqa_fold import fold_sdpa
from smoke_ref import ref
def T(fn, qs, ks, vs):
    mx.eval([fn(q,k,v) for q,k,v in zip(qs,ks,vs)]); r=[]
    for _ in range(5):
        outs=[fn(q,k,v) for q,k,v in zip(qs,ks,vs)]  # graph built outside the timer: GPU time only
        t=time.perf_counter(); mx.eval(outs); r.append((time.perf_counter()-t)/len(qs)*1e6)
    return statistics.median(r)
Hq,Hkv,D,L=24,4,256,16
for N,qL in [(1024,1),(8192,1),(32768,1),(4096,4),(4096,8)]:
    ks=[mx.random.normal((1,Hkv,N,D)).astype(mx.bfloat16) for _ in range(L)]; vs=[mx.random.normal((1,Hkv,N,D)).astype(mx.bfloat16) for _ in range(L)]
    qs=[mx.random.normal((1,Hq,qL,D)).astype(mx.bfloat16) for _ in range(L)]; mx.eval(ks,vs,qs)
    r=ref(qs[0],ks[0],vs[0],1/16)
    m=T(lambda q,k,v: mx.fast.scaled_dot_product_attention(q,k,v,scale=1/16,mask='causal' if qL>1 else None),qs,ks,vs)
    res=[]
    for b,n,rg,u in itertools.product((64,128,256,512),(1,2,4,8),(None,),(4,)):
        if b < n*u: continue
        e=mx.abs(fold_sdpa(qs[0],ks[0],vs[0],1/16,block=b,nsg=n,rg=rg,u=u).astype(mx.float32)-r).max().item()
        assert e<5e-3,(b,n,rg,u,e)
        res.append((T(lambda q,k,v: fold_sdpa(q,k,v,1/16,block=b,nsg=n,rg=rg,u=u),qs,ks,vs),b,n,rg,u))
    res.sort(); print(f"N={N} qL={qL} mlx={m:.1f}us best={res[:3]}", flush=True)
    del ks,vs
