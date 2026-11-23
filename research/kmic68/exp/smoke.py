import mlx.core as mx, sys
sys.path.insert(0,'.')
from gqa_fold import fold_sdpa
exec(open('bench_fold.py').read().split('def mlx_sdpa')[0].split('dt = mx.bfloat16')[1])
for Hq,Hkv,N,qL in [(24,4,1000,1),(24,4,4096,4),(16,4,777,8),(16,8,5000,16)]:
    q=mx.random.normal((1,Hq,qL,256)).astype(mx.bfloat16); k=mx.random.normal((1,Hkv,N,256)).astype(mx.bfloat16); v=mx.random.normal((1,Hkv,N,256)).astype(mx.bfloat16)
    r=ref(q,k,v,1/16); f=fold_sdpa(q,k,v,1/16)
    m=mx.fast.scaled_dot_product_attention(q,k,v,scale=1/16,mask='causal' if qL>1 else None)
    print(Hq,Hkv,N,qL,'fold err',mx.abs(f.astype(mx.float32)-r).max().item(),'mlx err',mx.abs(m.astype(mx.float32)-r).max().item())
