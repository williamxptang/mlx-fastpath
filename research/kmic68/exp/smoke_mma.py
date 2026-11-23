import sys; sys.path.insert(0,'.')
import mlx.core as mx
from gqa_mma import mma_sdpa
from smoke_ref import ref
mx.random.seed(1)
for Hq,Hkv,N,qL,nv in [(24,4,1024,1,None),(24,4,4096,4,None),(16,4,800,8,None),(16,8,4096,16,None),(24,4,2048,3,1500),(16,4,512,1,77)]:
    q=mx.random.normal((1,Hq,qL,256)).astype(mx.bfloat16); k=mx.random.normal((1,Hkv,N,256)).astype(mx.bfloat16); v=mx.random.normal((1,Hkv,N,256)).astype(mx.bfloat16)
    n=nv or N
    r=ref(q,k[:,:,:n],v[:,:,:n],1/16); f=mma_sdpa(q,k,v,1/16,n_valid=nv)
    m=mx.fast.scaled_dot_product_attention(q,k[:,:,:n],v[:,:,:n],scale=1/16,mask='causal' if qL>1 else None)
    print(Hq,Hkv,N,qL,nv,'mma err %.5f'%mx.abs(f.astype(mx.float32)-r).max().item(),'mlx err %.5f'%mx.abs(m.astype(mx.float32)-r).max().item())
