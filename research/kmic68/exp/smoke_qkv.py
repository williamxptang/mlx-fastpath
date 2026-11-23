import sys; sys.path.insert(0,'.')
import mlx.core as mx
mx.set_memory_limit(2*2**30)
from qkv_attn import qkv_sdpa, qkv_sdpa2
from smoke_ref import ref
for bits in (4,8):
  for Hq,Hkv,N,qL,nv in [(24,4,1000,1,None),(24,4,4096,4,None),(16,4,768,8,700),(16,8,2048,2,None)]:
    k=mx.random.normal((1,Hkv,N,256)).astype(mx.bfloat16); v=mx.random.normal((1,Hkv,N,256)).astype(mx.bfloat16)
    qk,qv=mx.quantize(k,64,bits),mx.quantize(v,64,bits); kd,vd=mx.dequantize(*qk,64,bits),mx.dequantize(*qv,64,bits)
    q=mx.random.normal((1,Hq,qL,256)).astype(mx.bfloat16); n=nv or N
    r=ref(q,kd[:,:,:n],vd[:,:,:n],1/16); f=qkv_sdpa2(q,qk,qv,1/16,bits,n_valid=nv)
    print(bits,Hq,Hkv,N,qL,nv,'err %.5f'%mx.abs(f.astype(mx.float32)-r).max().item())
