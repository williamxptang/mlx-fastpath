import mlx.core as mx, time, statistics
x=[mx.random.normal((64*1024*1024,)).astype(mx.bfloat16) for _ in range(4)]; mx.eval(x)
for name,f in [("sum",lambda a: a.astype(mx.float32).sum()),("copy",lambda a: a*1)]:
    r=[]
    for _ in range(7):
        t=time.perf_counter(); mx.eval([f(a) for a in x]); r.append(time.perf_counter()-t)
    b=sum(a.nbytes for a in x)*(1 if name=="sum" else 2)
    print(name, round(b/statistics.median(r)/1e9,1),"GB/s")
