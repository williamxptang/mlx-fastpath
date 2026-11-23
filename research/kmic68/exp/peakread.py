import mlx.core as mx, time, statistics
src = r"""
  uint i = thread_position_in_grid.x; uint n = threads_per_grid.x;
  uint4 acc = 0;
  for (uint j = i; j < size; j += n) acc ^= a[j];
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345678) out[0] = 1;
"""
k = mx.fast.metal_kernel(name="rd", input_names=["a", "size"], output_names=["out"], source=src.replace("uint4 acc","uint4 acc").replace("a[j]","((const device uint4*)a)[j]"))
x=[mx.random.randint(0,1<<30,(64*1024*1024,),dtype=mx.uint32) for _ in range(4)]; mx.eval(x)
for th in (16384, 65536, 262144):
    r=[]
    for _ in range(7):
        t=time.perf_counter()
        mx.eval([k(inputs=[a, a.size//4], grid=(th,1,1), threadgroup=(256,1,1), output_shapes=[(1,)], output_dtypes=[mx.uint32]) for a in x])
        r.append(time.perf_counter()-t)
    print(th, round(sum(a.nbytes for a in x)/statistics.median(r)/1e9,1), "GB/s read")
