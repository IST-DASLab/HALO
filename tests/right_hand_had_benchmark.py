from hadamard.hadamard_utils import matmul_hadU_cuda, matmul_hadUt, get_hadK, matmul_hadU
import torch, time


num_warmup_steps = 5
num_bench_steps = 10
bsz = 4
matrix_sizes = [4096, 11008, 8192, 14336]

def acc_benchmark(x):
    H = matmul_hadU(torch.eye(x.shape[-2]).cuda()).to(x.dtype)
    y_baseline = H @ x 
    hadK, K = get_hadK(x.shape[-2], transpose=True, device=x.device, dtype=x.dtype)
    y_fast_had = matmul_hadU_cuda(x.transpose(-1, -2).contiguous(),
                                  hadK, K).transpose(-1, -2).contiguous()
    # check if the above valuse are close
    print(f"Matrix size: {matrix_size}: ", end="")
    print(f"Error: {torch.max(torch.abs(y_baseline - y_fast_had))}!", end=" ")
    if torch.max(torch.abs(y_baseline - y_fast_had)) > 1e-3:
        print("Test failed")
    else:
        print("Test passed")
    
def runtime_benchmark(x):
    H = matmul_hadU(torch.eye(x.shape[-2]).cuda()).to(x.dtype)
    hadK, K = get_hadK(x.shape[-2], transpose=True, device=x.device, dtype=x.dtype)
    torch.cuda.synchronize()
    for i in range(num_warmup_steps):
        y_baseline = H @ x
        torch.cuda.synchronize()
        y_fast_had = matmul_hadU_cuda(x.transpose(-1, -2).contiguous(),
                                  hadK, K).transpose(-1, -2).contiguous()
        torch.cuda.synchronize()
        
    start_time = time.perf_counter()
    for i in range(num_bench_steps):
        y_baseline = H @ x
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    print('------------')
    print(f"Matrix size: {x.shape}")
    print(f"Baseline time: {(end_time - start_time)*1000/num_bench_steps} ms")
    baseline_time = (end_time - start_time)*1000/num_bench_steps
    
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for i in range(num_bench_steps):
        y_fast_had = matmul_hadU_cuda(x.transpose(-1, -2).contiguous(),
                                  hadK, K).transpose(-1, -2).contiguous()
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    fast_had_time = (end_time - start_time)*1000/num_bench_steps
    print(f"Fast Hadamard time: {(end_time - start_time)*1000/num_bench_steps} ms")
    
    # speedup up to 3digits
    print(f"Speedup: {baseline_time/fast_had_time:.3f}x")
    print('------------')

if __name__ == '__main__':
    
    
    for matrix_size in matrix_sizes:
        x = torch.rand(bsz, matrix_size, matrix_size, dtype=torch.float32).cuda()
        acc_benchmark(x)
        
    for matrix_size in matrix_sizes:
        x = torch.rand(bsz, matrix_size, matrix_size, dtype=torch.float32).cuda()
        runtime_benchmark(x)
        
        