import numpy as np
import time
from numba import jit

# =========================================================
# 选手 A: 原生 Python + NumPy (向量化实现)
# 也就是你队友之前用的优化逻辑，利用数组操作加速
# =========================================================

def native_minmod(r):
    return np.maximum(0, np.minimum(1, r))

def native_spatial_operator(U, dx, g):
    rows, N = U.shape
    h, q = U[0, :], U[1, :]
    eps = 1e-12 

    # 1. 计算斜率
    sigma_h = np.zeros(N)
    sigma_q = np.zeros(N)
    
    diff_fwd_h = h[2:] - h[1:-1]
    diff_bwd_h = h[1:-1] - h[:-2]
    denom_h = diff_bwd_h.copy()
    denom_h[np.abs(denom_h) < eps] = eps
    r_h = diff_fwd_h / denom_h
    
    diff_fwd_q = q[2:] - q[1:-1]
    diff_bwd_q = q[1:-1] - q[:-2]
    denom_q = diff_bwd_q.copy()
    denom_q[np.abs(denom_q) < eps] = eps
    r_q = diff_fwd_q / denom_q
    
    phi_h = native_minmod(r_h)
    phi_q = native_minmod(r_q)
    
    sigma_h[1:-1] = phi_h * diff_bwd_h / dx
    sigma_q[1:-1] = phi_q * diff_bwd_q / dx
    
    # 2. 重构
    h_L = h[:-1] + 0.5 * sigma_h[:-1] * dx
    q_L = q[:-1] + 0.5 * sigma_q[:-1] * dx
    h_R = h[1:] - 0.5 * sigma_h[1:] * dx
    q_R = q[1:] - 0.5 * sigma_q[1:] * dx
    
    h_L = np.maximum(h_L, eps)
    h_R = np.maximum(h_R, eps)
    u_L, u_R = q_L/h_L, q_R/h_R
    
    # 3. Rusanov 通量
    f1L, f1R = q_L, q_R
    f2L = (q_L**2 / h_L) + 0.5 * g * h_L**2
    f2R = (q_R**2 / h_R) + 0.5 * g * h_R**2
    
    c_L = np.abs(u_L) + np.sqrt(g * h_L)
    c_R = np.abs(u_R) + np.sqrt(g * h_R)
    C = np.maximum(c_L, c_R)
    
    Flux1 = 0.5 * (f1L + f1R) - 0.5 * C * (h_R - h_L)
    Flux2 = 0.5 * (f2L + f2R) - 0.5 * C * (q_R - q_L)
    
    # 4. 导数
    L = np.zeros_like(U)
    L[0, 1:-1] = -1.0/dx * (Flux1[1:] - Flux1[:-1])
    L[1, 1:-1] = -1.0/dx * (Flux2[1:] - Flux2[:-1])
    return L

def run_native_benchmark(steps, J):
    # 初始化数据
    dx = 5.0/J
    g = 9.81
    U = np.zeros((2, J+1))
    U[0, :J//2] = 5.0
    U[0, J//2:] = 2.0
    dt = 0.001 # 固定步长仅用于测速
    
    start_time = time.time()
    for _ in range(steps):
        # 模拟 TVD-RK2 的两个阶段
        L1 = native_spatial_operator(U, dx, g)
        U_star = U + dt * L1
        L2 = native_spatial_operator(U_star, dx, g)
        U = 0.5 * U + 0.5 * U_star + 0.5 * dt * L2
        # 简单的边界处理
        U[:, 0] = U[:, 1]
        U[:, -1] = U[:, -2]
        
    end_time = time.time()
    return end_time - start_time

# =========================================================
# 选手 B: Numba JIT (编译级优化)
# 未来的后端核心，将 Python 编译为机器码
# =========================================================

@jit(nopython=True, cache=True)
def numba_compute_step(U, dx, g, dt):
    rows, N = U.shape
    eps = 1e-12
    L = np.zeros((rows, N)) # 显式创建，虽然有点开销，但在JIT中很快
    
    sigma_h = np.zeros(N)
    sigma_q = np.zeros(N)
    
    # 循环融合 (Loop Fusion) 是 Numba 的强项
    for i in range(1, N-1):
        # 斜率计算
        diff_fwd = U[0, i+1] - U[0, i]
        diff_bwd = U[0, i] - U[0, i-1]
        denom = diff_bwd if abs(diff_bwd) > eps else eps
        r = diff_fwd / denom
        phi = max(0.0, min(1.0, r))
        sigma_h[i] = phi * diff_bwd / dx
        
        diff_fwd_q = U[1, i+1] - U[1, i]
        diff_bwd_q = U[1, i] - U[1, i-1]
        denom_q = diff_bwd_q if abs(diff_bwd_q) > eps else eps
        r_q = diff_fwd_q / denom_q
        phi_q = max(0.0, min(1.0, r_q))
        sigma_q[i] = phi_q * diff_bwd_q / dx

    Flux1 = np.zeros(N-1)
    Flux2 = np.zeros(N-1)
    
    for i in range(N-1):
        h_L = U[0, i] + 0.5 * sigma_h[i] * dx
        q_L = U[1, i] + 0.5 * sigma_q[i] * dx
        h_R = U[0, i+1] - 0.5 * sigma_h[i+1] * dx
        q_R = U[1, i+1] - 0.5 * sigma_q[i+1] * dx
        
        h_L = max(h_L, eps)
        h_R = max(h_R, eps)
        u_L = q_L / h_L
        u_R = q_R / h_R
        
        f1L, f1R = q_L, q_R
        f2L = (q_L**2 / h_L) + 0.5 * g * h_L**2
        f2R = (q_R**2 / h_R) + 0.5 * g * h_R**2
        
        c_L = abs(u_L) + np.sqrt(g * h_L)
        c_R = abs(u_R) + np.sqrt(g * h_R)
        C = max(c_L, c_R)
        
        Flux1[i] = 0.5 * (f1L + f1R) - 0.5 * C * (h_R - h_L)
        Flux2[i] = 0.5 * (f2L + f2R) - 0.5 * C * (q_R - q_L)

    for i in range(1, N-1):
        L[0, i] = -1.0/dx * (Flux1[i] - Flux1[i-1])
        L[1, i] = -1.0/dx * (Flux2[i] - Flux2[i-1])
        
    return L

@jit(nopython=True, cache=True)
def numba_integration_loop(U, dx, g, dt, steps):
    # 将整个时间循环也编译进去，减少 Python <-> C 的调用开销
    J = U.shape[1] - 1
    for _ in range(steps):
        # RK2 Step 1
        L1 = numba_compute_step(U, dx, g, dt)
        U_star = U + dt * L1
        U_star[:, 0] = U_star[:, 1]
        U_star[:, -1] = U_star[:, -2]
        
        # RK2 Step 2
        L2 = numba_compute_step(U_star, dx, g, dt)
        U = 0.5 * U + 0.5 * U_star + 0.5 * dt * L2
        U[:, 0] = U[:, 1]
        U[:, -1] = U[:, -2]
    return U

def run_numba_benchmark(steps, J):
    dx = 5.0/J
    g = 9.81
    U = np.zeros((2, J+1))
    U[0, :J//2] = 5.0
    U[0, J//2:] = 2.0
    dt = 0.001

    # 预热 (Warmup): 第一次运行会触发编译，比较慢，不能计入测试
    _ = numba_integration_loop(U.copy(), dx, g, dt, 1)
    
    # 正式测试
    start_time = time.time()
    numba_integration_loop(U, dx, g, dt, steps)
    end_time = time.time()
    
    return end_time - start_time

# =========================================================
# 主程序：竞技场
# =========================================================

def main():
    J = 400         # 网格数量
    STEPS = 500000    # 测试迭代步数
    
    print(f"==================================================")
    print(f"  CFD 性能基准测试: 浅水波方程求解 (J={J})")
    print(f"  迭代步数: {STEPS} 步 (RK2 时间积分)")
    print(f"==================================================\n")

    # --- 测试 Native NumPy ---
    print(f"正在测试 [原生 NumPy] 实现...")
    try:
        time_native = run_native_benchmark(STEPS, J)
        ips_native = STEPS / time_native
        print(f" -> 耗时: {time_native:.4f} s")
        print(f" -> 速度: {ips_native:.1f} iter/s")
    except Exception as e:
        print(f"原生测试出错: {e}")
        time_native = float('inf')

    print("-" * 50)

    # --- 测试 Numba JIT ---
    print(f"正在测试 [Numba JIT] 实现 (含预热)...")
    try:
        time_numba = run_numba_benchmark(STEPS, J)
        ips_numba = STEPS / time_numba
        print(f" -> 耗时: {time_numba:.4f} s")
        print(f" -> 速度: {ips_numba:.1f} iter/s")
    except Exception as e:
        print(f"Numba 测试出错: {e}")
        time_numba = float('inf')

    print("\n==================================================")
    print("  最终结果对比")
    print("==================================================")
    
    if time_numba > 0:
        speedup = ips_numba / ips_native
        print(f"原生 NumPy: {ips_native:10.1f} iter/s")
        print(f"Numba JIT : {ips_numba:10.1f} iter/s")
        print(f"\n性能提升倍数: {speedup:.2f} x  (倍)")
        
        if speedup > 10:
            print("\n结论: Numba 实现了巨大的性能飞跃！这是 C/C++ 级别的效率。")
        elif speedup > 2:
            print("\n结论: 显著的性能提升。")
        else:
            print("\n结论: 差异不明显，可能计算规模太小。")
    else:
        print("测试失败。")

if __name__ == "__main__":
    main()