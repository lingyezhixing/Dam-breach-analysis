import numpy as np
import matplotlib.pyplot as plt
import sys

# ==========================================
# 第一部分：辅助函数与限制器
# ==========================================

def minmod_limiter(r):
    """
    Minmod 限制器
    """
    # 对应 MATLAB: psi = max(0, min(1, r))
    return np.maximum(0, np.minimum(1, r))

# ==========================================
# 第二部分：空间算子 (核心修复部分)
# ==========================================

def spatial_operator(U, dx, g):
    """
    计算空间导数项 L(U)
    修复了数组广播错误，严格对齐网格与界面索引。
    """
    # U 的形状是 (2, N)，其中 N = J + 1
    # 物理网格点从 0 到 J
    rows, N = U.shape
    
    h = U[0, :]
    q = U[1, :]
    eps = 1e-12 

    # --- 步骤 1: 计算斜率 (Slope) ---
    # 我们先初始化斜率为全 0 (对应边界处斜率为0)
    sigma_h = np.zeros(N)
    sigma_q = np.zeros(N)
    
    # 仅计算内部网格 (索引 1 到 N-2) 的梯度比率 r
    # 需要用到 (i-1), i, (i+1)
    # Python 切片: [1:-1] 代表 i, [2:] 代表 i+1, [:-2] 代表 i-1
    diff_forward_h = h[2:] - h[1:-1]
    diff_backward_h = h[1:-1] - h[:-2]
    
    # 防止分母为0
    denom_h = diff_backward_h.copy()
    denom_h[np.abs(denom_h) < eps] = eps
    r_h = diff_forward_h / denom_h
    
    diff_forward_q = q[2:] - q[1:-1]
    diff_backward_q = q[1:-1] - q[:-2]
    
    denom_q = diff_backward_q.copy()
    denom_q[np.abs(denom_q) < eps] = eps
    r_q = diff_forward_q / denom_q
    
    # 计算限制器 phi
    phi_h = minmod_limiter(r_h)
    phi_q = minmod_limiter(r_q)
    
    # 填充内部斜率
    sigma_h[1:-1] = phi_h * (h[1:-1] - h[:-2]) / dx
    sigma_q[1:-1] = phi_q * (q[1:-1] - q[:-2]) / dx
    
    # --- 步骤 2: MUSCL 重构界面值 ---
    # 我们有 N 个网格中心，意味着有 N-1 个界面 (索引 0 到 N-2)
    # 界面 i 位于 网格 i 和 网格 i+1 之间
    
    # U_L (界面左侧): 来自网格 i 的右外推
    # U_L = U[i] + 0.5 * sigma[i] * dx
    # 使用切片 [:-1] 对应索引 0 到 N-2
    h_L = h[:-1] + 0.5 * sigma_h[:-1] * dx
    q_L = q[:-1] + 0.5 * sigma_q[:-1] * dx
    
    # U_R (界面右侧): 来自网格 i+1 的左外推
    # U_R = U[i+1] - 0.5 * sigma[i+1] * dx
    # 使用切片 [1:] 对应索引 1 到 N-1 (即网格 i+1)
    h_R = h[1:] - 0.5 * sigma_h[1:] * dx
    q_R = q[1:] - 0.5 * sigma_q[1:] * dx
    
    # 物理修正 (水深非负)
    h_L = np.maximum(h_L, eps)
    h_R = np.maximum(h_R, eps)
    
    u_L = q_L / h_L
    u_R = q_R / h_R
    
    # --- 步骤 3: Rusanov 通量计算 ---
    # 此时 Flux 的长度为 N-1，对应所有内部界面
    
    # 通量函数 f(U)
    f1L = q_L
    f1R = q_R
    
    f2L = (q_L**2 / h_L) + 0.5 * g * h_L**2
    f2R = (q_R**2 / h_R) + 0.5 * g * h_R**2
    
    # 局部波速
    c_L = np.abs(u_L) + np.sqrt(g * h_L)
    c_R = np.abs(u_R) + np.sqrt(g * h_R)
    C = np.maximum(c_L, c_R)
    
    # Rusanov 公式
    Flux1 = 0.5 * (f1L + f1R) - 0.5 * C * (h_R - h_L)
    Flux2 = 0.5 * (f2L + f2R) - 0.5 * C * (q_R - q_L)
    
    # --- 步骤 4: 更新导数 L ---
    # L[i] = -1/dx * (Flux[i] - Flux[i-1])
    # Flux[i] 是网格 i 右边的界面，Flux[i-1] 是网格 i 左边的界面
    
    L = np.zeros_like(U)
    
    # 我们只更新内部网格 1 到 N-2
    # Flux1[1:] 对应右界面 (i=1 到 N-2 的右界面)
    # Flux1[:-1] 对应左界面 (i=1 到 N-2 的左界面)
    # 长度检查: 
    # N=401 -> Flux长度 400. 
    # Flux1[1:] 长度 399. Flux1[:-1] 长度 399.
    # L[:, 1:-1] 长度 399. 
    # 维度完美匹配。
    
    L[0, 1:-1] = -1.0/dx * (Flux1[1:] - Flux1[:-1])
    L[1, 1:-1] = -1.0/dx * (Flux2[1:] - Flux2[:-1])
    
    return L

# ==========================================
# 第三部分：主程序
# ==========================================

def main():
    # 参数设置
    g = 9.81
    Xmax = 5.0
    Tmax = 0.5
    J = 400
    dx = Xmax / J
    
    # 初始化
    x = np.linspace(0, Xmax, J+1)
    U = np.zeros((2, J+1))
    
    # 初始条件：溃坝
    mask_left = x <= 2.5
    U[0, mask_left] = 5.0  # h left
    U[0, ~mask_left] = 2.0 # h right
    U[1, :] = 0.0          # q initial
    
    current_time = 0.0
    step_count = 0
    eps = 1e-8
    
    # 绘图设置
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    
    # 上图：水位
    line_h, = ax1.plot(x, U[0, :], 'b-', linewidth=2)
    ax1.set_ylim(0, 6)
    ax1.set_ylabel('Water Level h (m)')
    ax1.set_title('2D TVD-MUSCL Simulation')
    ax1.grid(True)
    
    # 下图：流速
    line_u, = ax2.plot(x, np.zeros_like(x), 'r-', linewidth=2)
    ax2.set_ylim(-1, 5)
    ax2.set_xlabel('Position x (m)')
    ax2.set_ylabel('Velocity u (m/s)')
    ax2.grid(True)
    
    print("开始计算...")
    
    try:
        while current_time < Tmax:
            # 提取变量
            h = U[0, :]
            q = U[1, :]
            u = np.zeros_like(q)
            mask_wet = h > eps
            u[mask_wet] = q[mask_wet] / h[mask_wet]
            
            # --- CFL 时间步长控制 ---
            # 仅在内部区域计算波速，避免边界干扰
            a_local = np.abs(u[1:-1]) + np.sqrt(g * h[1:-1])
            a_max = np.max(a_local) if a_local.size > 0 else 1.0
            if a_max < eps: a_max = 1.0
            
            Cr = 0.9
            dt = Cr * dx / a_max
            
            # 确保不超出 Tmax
            if current_time + dt > Tmax:
                dt = Tmax - current_time
            
            # 如果 dt 变得极小，强制停止防止死循环
            if dt < 1e-10:
                break

            # --- TVD-RK2 时间步进 ---
            
            # 处理边界 (零梯度)
            U[:, 0] = U[:, 1]
            U[:, -1] = U[:, -2]
            
            # Stage 1: Predictor
            L_n = spatial_operator(U, dx, g)
            U_star = U + dt * L_n
            
            # 修正 U_star
            U_star[0, :] = np.maximum(U_star[0, :], eps)
            U_star[:, 0] = U_star[:, 1]
            U_star[:, -1] = U_star[:, -2]
            
            # Stage 2: Corrector
            L_star = spatial_operator(U_star, dx, g)
            U_new = 0.5 * U + 0.5 * U_star + 0.5 * dt * L_star
            
            # 更新
            U = U_new
            U[0, :] = np.maximum(U[0, :], eps)
            
            current_time += dt
            step_count += 1
            
            # 绘图刷新 (每50步或最后一步)
            if step_count % 50 == 0 or np.abs(current_time - Tmax) < 1e-9:
                line_h.set_ydata(U[0, :])
                
                # 计算实时流速用于绘图
                h_plot = U[0, :]
                q_plot = U[1, :]
                u_plot = np.zeros_like(q_plot)
                mask = h_plot > eps
                u_plot[mask] = q_plot[mask] / h_plot[mask]
                line_u.set_ydata(u_plot)
                
                ax1.set_title(f'Time = {current_time:.4f} s | Step = {step_count}')
                plt.draw()
                plt.pause(0.001)
        
        print(f"计算完成。T = {current_time:.4f}s")
        plt.ioff()
        plt.show()

    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()