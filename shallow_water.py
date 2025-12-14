import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
from numba import jit, float64, int64

# ==========================================
# 第一部分：高性能计算内核 (Backend Core)
# 使用 Numba 进行 JIT 编译，达到 C++ 级速度
# ==========================================

@jit(nopython=True, cache=True)
def minmod(r):
    """Minmod 限制器 (JIT 加速版)"""
    return max(0.0, min(1.0, r))

@jit(nopython=True, cache=True)
def compute_step(U, dx, g, dt):
    """
    计算单个时间步的核心逻辑
    包含：斜率计算、MUSCL重构、Rusanov通量、时间积分
    输入 U: (2, J+1)
    """
    # 获取维度
    # 注意：Numba 中尽量减少数组切片创建副本，使用循环通常更快
    # 但为了保持代码逻辑可读性，这里优化了关键路径
    
    rows, N = U.shape
    eps = 1e-12
    
    # ---------------------------
    # 1. 空间算子 L(U) 计算
    # ---------------------------
    
    # 初始化导数项
    L = np.zeros_like(U)
    
    # 预分配数组以避免内存反复申请
    # 内部网格索引范围: 1 到 N-2
    # 界面数量: N-1 (索引 0 到 N-2)
    
    # --- 循环计算内部网格的斜率 ---
    # 这里我们使用显式循环，这在 Numba 中往往比切片更快且更节省内存
    
    sigma_h = np.zeros(N)
    sigma_q = np.zeros(N)
    
    for i in range(1, N-1):
        # h 的斜率
        diff_fwd = U[0, i+1] - U[0, i]
        diff_bwd = U[0, i] - U[0, i-1]
        denom = diff_bwd if abs(diff_bwd) > eps else eps
        r = diff_fwd / denom
        phi = max(0.0, min(1.0, r)) # Minmod 内联
        sigma_h[i] = phi * diff_bwd / dx
        
        # q 的斜率
        diff_fwd_q = U[1, i+1] - U[1, i]
        diff_bwd_q = U[1, i] - U[1, i-1]
        denom_q = diff_bwd_q if abs(diff_bwd_q) > eps else eps
        r_q = diff_fwd_q / denom_q
        phi_q = max(0.0, min(1.0, r_q))
        sigma_q[i] = phi_q * diff_bwd_q / dx

    # --- 计算通量 Flux (循环所有界面 i=0 到 N-2) ---
    # Flux_i 代表界面 i+1/2 的通量
    
    # 临时存储 Flux
    Flux1 = np.zeros(N-1)
    Flux2 = np.zeros(N-1)
    
    for i in range(N-1):
        # 左侧重构 (来自网格 i)
        h_L = U[0, i] + 0.5 * sigma_h[i] * dx
        q_L = U[1, i] + 0.5 * sigma_q[i] * dx
        
        # 右侧重构 (来自网格 i+1)
        h_R = U[0, i+1] - 0.5 * sigma_h[i+1] * dx
        q_R = U[1, i+1] - 0.5 * sigma_q[i+1] * dx
        
        # 物理修正
        h_L = max(h_L, eps)
        h_R = max(h_R, eps)
        
        u_L = q_L / h_L
        u_R = q_R / h_R
        
        # Rusanov 通量
        # F1 = q
        f1L = q_L
        f1R = q_R
        
        # F2 = q^2/h + 0.5*g*h^2
        f2L = (q_L**2 / h_L) + 0.5 * g * h_L**2
        f2R = (q_R**2 / h_R) + 0.5 * g * h_R**2
        
        # 波速
        c_L = abs(u_L) + np.sqrt(g * h_L)
        c_R = abs(u_R) + np.sqrt(g * h_R)
        C = max(c_L, c_R)
        
        Flux1[i] = 0.5 * (f1L + f1R) - 0.5 * C * (h_R - h_L)
        Flux2[i] = 0.5 * (f2L + f2R) - 0.5 * C * (q_R - q_L)

    # --- 更新 L (只更新内部网格 1 到 N-2) ---
    for i in range(1, N-1):
        # Flux[i] 是右界面，Flux[i-1] 是左界面
        L[0, i] = -1.0/dx * (Flux1[i] - Flux1[i-1])
        L[1, i] = -1.0/dx * (Flux2[i] - Flux2[i-1])
        
    return L

@jit(nopython=True, cache=True)
def get_dt(U, dx, g, cfl):
    """JIT加速的 CFL 步长计算"""
    rows, N = U.shape
    max_wave_speed = 0.0
    eps = 1e-8
    
    for i in range(1, N-1):
        h = U[0, i]
        q = U[1, i]
        if h > eps:
            u = q / h
            wave_speed = abs(u) + np.sqrt(g * h)
            if wave_speed > max_wave_speed:
                max_wave_speed = wave_speed
    
    if max_wave_speed < eps:
        max_wave_speed = 1.0
        
    return cfl * dx / max_wave_speed

# ==========================================
# 第二部分：仿真进程 (Producer)
# 职责：只负责算，算完扔进队列
# ==========================================

class SimulationProcess(mp.Process):
    def __init__(self, data_queue, config):
        super().__init__()
        self.queue = data_queue
        self.cfg = config
        self.daemon = True # 主程序退出时自动销毁
        
    def run(self):
        # 解包配置
        g = self.cfg['g']
        dx = self.cfg['dx']
        J = self.cfg['J']
        Tmax = self.cfg['Tmax']
        
        # 初始化数据
        x = np.linspace(0, self.cfg['Xmax'], J+1)
        U = np.zeros((2, J+1))
        
        # 初始条件 (Numpy 操作)
        mask_left = x <= self.cfg['Xmax']/2
        U[0, mask_left] = 5.0
        U[0, ~mask_left] = 2.0
        
        current_time = 0.0
        eps = 1e-8
        
        # 预热 Numba (第一次运行会编译，比较慢，不计入队列)
        print("[Compute] JIT Compiling physics kernel...")
        _ = compute_step(U, dx, g, 0.001)
        _ = get_dt(U, dx, g, 0.9)
        print("[Compute] Compilation done. Simulation started.")
        
        step = 0
        
        while current_time < Tmax:
            # 1. 计算时间步长
            dt = get_dt(U, dx, g, 0.9)
            if current_time + dt > Tmax:
                dt = Tmax - current_time
            if dt < 1e-10: break
                
            # 2. TVD-RK2 第一步 (Predictor)
            # 边界处理 (零梯度)
            U[:, 0] = U[:, 1]
            U[:, -1] = U[:, -2]
            
            L1 = compute_step(U, dx, g, dt)
            U_star = U + dt * L1
            
            # 物理约束
            for i in range(J+1):
                if U_star[0, i] < eps: U_star[0, i] = eps
            
            # 边界处理 U*
            U_star[:, 0] = U_star[:, 1]
            U_star[:, -1] = U_star[:, -2]
            
            # 3. TVD-RK2 第二步 (Corrector)
            L2 = compute_step(U_star, dx, g, dt)
            U_new = 0.5 * U + 0.5 * U_star + 0.5 * dt * L2
            
            # 更新状态
            U = U_new
            # 物理约束
            for i in range(J+1):
                if U[0, i] < eps: U[0, i] = eps
                
            current_time += dt
            step += 1
            
            # 4. 将结果推入队列
            # 发送 (时间, 水位, 流速) 的副本
            # 注意：为了性能，不要发送整个 U，只发用于显示的副本
            if self.queue.full():
                # 如果队列满了，计算进程会在这里阻塞
                # 这保证了显示进程能跟上，实现了"精确显示每一步"
                # 如果不需要精确显示，可以用 self.queue.put_nowait 并在异常时跳过
                pass
            
            # 计算流速用于传输
            h_send = U[0, :].copy()
            q_send = U[1, :].copy()
            u_send = np.zeros_like(h_send)
            mask = h_send > eps
            u_send[mask] = q_send[mask] / h_send[mask]
            
            self.queue.put((current_time, h_send, u_send, step))
        
        # 发送结束信号
        self.queue.put(None)
        print("[Compute] Simulation finished.")

# ==========================================
# 第三部分：显示进程 (Consumer)
# 职责：只负责画，不进行任何物理计算
# ==========================================

class VisualizationProcess(mp.Process):
    def __init__(self, data_queue, config):
        super().__init__()
        self.queue = data_queue
        self.cfg = config
        
    def run(self):
        # 初始化绘图窗口
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        x = np.linspace(0, self.cfg['Xmax'], self.cfg['J']+1)
        
        # 初始化线条
        line_h, = ax1.plot(x, np.zeros_like(x), 'b-', linewidth=2)
        ax1.set_ylim(0, 6)
        ax1.set_ylabel('Water Level (m)')
        ax1.set_title('Real-time Computation Stream')
        ax1.grid(True)
        
        line_u, = ax2.plot(x, np.zeros_like(x), 'r-', linewidth=2)
        ax2.set_ylim(-2, 6)
        ax2.set_ylabel('Velocity (m/s)')
        ax2.set_xlabel('Position (m)')
        ax2.grid(True)
        
        print("[Visual] Window ready. Waiting for data...")
        
        frame_count = 0
        last_render_time = time.time()
        
        while True:
            # 从队列获取数据
            try:
                # 阻塞式获取，等待计算结果
                item = self.queue.get()
            except Exception:
                break
                
            if item is None:
                print("[Visual] Received stop signal.")
                break
                
            # 解包数据
            sim_time, h, u, step = item
            
            # 渲染控制：限制FPS，避免绘图消耗过多资源导致队列堆积
            # 如果你想要"精确每一帧"，可以注释掉这个 if
            # 但实际上 matplotlib 绘图很慢，每秒只能画几十帧
            # current_wall_time = time.time()
            # if current_wall_time - last_render_time < 0.016: # 限制约 60 FPS
            #     continue
            
            # 更新图表
            line_h.set_ydata(h)
            line_u.set_ydata(u)
            ax1.set_title(f'Time: {sim_time:.4f}s | Step: {step}')
            
            # 极简重绘 (比 plt.pause 更快)
            fig.canvas.draw_idle()
            fig.canvas.start_event_loop(0.001)
            
            last_render_time = time.time()
            frame_count += 1
            
        print("[Visual] Closing window.")
        plt.close(fig)

# ==========================================
# 第四部分：主控制器
# ==========================================

if __name__ == "__main__":
    # 配置参数
    config = {
        'g': 9.81,
        'Xmax': 5.0,
        'Tmax': 0.25,
        'J': 400,
        'dx': 5.0/400
    }
    
    # 创建跨进程队列
    # maxsize 很关键：
    # 1. 如果你设得很小 (e.g., 10)，计算进程会因为队列满了而暂停，
    #    等待显示进程画完。这实现了【完全同步的逐帧显示】，但计算速度会被拖慢到绘图速度。
    # 2. 如果你设得很大 (e.g., 10000)，计算进程会飞快地跑，
    #    显示进程在后面慢慢追。
    queue = mp.Queue(maxsize=10000) 
    
    # 实例化进程
    sim_process = SimulationProcess(queue, config)
    vis_process = VisualizationProcess(queue, config)
    
    # 启动
    print("Main: Starting processes...")
    sim_process.start()
    vis_process.start()
    
    # 等待结束
    sim_process.join()
    vis_process.join()
    print("Main: All done.")