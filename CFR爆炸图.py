import matplotlib.pyplot as plt
import numpy as np
import platform

# === 1. 风格与字体配置 ===
def set_style():
    system_name = platform.system()
    if system_name == "Windows":
        font_name = ['SimHei', 'Microsoft YaHei']
    elif system_name == "Darwin":
        font_name = ['Arial Unicode MS', 'PingFang SC']
    else:
        font_name = ['WenQuanYi Micro Hei']
        
    plt.rcParams.update({
        'font.family': ['sans-serif'],
        'font.sans-serif': font_name + plt.rcParams['font.sans-serif'],
        'axes.unicode_minus': False,
        'font.size': 11,
        'axes.linewidth': 1.5,
        'figure.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
set_style()

# === 2. 求解器核心 (复刻 HTML 逻辑) ===
def solve_cfl_test(J=200, cfl_factor=0.9):
    L = 5.0
    dx = L / J
    g = 9.81
    Tmax = 0.25 # 目标时间
    
    x = np.linspace(0, L, J)
    h = np.ones(J) * 2.0; h[x <= 2.5] = 5.0
    q = np.zeros(J)
    
    def minmod(r): return np.maximum(0, np.minimum(1, r))
    
    t = 0
    
    while t < Tmax:
        # 1. 计算时间步长
        h_safe = np.maximum(h, 1e-8)
        u = q / h_safe
        a = np.abs(u) + np.sqrt(g * h_safe)
        max_a = np.max(a[np.isfinite(a)]) # 忽略 NaN 值
        if max_a < 1e-8: max_a = 1.0
        
        dt_base = dx / max_a
        dt = cfl_factor * dt_base
        
        # 2. TVD-RK2
        def compute_L(h_in, q_in):
            # 内部函数，只用于求解
            sigma_h = np.zeros(J); sigma_q = np.zeros(J)
            dh = h_in[1:] - h_in[:-1]
            dq = q_in[1:] - q_in[:-1]
            r_h = dh[1:] / (dh[:-1] + 1e-12)
            phi_h = minmod(r_h)
            sigma_h[1:-1] = phi_h * dh[:-1] / dx
            r_q = dq[1:] / (dq[:-1] + 1e-12)
            phi_q = minmod(r_q)
            sigma_q[1:-1] = phi_q * dq[:-1] / dx
            
            flux_h = np.zeros(J-1); flux_q = np.zeros(J-1)
            for i in range(J-1):
                hL = h_in[i] + 0.5 * sigma_h[i] * dx
                hR = h_in[i+1] - 0.5 * sigma_h[i+1] * dx
                qL = q_in[i] + 0.5 * sigma_q[i] * dx
                qR = q_in[i+1] - 0.5 * sigma_q[i+1] * dx
                hL = max(hL, 1e-8); hR = max(hR, 1e-8)
                uL = qL/hL; uR = qR/hR
                cL = abs(uL) + np.sqrt(g*hL); cR = abs(uR) + np.sqrt(g*hR)
                C = max(cL, cR)
                F1L = qL; F1R = qR
                F2L = qL**2/hL + 0.5*g*hL**2; F2R = qR**2/hR + 0.5*g*hR**2
                flux_h[i] = 0.5*(F1L+F1R) - 0.5*C*(hR-hL)
                flux_q[i] = 0.5*(F2L+F2R) - 0.5*C*(qR-qL)
            
            Lh = np.zeros(J); Lq = np.zeros(J)
            Lh[1:-1] = -1/dx * (flux_h[1:] - flux_h[:-1])
            Lq[1:-1] = -1/dx * (flux_q[1:] - flux_q[:-1])
            return Lh, Lq

        Lh1, Lq1 = compute_L(h, q)
        h_star = h + dt * Lh1; q_star = q + dt * Lq1
        Lh2, Lq2 = compute_L(h_star, q_star)
        h = 0.5*h + 0.5*h_star + 0.5*dt*Lh2
        q = 0.5*q + 0.5*q_star + 0.5*dt*Lq2
        
        # 边界
        h[0]=h[1]; q[0]=q[1]; h[-1]=h[-2]; q[-1]=q[-2]
        
        t += dt
        
        # 提前退出条件
        if np.any(~np.isfinite(h)) or np.max(np.abs(h)) > 10.0:
            break
            
    return x, h

# === 3. 生成数据 ===
# 稳定组
x_stable, h_stable = solve_cfl_test(J=200, cfl_factor=0.9)

# 爆炸组 (严格使用魔法数字)
magic_factor = 1.136
x_unstable, h_unstable = solve_cfl_test(J=200, cfl_factor=magic_factor)

# === 4. 绘图 ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8), sharex=True)
plt.subplots_adjust(hspace=0.25)

# --- 上图：稳定 ---
ax1.plot(x_stable, h_stable, color='#27ae60', linewidth=2.0)
ax1.fill_between(x_stable, 0, h_stable, color='#27ae60', alpha=0.2)
ax1.set_ylim(0, 6.5)
ax1.set_title("满足 CFL 条件 (Cr $\leq$ 1.0)", fontsize=14, fontweight='bold', color='#27ae60', loc='left')
ax1.text(3.5, 5.5, '计算稳定\n结果光滑', color='#27ae60', fontweight='bold', ha='center',
         bbox=dict(facecolor='white', edgecolor='#27ae60', alpha=0.9, boxstyle='round,pad=0.5'))
ax1.set_ylabel('水位 h (m)', fontweight='bold')
ax1.grid(axis='y', linestyle='--', alpha=0.3)

# --- 下图：爆炸 ---
ax2.plot(x_unstable, h_unstable, color='#c0392b', linewidth=1.5)
ax2.fill_between(x_unstable, -5, h_unstable, color='#c0392b', alpha=0.1)

ax2.set_ylim(-2, 8) 
ax2.set_title(f"违反 CFL 条件 (Cr $\\approx$ {magic_factor:.4f})", fontsize=14, fontweight='bold', color='#c0392b', loc='left')

# 文本框移到右上方
ax2.text(4.0, 6.0, '数值震荡 / 计算发散\nNUMERICAL EXPLOSION', color='#c0392b', fontweight='bold', ha='center', fontsize=12,
         bbox=dict(facecolor='#fff5f5', edgecolor='#c0392b', alpha=1.0, boxstyle='round,pad=0.5'))

ax2.set_ylabel('水位 h (m)', fontweight='bold')
ax2.set_xlabel('位置 x (m)', fontweight='bold')
ax2.grid(axis='y', linestyle='--', alpha=0.3)

plt.savefig('cfl_explosion_vertical_final.png', dpi=300)
plt.show()