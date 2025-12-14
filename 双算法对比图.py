import matplotlib.pyplot as plt
import numpy as np
import platform

# === 1. 全局风格与中文字体配置 ===
def set_style():
    # 自动检测系统并设置中文字体
    system_name = platform.system()
    if system_name == "Windows":
        font_name = ['SimHei', 'Microsoft YaHei'] # 优先用黑体/微软雅黑
    elif system_name == "Darwin": # Mac
        font_name = ['Arial Unicode MS', 'PingFang SC'] 
    else: # Linux
        font_name = ['WenQuanYi Micro Hei']
        
    plt.rcParams.update({
        'font.family': ['sans-serif'],
        'font.sans-serif': font_name + plt.rcParams['font.sans-serif'], # 将中文字体加到首选列表
        'axes.unicode_minus': False,        # 解决负号显示为方块的问题
        'font.size': 12,
        'axes.linewidth': 1.5,
        'lines.linewidth': 2.5,
        'figure.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })
set_style()

# === 2. 简易求解器 (保持不变) ===
def solve_dam_break(J=200, order=1):
    L = 5.0; dx = L / J; g = 9.81; dt = 0.001; Tmax = 0.25
    x = np.linspace(0, L, J)
    h = np.ones(J) * 2.0; h[x <= 2.5] = 5.0
    q = np.zeros(J)
    def minmod(r): return np.maximum(0, np.minimum(1, r))
    
    t = 0
    while t < Tmax:
        h_old, q_old = h.copy(), q.copy()
        sigma_h, sigma_q = np.zeros(J), np.zeros(J)
        if order == 2:
            dh = h[1:] - h[:-1]; dq = q[1:] - q[:-1]
            for i in range(1, J-1):
                if abs(dh[i-1]) > 1e-6: sigma_h[i] = minmod(dh[i]/dh[i-1]) * dh[i-1]/dx
                if abs(dq[i-1]) > 1e-6: sigma_q[i] = minmod(dq[i]/dq[i-1]) * dq[i-1]/dx
        flux_h = np.zeros(J-1); flux_q = np.zeros(J-1)
        for i in range(J-1):
            hL = h[i] + 0.5 * sigma_h[i] * dx; hR = h[i+1] - 0.5 * sigma_h[i+1] * dx
            qL = q[i] + 0.5 * sigma_q[i] * dx; qR = q[i+1] - 0.5 * sigma_q[i+1] * dx
            uL = qL/hL if hL>1e-6 else 0; uR = qR/hR if hR>1e-6 else 0
            a = max(abs(uL)+np.sqrt(g*hL), abs(uR)+np.sqrt(g*hR))
            flux_h[i] = 0.5*(qL + qR) - 0.5*a*(hR - hL)
            flux_q[i] = 0.5*((qL**2/hL + 0.5*g*hL**2) + (qR**2/hR + 0.5*g*hR**2)) - 0.5*a*(qR - qL)
        h[1:-1] -= dt/dx * (flux_h[1:] - flux_h[:-1])
        q[1:-1] -= dt/dx * (flux_q[1:] - flux_q[:-1])
        t += dt
    return x, h

# === 3. 生成数据 ===
x, h1 = solve_dam_break(J=200, order=1)
x, h2 = solve_dam_break(J=200, order=2)

# === 4. 绘图 (中文版) ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8), sharex=True, sharey=True)
plt.subplots_adjust(hspace=0.15) 

# --- 定义关键点 (中文标签) ---
key_points = [
    (0.8, '#95a5a6', '稀疏波前沿', '--'),   
    (2.5, '#bdc3c7', '初始坝址', '-.'),      
    (4.05, '#e74c3c', '激波锋面', '--')      
]

def draw_ref_lines(ax, show_text=False):
    for pos, color, label, style in key_points:
        ax.axvline(x=pos, color=color, linestyle=style, linewidth=1.5, alpha=0.8, zorder=0)
        if show_text:
            ax.text(pos, 5.5, label, color=color, fontsize=10, ha='center', fontweight='bold', backgroundcolor='white')

# --- 上图：一阶 ---
draw_ref_lines(ax1, show_text=True)
ax1.plot(x, h1, color='#7f8c8d', linewidth=3)
ax1.fill_between(x, 0, h1, color='#95a5a6', alpha=0.2)

# 标题与注释
ax1.set_title("方法一：一阶 Rusanov 格式", fontsize=14, fontweight='bold', loc='left', pad=20)
ax1.annotate('数值耗散严重\n(滞后/抹平)', xy=(4.05, 3.0), xytext=(2.2, 4.0),
             arrowprops=dict(facecolor='#c0392b', shrink=0.05, width=2),
             fontsize=11, color='#c0392b', fontweight='bold')

ax1.set_ylabel('水位 h (m)', fontweight='bold', fontsize=12)
ax1.set_ylim(0, 6.5)
ax1.grid(axis='y', linestyle=':', alpha=0.3)

# --- 下图：二阶 ---
draw_ref_lines(ax2, show_text=False)
ax2.plot(x, h2, color='#2980b9', linewidth=3)
ax2.fill_between(x, 0, h2, color='#2980b9', alpha=0.3)

# 标题与注释
ax2.set_title("方法二：二阶 TVD-MUSCL 格式", fontsize=14, fontweight='bold', loc='left', pad=10)
ax2.annotate('激波锋面锐利\n(高分辨率)', xy=(4.05, 2.7), xytext=(2.2, 3.8),
             arrowprops=dict(facecolor='#27ae60', shrink=0.05, width=2),
             fontsize=11, color='#27ae60', fontweight='bold')

ax2.set_ylabel('水位 h (m)', fontweight='bold', fontsize=12)
ax2.set_xlabel('位置 x (m)', fontweight='bold', fontsize=12)
ax2.set_xlim(0, 5)
ax2.grid(axis='y', linestyle=':', alpha=0.3)

plt.savefig('vertical_comparison_cn.png', dpi=300)
plt.show()