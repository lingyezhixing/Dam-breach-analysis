import matplotlib.pyplot as plt
import numpy as np
import platform

# === 1. 全局风格与中文字体配置 (与上一张图保持一致) ===
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

# === 2. 简易求解器 (只算一阶) ===
def solve_first_order():
    J = 200; L = 5.0; dx = L / J; g = 9.81; dt = 0.001; Tmax = 0.25
    x = np.linspace(0, L, J)
    h = np.ones(J) * 2.0; h[x <= 2.5] = 5.0
    q = np.zeros(J)
    
    t = 0
    while t < Tmax:
        h_old, q_old = h.copy(), q.copy()
        # 简化的通量计算
        hL, hR = h[:-1], h[1:]
        qL, qR = q[:-1], q[1:]
        uL = np.divide(qL, hL, where=hL>1e-6)
        uR = np.divide(qR, hR, where=hR>1e-6)
        
        a = np.maximum(np.abs(uL)+np.sqrt(g*hL), np.abs(uR)+np.sqrt(g*hR))
        
        # Rusanov Flux
        F1_L, F1_R = qL, qR
        F2_L = qL**2/hL + 0.5*g*hL**2
        F2_R = qR**2/hR + 0.5*g*hR**2
        
        Flux1 = 0.5*(F1_L + F1_R) - 0.5*a*(hR - hL)
        Flux2 = 0.5*(F2_L + F2_R) - 0.5*a*(qR - qL)
        
        h[1:-1] -= dt/dx * (Flux1[1:] - Flux1[:-1])
        q[1:-1] -= dt/dx * (Flux2[1:] - Flux2[:-1])
        t += dt
    return x, h

# === 3. 绘图 (中文版 1:1) ===
x, h = solve_first_order()

# 正方形布局
fig, ax = plt.subplots(figsize=(6, 6))

# 绘制曲线
ax.plot(x, h, color='#2c3e50', linewidth=3, label='一阶 Rusanov 格式\n(t = 0.25s)')
ax.fill_between(x, 0, h, color='#3498db', alpha=0.3)

# === 4. 中文标注 (对应 PPT 文字) ===

# A. 稀疏波
ax.annotate('稀疏波\n(水位平滑过渡)', xy=(1.5, 3.5), xytext=(0.8, 2.0),
            arrowprops=dict(arrowstyle='->', color='#e67e22', lw=2, connectionstyle="arc3,rad=-0.2"),
            color='#e67e22', fontsize=12, fontweight='bold', ha='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#e67e22", alpha=0.9))

# B. 恒定流区
ax.text(2.6, 3.3, '恒定流区\n(流场稳定)', color='#27ae60', 
        fontsize=12, fontweight='bold', ha='center',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))

# C. 激波 (强调耗散)
ax.annotate('激波\n(被数值耗散抹平)', xy=(4.0, 2.8), xytext=(3.5, 5.0),
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=2),
            color='#c0392b', fontsize=12, fontweight='bold', ha='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#c0392b", alpha=0.9))

# === 5. 坐标轴调整 ===
ax.set_xlim(0, 5)
ax.set_ylim(0, 6.5) # 稍微高一点，给上面的文字留空间
ax.set_xlabel('位置 x (m)', fontweight='bold', fontsize=13)
ax.set_ylabel('水位 h (m)', fontweight='bold', fontsize=13)

# 网格线
ax.grid(axis='y', linestyle='--', alpha=0.3)

# 标题 (可选，如果PPT上有标题可以注释掉这一行)
ax.set_title('一阶 Rusanov 格式计算结果', fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('rusanov_structure_cn.png', dpi=300)
plt.show()