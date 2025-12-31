
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import os

print("开始冻土分析...")

# ==================== 第一部分：生成不同阈值的冻土分布文件 ====================
print("\n第一部分：生成冻土分布文件...")

FillValue = -9999
f1 = 0
FI_m = np.full((17, 600, 1440), -9999, dtype=np.float32)
TI_m = np.full((17, 600, 1440), -9999, dtype=np.float32)

# 读取2000-2016年的数据
for f in range(2000, 2017):
    print(f"  读取年份 {f}...")
    
    # 读取FI数据
    a = xr.open_dataset(f"FI_ensemble_mean_{f}.nc")
    FI = a['FI'].values
    
    # 读取TI数据
    b = xr.open_dataset(f"TI_ensemble_mean_{f}.nc")
    TI = b['TI'].values
    
    FI_m[f1, :, :] = FI
    TI_m[f1, :, :] = TI
    
    if f1 == 0:
        lat = a['lat'].values
        lon = a['lon'].values
    
    a.close()
    b.close()
    f1 += 1

nlat = len(lat)
nlon = len(lon)

# 计算平均值
print("  计算多年平均...")
FI_m1 = np.nanmean(FI_m, axis=0)
TI_m1 = np.nanmean(TI_m, axis=0)

# 读取地形掩膜
print("  读取地形掩膜...")
mask1 = xr.open_dataset("GTOPOP_v4.nc")
mask2 = mask1['elevation'].values
mask1.close()

# 生成不同阈值的冻土分类
print("  生成不同阈值的冻土分类文件...")
N0 = 0.01
N = N0

for n in range(100):
    if n % 10 == 0:
        print(f"    处理阈值 {n+1}/100...")
    
    # 计算冻土指数
    frozen = np.sqrt(FI_m1) / (np.sqrt(FI_m1) + np.sqrt(TI_m1))
    
    # 分类
    frozen_classified = np.full_like(frozen, -9999, dtype=np.float32)
    frozen_classified[(frozen >= N) & (frozen <= 1)] = 3
    frozen_classified[frozen < N] = 0
    
    # 应用地形掩膜
    frozen_classified = np.where(mask2 == 1, frozen_classified, -9999)
    
    # 保存
    ds_out = xr.Dataset(
        {'frozen': (['lat', 'lon'], frozen_classified)},
        coords={'lat': (['lat'], lat), 'lon': (['lon'], lon)}
    )
    ds_out.to_netcdf(f"Frozen_ground_N={N:.2f}.nc")
    ds_out.close()
    
    N += 0.01


# ==================== 函数：计算Kappa系数 ====================
def closest_val(target, array):
    """找到最接近目标值的索引"""
    idx = np.abs(array - target).argmin()
    return idx


def calculate_kappa(frozen_files, obser_file, n_range, n_step, categories):
    """
    计算Kappa系数
    
    Parameters:
    -----------
    frozen_files : list of str
        预测的冻土文件路径列表
    obser_file : str
        观测数据文件路径
    n_range : int
        阈值数量
    n_step : float
        阈值步长
    categories : list
        分类类别列表
    """
    K = np.full(n_range, 9999, dtype=np.float32)
    POD = np.full(n_range, 9999, dtype=np.float32)
    N11 = np.full(n_range, 9999, dtype=np.float32)
    
    N1 = 0.01
    
    # 读取观测数据
    obser = xr.open_dataset(obser_file)
    obser_frozen = obser['frozen'].values
    obser_lat = obser['lat'].values
    obser_lon = obser['lon'].values
    obser.close()
    
    for ij in range(n_range):
        if ij % 10 == 0:
            print(f"    计算阈值 {ij+1}/{n_range}...")
        
        # 初始化计数器
        counts_pred = {cat: 0 for cat in categories}
        counts_obs = {cat: 0 for cat in categories}
        a = 0  # 匹配数量
        N_count = 0  # 总样本数
        
        # 读取预测数据
        in_file = xr.open_dataset(f"Frozen_ground_N={N1:.2f}.nc")
        frozen = in_file['frozen'].values
        lat_temp = in_file['lat'].values
        lon_temp = in_file['lon'].values
        in_file.close()
        
        nlat = len(lat_temp)
        nlon = len(lon_temp)
        
        # 遍历所有网格点
        for i in range(nlat):
            for j in range(nlon):
                pre = frozen[i, j]
                
                if pre != -9999:  # 非缺失值
                    longitude = lon_temp[j]
                    latitude = lat_temp[i]
                    
                    # 找到观测数据中最近的点
                    index_lat = closest_val(latitude, obser_lat)
                    index_lon = closest_val(longitude, obser_lon)
                    
                    rain = obser_frozen[index_lat, index_lon]
                    
                    if rain != -9999:  # 观测值非缺失
                        oo = pre
                        mm = rain
                        
                        # 统计匹配
                        if mm == oo:
                            a += 1
                        
                        # 统计各类别数量
                        for cat in categories:
                            if oo == cat:
                                counts_pred[cat] += 1
                            if mm == cat:
                                counts_obs[cat] += 1
                        
                        N_count += 1
        
        # 计算Kappa系数
        if N_count > 0:
            a_1 = float(a)
            N_1 = float(N_count)
            
            # 计算p0（总体准确率）
            p0 = a_1 / N_1
            
            # 计算pe（随机一致性概率）
            pe = 0
            for cat in categories:
                a_cat = float(counts_pred[cat])
                b_cat = float(counts_obs[cat])
                pe += (a_cat * b_cat) / (N_1 * N_1)
            
            # 计算Kappa
            if (1 - pe) != 0:
                k = (p0 - pe) / (1 - pe)
            else:
                k = 0
            
            K[ij] = k
            POD[ij] = p0
            N11[ij] = N1
        
        N1 += n_step
    
    return K, POD, N11


# ==================== 第二部分：Ft1分析 ====================
print("\n第二部分：Ft1 分析（Ran_permafrost_area_observe.nc）...")

K1, POD1, N11_1 = calculate_kappa(
    frozen_files=None,
    obser_file="Ran_permafrost_area_observe.nc",
    n_range=100,
    n_step=0.01,
    categories=[0, 3]
)

# 找到最大Kappa值
y1_max = np.max(K1[K1 != 9999])
y1_ind = np.where(K1 == y1_max)[0][0]
x1_max = N11_1[y1_ind]

print(f"\nFt1 最优阈值: F = {x1_max:.2f}, Kappa = {y1_max:.4f}")


# ==================== 第三部分：Ft2分析 ====================
print("\n第三部分：Ft2 分析（frozen_2000_code_3_2_0.nc）...")

K2, POD2, N11_2 = calculate_kappa(
    frozen_files=None,
    obser_file="frozen_2000_code_3_2_0.nc",
    n_range=50,
    n_step=0.01,
    categories=[0, 2, 3]
)

# 找到最大Kappa值
y2_max = np.max(K2[K2 != 9999])
y2_ind = np.where(K2 == y2_max)[0][0]
x2_max = N11_2[y2_ind]

print(f"\nFt2 最优阈值: F = {x2_max:.2f}, Kappa = {y2_max:.4f}")


# ==================== 第四部分：Ft3分析 ====================
print("\n第四部分：Ft3 分析（frozen_2000.nc）...")

K3, POD3, N11_3 = calculate_kappa(
    frozen_files=None,
    obser_file="frozen_2000.nc",
    n_range=6,
    n_step=0.01,
    categories=[0, 1, 2, 3]
)

# 找到最大Kappa值
y3_max = np.max(K3[K3 != 9999])
y3_ind = np.where(K3 == y3_max)[0][0]
x3_max = N11_3[y3_ind]

print(f"\nFt3 最优阈值: F = {x3_max:.2f}, Kappa = {y3_max:.4f}")


# ==================== 第五部分：绘图 ====================
print("\n第五部分：生成图表...")

# 设置图表样式
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'

# 绘制Ft1结果
fig1, ax1 = plt.subplots(figsize=(10, 7.5), dpi=200)

valid_K1 = K1[K1 != 9999]
valid_N1 = N11_1[K1 != 9999]

ax1.plot(valid_N1, valid_K1, 'k-', linewidth=2.5, label='Kappa')
ax1.axvline(x=x1_max, ymin=0, ymax=y1_max, color='k', linestyle='--', linewidth=1.5)
ax1.axhline(y=y1_max, xmin=0, xmax=x1_max/1.0, color='k', linestyle='--', linewidth=1.5)

ax1.text(x1_max + 0.01, y1_max + 0.075, f'F = {x1_max:.2f}', 
         fontsize=14, verticalalignment='bottom')
ax1.text(x1_max + 0.01, y1_max + 0.035, f'Kappa = {y1_max:.2f}', 
         fontsize=14, verticalalignment='bottom')

ax1.set_xlabel('F', fontsize=14)
ax1.set_ylabel('Kappa', fontsize=14)
ax1.set_xlim([0, 1.0])
ax1.set_ylim([0, 1.0])
ax1.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
ax1.grid(True, alpha=0.3)
ax1.set_title('Ft1: Kappa vs F Threshold', fontsize=15)

plt.tight_layout()
plt.savefig('Kappa_Ft1.png', dpi=200, bbox_inches='tight')
plt.close()

# 绘制Ft2结果
fig2, ax2 = plt.subplots(figsize=(10, 7.5), dpi=200)

valid_K2 = K2[K2 != 9999]
valid_N2 = N11_2[K2 != 9999]

ax2.plot(valid_N2, valid_K2, 'k-', linewidth=2.5, label='Kappa')
ax2.axvline(x=x2_max, ymin=0, ymax=y2_max, color='k', linestyle='--', linewidth=1.5)
ax2.axhline(y=y2_max, xmin=0, xmax=x2_max/0.5, color='k', linestyle='--', linewidth=1.5)

ax2.text(x2_max + 0.01, y2_max + 0.075, f'F = {x2_max:.2f}', 
         fontsize=14, verticalalignment='bottom')
ax2.text(x2_max + 0.01, y2_max + 0.035, f'Kappa = {y2_max:.2f}', 
         fontsize=14, verticalalignment='bottom')

ax2.set_xlabel('F', fontsize=14)
ax2.set_ylabel('Kappa', fontsize=14)
ax2.set_xlim([0, 0.5])
ax2.set_ylim([0, 1.0])
ax2.set_xticks([0.1, 0.3, 0.5])
ax2.grid(True, alpha=0.3)
ax2.set_title('Ft2: Kappa vs F Threshold', fontsize=15)

plt.tight_layout()
plt.savefig('Kappa_Ft2.png', dpi=200, bbox_inches='tight')
plt.close()

# 绘制Ft3结果
fig3, ax3 = plt.subplots(figsize=(10, 7.5), dpi=200)

valid_K3 = K3[K3 != 9999]
valid_N3 = N11_3[K3 != 9999]

ax3.plot(valid_N3, valid_K3, 'k-', linewidth=2.5, label='Kappa')
ax3.axvline(x=x3_max, ymin=0, ymax=y3_max, color='k', linestyle='--', linewidth=1.5)
ax3.axhline(y=y3_max, xmin=0, xmax=x3_max/0.06, color='k', linestyle='--', linewidth=1.5)

ax3.text(x3_max + 0.002, y3_max + 0.075, f'F = {x3_max:.2f}', 
         fontsize=14, verticalalignment='bottom')
ax3.text(x3_max + 0.002, y3_max + 0.035, f'Kappa = {y3_max:.2f}', 
         fontsize=14, verticalalignment='bottom')

ax3.set_xlabel('F', fontsize=14)
ax3.set_ylabel('Kappa', fontsize=14)
ax3.set_xlim([0, 0.06])
ax3.set_ylim([0, 1.0])
ax3.set_xticks([0.01, 0.03, 0.05])
ax3.grid(True, alpha=0.3)
ax3.set_title('Ft3: Kappa vs F Threshold', fontsize=15)

plt.tight_layout()
plt.savefig('Kappa_Ft3.png', dpi=200, bbox_inches='tight')
plt.close()

print("\n分析完成！")
print(f"生成的图表: Kappa_Ft1.png, Kappa_Ft2.png, Kappa_Ft3.png")