import numpy as np
import xarray as xr
import os
from pathlib import Path

print("开始冻土分类处理...")

# 定义模型列表和情景
models = [
    "ACCESS-CM2", "ACCESS-ESM1-5", "BCC-CSM2-MR", "CanESM5", "CMCC-ESM2",
    "CNRM-CM6-1", "CNRM-ESM2-1", "EC-Earth3", "EC-Earth3-Veg-LR", "FGOALS-g3",
    "GFDL-ESM4", "GISS-E2-1-G", "INM-CM4-8", "INM-CM5-0", "IPSL-CM6A-LR",
    "KACE-1-0-G", "MIROC-ES2L", "MIROC6", "MPI-ESM1-2-HR", "MPI-ESM1-2-LR",
    "MRI-ESM2-0", "NorESM2-LM", "NorESM2-MM", "TaiESM1", "UKESM1-0-LL"
]

scenarios = ["ssp126", "ssp245", "ssp370", "ssp585"]

# 读取地形掩膜数据
print("读取地形掩膜数据...")
mask1 = xr.open_dataset("GTOPOP_v4.nc")
mask2 = mask1['elevation'].values
mask1.close()

# 遍历所有情景
for s, scenario in enumerate(scenarios):
    scenario_upper = scenario.upper()
    
    data_dir = f"./{scenario_upper}/"
    output_dir = f"./Frozen_{scenario_upper}/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n处理情景 {s+1}/{len(scenarios)}: {scenario_upper}")
    
    # 遍历年份 1950-2099
    for y in range(1950, 2100):
        print(f"  处理年份 {y}...")
        
        # 处理每个模型
        for m, model in enumerate(models):
            fi_filename = os.path.join(data_dir, f"FI_{model}_{y}.nc")
            ti_filename = os.path.join(data_dir, f"TI_{model}_{y}.nc")
            
            # 检查文件是否存在
            if not os.path.exists(fi_filename) or not os.path.exists(ti_filename):
                print(f"    警告: 模型 {model} 的文件不存在，跳过...")
                continue
            
            try:
                # 读取FI和TI数据
                a = xr.open_dataset(fi_filename)
                FI = a['FI'].values
                lat = a['lat'].values
                lon = a['lon'].values
                
                b = xr.open_dataset(ti_filename)
                TI = b['TI'].values
                
                nlat = len(lat)
                nlon = len(lon)
                
                # 计算冻土指数 (Tibetan Plateau区域)
                frozen_TP = np.sqrt(FI) / (np.sqrt(FI) + np.sqrt(TI))
                
                # 分类 - TP区域 (3类: 0, 2, 3)
                frozen_TP_classified = np.full_like(frozen_TP, -9999, dtype=np.float32)
                frozen_TP_classified[(frozen_TP >= 0.55) & (frozen_TP <= 1)] = 3
                frozen_TP_classified[(frozen_TP < 0.55) & (frozen_TP >= 0.01)] = 2
                frozen_TP_classified[frozen_TP < 0.01] = 0
                
                # 计算冻土指数 (TP外部区域)
                frozen_TP_outside = np.sqrt(FI) / (np.sqrt(FI) + np.sqrt(TI))
                
                # 分类 - TP外部区域 (4类: 0, 1, 2, 3)
                frozen_TP_outside_classified = np.full_like(frozen_TP_outside, -9999, dtype=np.float32)
                frozen_TP_outside_classified[(frozen_TP_outside >= 0.53) & (frozen_TP_outside <= 1)] = 3
                frozen_TP_outside_classified[(frozen_TP_outside < 0.53) & (frozen_TP_outside >= 0.1)] = 2
                frozen_TP_outside_classified[(frozen_TP_outside < 0.1) & (frozen_TP_outside >= 0.01)] = 1
                frozen_TP_outside_classified[frozen_TP_outside < 0.01] = 0
                
                # 根据掩膜选择分类
                # mask2 == 1: 青藏高原区域，使用frozen_TP
                # mask2 != 1: 其他区域，使用frozen_TP_outside
                SFD = np.where(mask2 == 1, frozen_TP_classified, frozen_TP_outside_classified)
                
                # 保存结果
                output_filename = os.path.join(output_dir, f"{model}_Frozen_{scenario}_{y}.nc")
                ds_out = xr.Dataset(
                    {'frozen': (['lat', 'lon'], SFD)},
                    coords={
                        'lat': (['lat'], lat),
                        'lon': (['lon'], lon)
                    }
                )
                ds_out.to_netcdf(output_filename)
                ds_out.close()
                
                a.close()
                b.close()
                
            except Exception as e:
                print(f"    错误: 处理模型 {model} 时出错: {str(e)}")
                continue
        
        # 处理集合平均数据
        fi_ensemble_filename = os.path.join(data_dir, f"FI_ensemble_mean_{y}.nc")
        ti_ensemble_filename = os.path.join(data_dir, f"TI_ensemble_mean_{y}.nc")
        
        if os.path.exists(fi_ensemble_filename) and os.path.exists(ti_ensemble_filename):
            try:
                # 读取集合平均数据
                a_ens = xr.open_dataset(fi_ensemble_filename)
                FI_ens = a_ens['FI'].values
                lat = a_ens['lat'].values
                lon = a_ens['lon'].values
                
                b_ens = xr.open_dataset(ti_ensemble_filename)
                TI_ens = b_ens['TI'].values
                
                # 计算冻土指数 (TP区域)
                frozen_TP_ens = np.sqrt(FI_ens) / (np.sqrt(FI_ens) + np.sqrt(TI_ens))
                
                # 分类 - TP区域
                frozen_TP_ens_classified = np.full_like(frozen_TP_ens, -9999, dtype=np.float32)
                frozen_TP_ens_classified[(frozen_TP_ens >= 0.55) & (frozen_TP_ens <= 1)] = 3
                frozen_TP_ens_classified[(frozen_TP_ens < 0.55) & (frozen_TP_ens >= 0.01)] = 2
                frozen_TP_ens_classified[frozen_TP_ens < 0.01] = 0
                
                # 计算冻土指数 (TP外部区域)
                frozen_TP_outside_ens = np.sqrt(FI_ens) / (np.sqrt(FI_ens) + np.sqrt(TI_ens))
                
                # 分类 - TP外部区域
                frozen_TP_outside_ens_classified = np.full_like(frozen_TP_outside_ens, -9999, dtype=np.float32)
                frozen_TP_outside_ens_classified[(frozen_TP_outside_ens >= 0.53) & (frozen_TP_outside_ens <= 1)] = 3
                frozen_TP_outside_ens_classified[(frozen_TP_outside_ens < 0.53) & (frozen_TP_outside_ens >= 0.1)] = 2
                frozen_TP_outside_ens_classified[(frozen_TP_outside_ens < 0.1) & (frozen_TP_outside_ens >= 0.01)] = 1
                frozen_TP_outside_ens_classified[frozen_TP_outside_ens < 0.01] = 0
                
                # 根据掩膜选择分类
                SFD_ens = np.where(mask2 == 1, frozen_TP_ens_classified, frozen_TP_outside_ens_classified)
                
                # 保存集合平均结果
                ensemble_output_filename = os.path.join(output_dir, f"Ensemble_Mean_Frozen_{scenario}_{y}.nc")
                ds_ens_out = xr.Dataset(
                    {'frozen': (['lat', 'lon'], SFD_ens)},
                    coords={
                        'lat': (['lat'], lat),
                        'lon': (['lon'], lon)
                    }
                )
                ds_ens_out.to_netcdf(ensemble_output_filename)
                ds_ens_out.close()
                
                a_ens.close()
                b_ens.close()
                
                print(f"    完成年份 {y} (包括 {len(models)} 个模型 + 集合平均)")
                
            except Exception as e:
                print(f"    错误: 处理集合平均数据时出错: {str(e)}")
        else:
            print(f"    警告: 年份 {y} 的集合平均文件不存在")

print("\n冻土分类处理完成！")