from statsmodels.tsa.seasonal import STL
import xarray as xr
import numpy as np
ds = xr.open_dataset("/mnt/e/Research_life/DATA/GRACE-RAW/TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.1_V3/GRCTellus.JPL.200204_202211.GLO.RL06.1M.MSCNv03CRI.nc")
time = ds.time[:]
time = np.append(time, np.datetime64('2022-12-16T00:00:00','s'))
ds1 = xr.open_dataset("/mnt/e/Research_life/DATA/GRACE-mascon/CSR_GRACE_GRACE-FO_RL06_Mascons_all-corrections_v02.nc")
grace = ds1.assign_coords(time=time).lwe_thickness
a = grace.to_series()
res_robust = STL(a,period=7,robust=True).fit()
res_ds=res_robust.trend
b = res_ds.to_xarray()
b.to_netcdf("/mnt/e/Research_life/DATA/GRACE-mascon/deseasonal.nc") # 保存为result.nc文件

