# Demo

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 22:40:48 2022

@author: DELL
"""
#%% code interpretation
# 1.random forest test
# 2.Reference 
#   https://data36.com/random-forest-in-python/
#   https://mljar.com/blog/feature-importance-in-random-forest/#:~:text=Random%20Forest%20Built%2Din%20Feature,a%20set%20of%20Decision%20Trees.
#%%
import sys
sys.path.append("/mnt/c/Users/DELL/Desktop/play_code_d/Linux")
import pandas as pd
import shap
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
from identify_staionpointis import identify_staionpoint_isin_whichgird
#%%
# =============================================================================
# Variabels:TWC,ET,Pre,PET,SM_pre,NDVI
# =============================================================================
## load data
drought_event = pd.read_csv("/mnt/c/Users/DELL/Desktop/2021wr031829-sup-0002-data set si-s01.csv")

grace = xr.open_dataset("/mnt/e/Research_life/DATA/GRACE-REC/7670849/01_monthly_grids_ensemble_means_allmodels/01_monthly_grids_ensemble_means_allmodels/GRACE_REC_v03_JPL_ERA5_monthly_ensemble_mean.nc").rec_ensemble_mean
with xr.open_mfdataset('/mnt/e/Research_life/DATA/Evapotranspiration-DOLCE/*.nc') as f:  # 批量读取文件并合并
    ET = f['hfls'] 
EP = xr.open_dataset("/mnt/e/Research_life/DATA/GLEAM/data/v3.6a/monthly/Ep_1980-2021_GLEAM_v3.6a_MO.nc").Ep
Pre = xr.open_dataset("/mnt/e/Research_life/DATA/Precipitation/precip.mon.total.v2018.nc").precip
SM_Surf = xr.open_dataset("/mnt/e/Research_life/DATA/GLEAM/data/v3.6a/monthly/SMsurf_1980-2021_GLEAM_v3.6a_MO.nc").SMsurf
NDVI = xr.open_dataset("/mnt/e/Research_life/DATA/NDVI/GMMIS/NDVI_mvc_USA_mon_1982_2015.nc").NDVI_mvc

Pre['lon'] = Pre['lon'].values-360 # original data ranges from 0 to 360


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

## interp
Lat = np.arange(19.5,50.5,0.5)
Lon=np.arange(-110.5,-89.5,0.5)

grace = grace.interp(
    lat = Lat,
    lon = Lon,
    method_non_numeric='nearest',
    kwargs={
        "fill_value":"extrapolate"
        }
    )

ET = ET.interp(
    lat = Lat,
    lon = Lon,
    method_non_numeric='nearest',
    kwargs={
        "fill_value":"extrapolate"
        }
    )

EP = EP.interp(
    lat = Lat,
    lon = Lon,
    method_non_numeric='nearest',
    kwargs={
        "fill_value":"extrapolate"
        }
    )

Pre = Pre.interp(
    lat = Lat,
    lon = Lon,
    method_non_numeric='nearest',
    kwargs={
        "fill_value":"extrapolate"
        }
    )

SM_Surf = SM_Surf.interp(
    lat = Lat,
    lon = Lon,
    method_non_numeric='nearest',
    kwargs={
        "fill_value":"extrapolate"
        }
    )

NDVI = NDVI.interp(
    lat = Lat,
    lon = Lon,
    method_non_numeric='nearest',
    kwargs={
        "fill_value":"extrapolate"
        }
    )

# NDVI['time'] = np.arange('2001-01', '2016-01', dtype='datetime64[M]')

#%
## Extract all variable of the point according to <one> location of the selected point <site_name>
def total_RF(Lon,Lat,loc,site_name):
    ## load station site location 
    lat,lon = identify_staionpoint_isin_whichgird(Lon,Lat,loc,site_name)[0]

    time = pd.to_datetime(drought_event[['year']+['month']+['day']])
    drought_num = drought_event.iloc[:,3]

    # bool_value[j] = i.time.dt.strftime('%Y-%m').isin(time.dt.strftime('%Y-%m'))

    et = ET.loc[ET.time.dt.strftime('%Y-%m').isin(time.dt.strftime('%Y-%m')),lat,lon]
    Et = np.squeeze(et.values.reshape(-1,1))

    twc = grace.loc[grace.time.dt.strftime('%Y-%m').isin(time.dt.strftime('%Y-%m')),lat,lon]
    TWC =  np.squeeze(twc.values.reshape(-1,1))

    Ep = np.squeeze(EP.loc[EP.time.dt.strftime('%Y-%m').isin(time.dt.strftime('%Y-%m')),lat,lon].values.reshape(-1,1))
    pre = np.squeeze(Pre.loc[Pre.time.dt.strftime('%Y-%m').isin(time.dt.strftime('%Y-%m')),lat,lon].values.reshape(-1,1))
    SM_surf = np.squeeze(SM_Surf.loc[SM_Surf.time.dt.strftime('%Y-%m').isin((time-np.timedelta64(1,'M')).dt.strftime('%Y-%m')),lat,lon].values.reshape(-1,1))
    ndvi = np.squeeze(NDVI.loc[NDVI.time.dt.strftime('%Y-%m').isin(time.dt.strftime('%Y-%m')),lat,lon].values.reshape(-1,1))

    dro_inp = pd.concat([drought_num,pd.Series(Et),pd.Series(TWC),pd.Series(Ep),pd.Series(pre),pd.Series(SM_surf),pd.Series(ndvi)],axis=1)
    dro_inp = dro_inp.dropna(how='any')
    dro_inp.columns = ['Binary','Et','TWC','Ep','pre','SM_surf','ndvi']
    return dro_inp,lat,lon

#% Extract all variable of the point according to <all of> location of the selected point
Lat = np.arange(19.5,50.5,0.5)
Lon=np.arange(-110.5,-89.5,0.5)
loc = pd.read_excel('/mnt/c/Users/DELL/Desktop/wrr_site_loc.xlsx').dropna(how='any')

dro_inp = pd.DataFrame(columns = ['Binary','Et','TWC','Ep','pre','SM_surf','ndvi'])
LAT = []
LON = []
for i in range(loc.shape[0]):
    dro_inp = pd.concat([dro_inp,total_RF(Lon,Lat,loc,loc['name'][i])[0]]) # [0] for ['Binary','Et','TWC','Ep','pre','SM_surf','ndvi']
    LON.append(total_RF(Lon,Lat,loc,loc['name'][i])[1]) # [1] for lat
    LAT.append(total_RF(Lon,Lat,loc,loc['name'][i])[2]) # [2] for lon
    # lat = np.concatenate(total_RF(Lon,Lat,loc,loc['name'][i])[1])
    # lon = np.concatenate(total_RF(Lon,Lat,loc,loc['name'][i])[2])
# total_RF(Lon,Lat,loc,loc['name'][0])

#% random forest model
X = dro_inp.iloc[:,1:]
y = dro_inp.iloc[:,0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

rf_model = RandomForestClassifier(n_estimators=500, max_features="auto", bootstrap = True, random_state=44,oob_score=True)
rf_model.fit(X_train, y_train)

## predict with our model 
predictions = rf_model.predict(X_test)
predictions

rf_model.predict_proba(X_test)
rf_model.classes_
importances = rf_model.feature_importances_

## graph fig1
plt.rcParams.update({'figure.figsize': (12.0, 8.0)})
plt.rcParams.update({'font.size': 14})
plt.barh(X.columns, importances)

## fig2
sorted_idx = importances.argsort()
plt.barh(X.columns[sorted_idx], importances[sorted_idx])
plt.xlabel("Random Forest Feature Importance")

## report
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, predictions)
print("Classification Report:")
print (result1)
result2 = accuracy_score(y_test,predictions)
print("Accuracy:",result2)
print("oob error:",1-rf_model.oob_score_)


#% extract six variables for all of grid

# Variabels:TWC,ET,Pre,PET,SM_Surf,NDVI
# start_yr = 1982
# end_yr = 2015

# start_yr = 2010
# end_yr = 2012

# interval = end_yr-start_yr+1
# n_var = 6
# x_predict = np.full((12*interval*n_var,len(grace.lat),len(grace.lon)),np.nan)
# for i in range(len(grace.lat)):
#     for j in range(len(grace.lon)):
#         x_predict[:,i,j] = np.concatenate((grace.loc[str(start_yr):str(end_yr)][:,i,j],ET.loc[str(start_yr):str(end_yr)][:,i,j],
#                             EP.loc[str(start_yr):str(end_yr)][:,i,j],Pre.loc[str(start_yr):str(end_yr)][:,i,j],
#                             SM_Surf.loc[str(start_yr):str(end_yr)][:,i,j],NDVI.loc[str(start_yr):str(end_yr)][:,i,j]),axis=0)



## multi processes
import time
from multiprocessing.dummy import Pool as ThreadPool
import threading
start_yr = 2010
end_yr = 2012
interval = end_yr-start_yr+1
n_var = 6
x_predict = np.full((12*interval*n_var,len(grace.lat),len(grace.lon)),np.nan)

def predict(grace,ET,x_predict,start_yr,end_yr):
    for i in range(len(grace.lat)):
        for j in range(len(grace.lon)):
            x_predict[:,i,j] = np.concatenate((grace.loc[str(start_yr):str(end_yr)][:,i,j],ET.loc[str(start_yr):str(end_yr)][:,i,j],
                                EP.loc[str(start_yr):str(end_yr)][:,i,j],Pre.loc[str(start_yr):str(end_yr)][:,i,j],
                                SM_Surf.loc[str(start_yr):str(end_yr)][:,i,j],NDVI.loc[str(start_yr):str(end_yr)][:,i,j]),axis=0)
    return x_predict

t = threading.Thread(target=predict,args=(grace,ET,x_predict,start_yr,end_yr))
t.start()
t.join()
#%% compute drought probability for <all of> grid

n = int(len(x_predict)/n_var)
mon_id = 0
# x_predict[0:n,i,j][mon_id]
# x_predict[n:n*2,i,j][mon_id]
# x_predict[n*2:n*3,i,j][mon_id]
# x_predict[n*3:n*4,i,j][mon_id]
# x_predict[n*4:n*5,i,j][mon_id]
# x_predict[n*4:n*6,i,j][mon_id]

drought_pro = np.full((len(grace.lat),len(grace.lon)),np.nan)
for i in range(len(grace.lat)):
    for j in range(len(grace.lon)):
        X_predict = pd.DataFrame(np.array([x_predict[0:n,i,j][mon_id],x_predict[n:n*2,i,j][mon_id],
                                            x_predict[n*2:n*3,i,j][mon_id],x_predict[n*3:n*4,i,j][mon_id],
                                            x_predict[n*4:n*5,i,j][mon_id],x_predict[n*4:n*6,i,j][mon_id]])).T
        X_predict.columns = ['Et','TWC','Ep','pre','SM_surf','ndvi']
        try:
            drought_pro[i,j] = rf_model.predict_proba(X_predict)[0][1]
        except:
            drought_pro[i,j] = np.nan

