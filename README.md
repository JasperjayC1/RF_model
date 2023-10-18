# RF_model

1. USA_Texas文件是WRR文章中参考干旱报告进行随机森林的模拟，其中包含identify_staionpointis.ipynb、Random forest.ipynb两个文件，分别用来识别Texas下州县以及运算；

2. China文件是根据七大气候分区建立随机森林模型预测干旱。
   
   其中MODIS_landcover.ipynb目的是将MODIS年土地利用hdf数据转为月netcdf数据（LC数据赋给每月的原因是预测干旱的研究中需要每月的土地利用数据）
   
3. RF_model.py文件为demo
