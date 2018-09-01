# CHANGELOG

## 20180829

### 解决内存问题

提供的原始数据集过大, 直接使用pandas打开16G内存吃紧, 改进方案如下:
1. 使用`read_csv()`方法时, 指定`dtype`参数, 将浮点型数据转换为`float32`类型进行解析;
2. 使用`read_csv()`方法时, 使用`chunksize`参数, 依次分块读取数据;

### 加快读取速度

1. 将整个工程移至固态硬盘所在的分区;
2. 将`DataFrame`存储为`feather`格式.

## 20180901

### 今日进展

1. `pickup_datetime` 数据离散化, 以及相关信息Exploratory Data Analysis;
2. `pickup_longitude` & `pickup_latitude` & `dropoff_longitude` & `dropoff_latitude` 数据离散化;
3. 测试集数据EDA;

### 特征工程

- 是否节假日
- 夜间时段(8:00 p.m - 6:00 a.m.)
- 时区转换(包括夏时令)
- 起点 / 终点距离估算
- 是否出城 / 进城
- 是否机场上 / 下车

