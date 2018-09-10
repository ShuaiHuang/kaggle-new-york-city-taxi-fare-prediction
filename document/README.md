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

## 20180902

### 今日进展

1. 打通整个流程, 并整理完成所有的python脚本文件;
2. 使用`xgboost`模型提交, 准确率如下:
   > 2018-09-02 21:05:50.908984	6e6f556555d7af3dffe2858d59a40155	3.943071	3.19377

### 遗留问题

- 只使用了一部分训练集, 尽量尝试使用全部的数据集;
- 时间特征使用UTC时间, 改成夏时令更好;
- 地理位置特征未做加工, 需要进一步进行细化.

## 20180903

### 环境配置

1. `Anaconda`环境配置 Python3
2. `PyCharm` IDE配置
3. `conda install numpy feather`

### 用到的模型

1. [`xgboost`](http://dl.acm.org/citation.cfm?doid=2939672.2939785)

## 20180908

### TASK

- [x] 已有特征列描述 [Link](FEATURE-COLUMNS.md)
- [x] 分析各个生成特征的分布
- [x] 优化清洗环节
- [x] 对已有的数据加入更加严格的清洗条件
- [ ] 根据各个特征的分布对样本进行合理的采样

### 位置特征抽取

- 对于上下车经纬度全为0的记录进行删除;
- 对于上下车经纬度相同的记录, 可能是用户叫车后又取消, 出租车收取等待费用; 还有一种可能是用户把出租车当做观光巴士, 围绕某个景区产生的费用.

## 20180909

### TASK

- [x] 对地理位置信息进行抽取, 清洗掉明显不属于纽约市区的记录(注意保留与机场相关的记录);
- [x] 对训练集和验证集进行采样, 使用采样结果进行训练;
- [ ] 使用TensorFlow构造神经网络模型.

### 如何加速特征抽取

目前特征提取步骤越来越复杂, 特征提取时间越来越长(大约2h+). 从System Monitor中可以看到, 只有一个CPU核在工作, 如果使用并行化处理方式的话, 可以缩短特征提取所耗费的时间.

- [并行化 Python 中的 map, apply 等函数，aka 最简单的 Python 并行方案（示例及时间对比）](http://blog.fangzhou.me/posts/2017-07-02.html)
- [multiprocessing — Process-based parallelism](https://docs.python.org/3.7/library/multiprocessing.html)

### 总结与计划 (By Solomon)

本周的主要进展如下:
- 完善时间特征, 将原始的UTC时间转换成美国东部时间, 并加入了夏令时;
- 基于完善后的日期和时间特征, 抽取出相关特征, 包括是否夜间行驶 / 是否节假日 / 是否工作日下班高峰;
- 基于地理位置信息, 加入了起点或者终点是机场的特征, 加入预约后又取消的特征, 加强数据清洗, 对于明显不是纽约市区的行车记录进行了清洗.

但是最终训练出来的模型效果, 仅仅与最好记录持平, 并没有显著的提升. 基于这个现状, 下周的工作计划安排如下:
- 花费三天时间, 每天研究一个排名靠前的kernel, 重点关注如何进行EDA, 如何在EDA基础上设计特征, 如何进行特征选择, 模型参数调整的trick;
- 对于数据清洗和特征工程脚本, 引入并行化策略, 加速脚本运行速率;
- 考察新模型应用于此项目的可行性.

## 20180910

### Kernel分析(1)

[Top Ten Rank - R 22M Rows(2.90) LightGBM
](https://www.kaggle.com/jsylas/top-ten-rank-r-22m-rows-2-90-lightgbm)

### Kernel分析(2)

[A Walkthrough and a Challenge](https://www.kaggle.com/willkoehrsen/a-walkthrough-and-a-challenge)