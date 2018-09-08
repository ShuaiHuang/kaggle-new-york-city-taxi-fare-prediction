# FEATURE COLUMNS DESCRIPTION

## COLUMN-BY-COLUMN DESCRIPTION

|列名|是否原始特征|特征描述|加入时间|备注|
|----:|:----:|:----|:----:|:----|
|key|Y|载客记录键值, 每一条载客记录都有唯一一条键值|-|-|
|fare_amount|Y|此次行程所产生车费|-|-|
|pickup_datetime|Y|此次行程开始时间|-|将此列拆解为单独的列|
|pickup_longtitude|Y|上车地点经度|-|-|
|pickup_latitude|Y|上车地点纬度|-|-|
|dropoff_longtitude|Y|下车地点经度|-|-|
|dropoff_latitude|Y|下车地点纬度|-|-|
|passenger_count|Y|乘客人数|-|-|
|pickup_datetime_obj|N|将`pickup_datetime`字符串格式化为`datetime`对象的结果|20180902|-|
|pickup_timezone|N|上车时间的时区|20180902|-|
|pickup_year|N|上车日期的年份|20180902|-|
|pickup_month|N|上车日期的月份|20180902|-|
|pickup_day|N|上车日期的天数|20180902|-|
|pickup_hour|N|上车时间的小时|20180902|-|
|pickup_minute|N|上车时间的分钟|20180902|-|
|pickup_second|N|上车时间的秒数|20180902|-|
|pickup_weekday|N|上车日期对应星期几|20180902|-|
|pickup_dropoff_distance|N|上下车地点之间球面距离|20180902|-|
|airport_jfk|N|起点或者重点是否是JFK机场|20180904|-|
|airport_lga|N|起点或者重点是否是LGA机场|20180904|-|
|airport_ewr|N|起点或者重点是否是EWR机场|20180904|-|

## Note

- `pickup_minute` / `pickup_second` 暂时未参与训练;
- `pickup_year` / `pickup_month` / `pickup_day` / `pickup_hour` / `pickup_weekday` 使用的是`one-hot`编码格式.