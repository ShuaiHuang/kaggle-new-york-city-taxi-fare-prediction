# 代码文件说明

| 文件       | 说明 |
| ------------ | ---- |
| exploratory-data-analysis.ipynb | EDA验证用jupyter notebook文件. |
| convert-csv-to-feather.py | 将csv格式文件转换成feather格式文件的脚本, 验证用. |
| load-feather-dataset.py | 载入feather格式文件的脚本, 验证用. |
| split-dataset-into-chunks.py | 将原始大文件分割成多个小尺寸文件. |
| clean-dataset.py | 原始数据清洗和特征提取脚本. |
| generate-dataset-for-training.py | 将清洗后的数据转换为模型输入格式. |
| train-model.py | 训练XGBoost模型. |