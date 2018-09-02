# XGBoost 调参方法

1. 选定一组基准参数，这些参数有经验的话，用经验值，没有经验可以用官方的默认值
2. `max_depth` & `min_child_weight` 参数调优
3. `gamma`参数调优
4. 调整 `subsample` & `colsample_bytree` 参数调优
5. 正则化参数调优 `reg_alpha` & `reg_lambda`
6. 降低学习率和使用更多的树 `learning_rate` & `n_estimators`
7. 可以探索的参数 `max_delta_step` & `scale_pos_weight` & `base_score`

## References

- [XGBoost 参数调优(python)](https://wuhuhu800.github.io/2018/02/28/XGboost_param_share/)