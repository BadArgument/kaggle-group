# 当前最好（越低越好，更新请及时修改）
0.12183（原0.12269）

# 代码简述
当前最优模型：stacking方案，岭回归学习器学习XGBoost、LightGBM、Lasso、ElasticNet、Ridge五种模型的最优组合

#### update
`improved_stacking_method/`采用双层 Stacking 架构，集成 12 个异构基模型（含树模型、线性模型及 SVM），并利用 LightGBM 元模型结合原始重要特征进行非线性融合。同时引入 Optuna 超参数优化、伪标签半监督学习及自适应加权融合。

# 运行代码
库下载  
pip install pandas numpy scikit-learn xgboost lightgbm catboost mlxtend  

进入项目根目录  
cd kaggle-prediction  

运行预测代码  
python src/predict.py  

stacking方案：运行stacking_method/house_prices_solution.py

分层stacking方案：运行improved_stacking_method/house_prices_solution_improved.py

# 协作方式
1、Fork目标项目  
    在 GitHub 上把它fork到你自己的账号下  
2、将项目Clone到自己电脑  
3、进入项目并创建新分支  
4、保存并提交（Commit）你的修改  
5、推送到（Push）你的 GitHub 仓库  
6、发起 Pull Request (PR)，把修改请求发到我的仓库来  
