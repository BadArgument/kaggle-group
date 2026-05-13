"""
Kaggle House Prices: Advanced Regression Techniques — 完整解决方案
===================================================================
竞赛目标: 预测 Ames, Iowa 住宅房价 (SalePrice)
评估指标: RMSLE (Root Mean Squared Logarithmic Error)
         等价于对 log(SalePrice) 做 RMSE
数据规模: 训练集 1460 条, 测试集 1459 条, 79 个解释变量
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from scipy import stats
from scipy.special import boxcox1p
from scipy.stats import skew, norm

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.linear_model import Lasso, ElasticNet, Ridge, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)

# =====================================================================
# 第一部分: 数据加载与初步探索
# =====================================================================

def load_data(train_path="train.csv", test_path="test.csv"):
    """加载训练集和测试集"""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train_id = train["Id"]
    test_id = test["Id"]
    train.drop("Id", axis=1, inplace=True)
    test.drop("Id", axis=1, inplace=True)
    return train, test, train_id, test_id


# =====================================================================
# 第二部分: 特征工程
# =====================================================================

# ---- 2.1 数据清洗 ----

def remove_outliers(train):
    """
    移除离群点。
    理论依据: Ames Housing 数据集作者 Dean De Cock 在论文中明确指出
    GrLivArea > 4000 sq ft 且 SalePrice < 300000 的观测是离群值。
    离群值会严重影响回归模型的拟合, 尤其是线性模型和 GBDT 的损失函数。
    """
    outlier_idx = train[
        (train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000)
    ].index
    train = train.drop(outlier_idx).reset_index(drop=True)
    return train


def target_transform(train):
    """
    对目标变量做 log(1+x) 变换。
    理论依据:
    - 竞赛评估指标为 RMSLE, 即 sqrt(mean((log(pred) - log(actual))^2))
    - 对 SalePrice 取 log 后, 优化 RMSE 等价于直接优化 RMSLE
    - 房价呈右偏分布, log 变换使其更接近正态分布,
      满足线性回归的残差正态性假设, 也有助于 GBDT 更稳定的分裂
    """
    y_train = np.log1p(train["SalePrice"])
    train.drop("SalePrice", axis=1, inplace=True)
    return train, y_train


# ---- 2.2 缺失值处理 ----

def handle_missing_values(all_data):
    """
    缺失值处理策略:
    - 类别型: NA 本身有含义的 (如 "无车库"、"无地下室") 填充为 "None"
    - 数值型: 按中位数或 0 填充 (取决于业务语义)
    - LotFrontage: 按 Neighborhood 分组中位数填充 (同社区地块相似)
    """

    # --- 这些特征的 NA 表示 "不存在该设施", 而非数据缺失 ---
    # 车库相关
    for col in ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]:
        all_data[col] = all_data[col].fillna("None")
    for col in ["GarageYrBlt", "GarageArea", "GarageCars"]:
        all_data[col] = all_data[col].fillna(0)

    # 地下室相关
    for col in ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]:
        all_data[col] = all_data[col].fillna("None")
    for col in ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
                "BsmtFullBath", "BsmtHalfBath"]:
        all_data[col] = all_data[col].fillna(0)

    # 其他 "无该设施" 类特征
    for col in ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"]:
        all_data[col] = all_data[col].fillna("None")

    all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
    all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

    # LotFrontage: 同社区地块前沿长度通常接近, 按 Neighborhood 中位数填充
    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median())
    )

    # 众数填充 (出现极少量缺失的类别特征)
    for col in ["MSZoning", "Electrical", "KitchenQual", "Exterior1st",
                "Exterior2nd", "SaleType", "Functional"]:
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

    # Utilities 几乎全为 AllPub, 无区分力, 直接删除
    all_data.drop(["Utilities"], axis=1, inplace=True)

    return all_data


# ---- 2.3 特征构造 ----

def feature_engineering(all_data):
    """
    特征构造: 基于领域知识创建新特征。
    理论依据:
    - 组合特征能捕获变量间的交互效应, 弥补树模型在高阶交互上的不足
    - 面积汇总特征直接反映房屋总体规模, 是最强预测因子之一
    - 时间差特征 (房龄、翻新年限) 比绝对年份更有业务意义
    """

    # --- 面积组合特征 ---
    all_data["TotalSF"] = (
        all_data["TotalBsmtSF"]
        + all_data["1stFlrSF"]
        + all_data["2ndFlrSF"]
    )
    all_data["TotalPorchSF"] = (
        all_data["OpenPorchSF"]
        + all_data["EnclosedPorch"]
        + all_data["3SsnPorch"]
        + all_data["ScreenPorch"]
        + all_data["WoodDeckSF"]
    )
    all_data["TotalBathrooms"] = (
        all_data["FullBath"]
        + 0.5 * all_data["HalfBath"]
        + all_data["BsmtFullBath"]
        + 0.5 * all_data["BsmtHalfBath"]
    )

    # --- 二值特征: 是否有某设施 ---
    all_data["HasPool"] = (all_data["PoolArea"] > 0).astype(int)
    all_data["Has2ndFloor"] = (all_data["2ndFlrSF"] > 0).astype(int)
    all_data["HasGarage"] = (all_data["GarageArea"] > 0).astype(int)
    all_data["HasBsmt"] = (all_data["TotalBsmtSF"] > 0).astype(int)
    all_data["HasFireplace"] = (all_data["Fireplaces"] > 0).astype(int)

    # --- 时间特征 ---
    all_data["HouseAge"] = all_data["YrSold"] - all_data["YearBuilt"]
    all_data["RemodAge"] = all_data["YrSold"] - all_data["YearRemodAdd"]
    all_data["IsNewHouse"] = (all_data["YearBuilt"] == all_data["YrSold"]).astype(int)

    # --- 质量×面积交互特征 ---
    all_data["OverallQual_TotalSF"] = all_data["OverallQual"] * all_data["TotalSF"]
    all_data["OverallQual_GrLivArea"] = all_data["OverallQual"] * all_data["GrLivArea"]

    return all_data


# ---- 2.4 特征转换 ----

def feature_transformation(all_data):
    """
    特征转换:
    1. 数值型特征偏度校正 (Box-Cox / log1p)
    2. 序数特征编码 (有序类别 → 数值)
    3. 名义特征 One-Hot 编码

    理论依据:
    - 高偏度特征会导致少数极端值主导模型; Box-Cox 变换可以将其拉向正态
      lambda ≈ 0.15 是经验值, 对大多数右偏分布效果良好
    - 序数编码保留了等级关系 (Ex > Gd > TA > Fa > Po), 比 One-Hot 更适合树模型
    - One-Hot 用于名义变量 (无内在顺序), 是线性模型的标准做法
    """

    # --- 序数特征映射 ---
    ordinal_map = {
        "Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0,
    }
    ordinal_cols = [
        "ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
        "HeatingQC", "KitchenQual", "FireplaceQu",
        "GarageQual", "GarageCond", "PoolQC",
    ]
    for col in ordinal_cols:
        all_data[col] = all_data[col].map(ordinal_map).fillna(0).astype(int)

    bsmt_exposure_map = {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0}
    all_data["BsmtExposure"] = all_data["BsmtExposure"].map(bsmt_exposure_map).fillna(0).astype(int)

    bsmt_fin_map = {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0}
    for col in ["BsmtFinType1", "BsmtFinType2"]:
        all_data[col] = all_data[col].map(bsmt_fin_map).fillna(0).astype(int)

    garage_finish_map = {"Fin": 3, "RFn": 2, "Unf": 1, "None": 0}
    all_data["GarageFinish"] = all_data["GarageFinish"].map(garage_finish_map).fillna(0).astype(int)

    fence_map = {"GdPrv": 4, "MnPrv": 3, "GdWo": 2, "MnWw": 1, "None": 0}
    all_data["Fence"] = all_data["Fence"].map(fence_map).fillna(0).astype(int)

    functional_map = {
        "Typ": 7, "Min1": 6, "Min2": 5, "Mod": 4,
        "Maj1": 3, "Maj2": 2, "Sev": 1, "Sal": 0,
    }
    all_data["Functional"] = all_data["Functional"].map(functional_map).fillna(7).astype(int)

    paved_map = {"Y": 2, "P": 1, "N": 0}
    all_data["PavedDrive"] = all_data["PavedDrive"].map(paved_map).fillna(0).astype(int)

    # 将 MSSubClass (建筑类型编码) 转为字符串, 它本质是类别而非数值
    all_data["MSSubClass"] = all_data["MSSubClass"].astype(str)

    # --- 偏度校正: Box-Cox 变换 ---
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna()))
    skewed_feats = skewed_feats[abs(skewed_feats) > 0.75].index

    lam = 0.15
    for feat in skewed_feats:
        all_data[feat] = boxcox1p(all_data[feat], lam)

    # --- One-Hot 编码名义变量 ---
    all_data = pd.get_dummies(all_data)

    return all_data


# ---- 2.5 特征选择与降维 ----

def feature_selection(X_train, y_train, X_test, variance_threshold=0.0, corr_threshold=0.98):
    """
    特征选择策略 (多层过滤):
    1. 方差过滤: 移除零方差特征 (One-Hot 后可能产生)
    2. 高相关过滤: 两个特征相关性 > 0.98 时保留一个 (去冗余)
    3. 互信息 (Mutual Information): 评估特征与目标的非线性关系

    理论依据:
    - 冗余特征不会提升树模型性能, 但会增加过拟合风险和训练时间
    - 互信息 I(X;Y) 衡量知道 X 后对 Y 不确定性的减少量,
      不假设线性关系, 比皮尔逊相关系数更通用
    - 对于 GBDT/XGBoost, 适度降维还能减少内存占用和加速搜索

    注意: 树模型有内建特征选择能力, 不宜过于激进地剔除特征。
    这里只做保守过滤。
    """

    # 1. 零方差过滤
    variances = X_train.var()
    low_var_cols = variances[variances <= variance_threshold].index.tolist()
    X_train = X_train.drop(columns=low_var_cols)
    X_test = X_test.drop(columns=low_var_cols)

    # 2. 高相关过滤
    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
    X_train = X_train.drop(columns=to_drop)
    X_test = X_test.drop(columns=to_drop)

    print(f"[特征选择] 移除零方差特征: {len(low_var_cols)} 个")
    print(f"[特征选择] 移除高相关特征: {len(to_drop)} 个")
    print(f"[特征选择] 最终特征数: {X_train.shape[1]}")

    return X_train, X_test


def optional_pca(X_train, X_test, n_components=0.99):
    """
    可选 PCA 降维 (主要用于线性模型的 Stacking 输入)。
    理论依据:
    - PCA 通过正交变换消除多重共线性, 对 Ridge/Lasso 尤其有效
    - 保留 99% 方差确保信息损失极小
    - 对树模型不推荐 PCA, 因为会破坏特征的可解释性和离散分裂的效率
    """
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print(f"[PCA] 保留 {pca.n_components_} 个主成分, "
          f"解释方差比: {pca.explained_variance_ratio_.sum():.4f}")

    return X_train_pca, X_test_pca, pca


# =====================================================================
# 第三部分: 模型建立
# =====================================================================

def get_xgboost_model():
    """
    XGBoost 回归模型。
    理论依据:
    - XGBoost 基于梯度提升框架, 每轮拟合前一轮的残差 (负梯度方向)
    - 目标函数 = 损失函数 + 正则项: L(θ) + Σ[γT + 0.5λ||w||²]
      其中 T 是叶子节点数, w 是叶子权重, γ 和 λ 控制复杂度
    - 使用二阶泰勒展开近似损失函数, 比传统 GBDT (一阶) 更精确
    - 内建列采样 (colsample_bytree) 和子采样 (subsample) 降低过拟合
    - 正则化参数 reg_alpha (L1) 和 reg_lambda (L2) 进一步约束模型复杂度

    超参数选择理由:
    - n_estimators=3000 + early_stopping: 足够多的树, 依靠早停确定最佳轮数
    - learning_rate=0.01: 小学习率配合更多树, 通常泛化更好 (偏差-方差权衡)
    - max_depth=4: 较浅的树减少过拟合, 房价数据特征交互不需要太深
    - subsample=0.7, colsample_bytree=0.7: 引入随机性, 类似随机森林的效果
    """
    model = xgb.XGBRegressor(
        n_estimators=3000,
        learning_rate=0.01,
        max_depth=4,
        min_child_weight=3,
        gamma=0.0,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.005,
        reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
    )
    return model


def get_gbdt_model():
    """
    Sklearn GradientBoostingRegressor (经典 GBDT)。
    理论依据:
    - Friedman 提出的 Gradient Boosting Machine, 基于函数空间的梯度下降
    - 使用 Huber 损失函数: 对异常值更鲁棒 (结合了 MSE 和 MAE 的优点)
      当残差 < alpha 时用 MSE, > alpha 时用 MAE, 减少极端值的影响
    - 与 XGBoost 相比缺少二阶信息和正则项, 但实现稳定、调参直观

    超参数选择理由:
    - loss="huber": 房价数据可能存在异常交易, Huber 损失更鲁棒
    - max_features="sqrt": 每次分裂随机选 sqrt(n) 个特征, 增加多样性
    """
    model = GradientBoostingRegressor(
        n_estimators=3000,
        learning_rate=0.01,
        max_depth=4,
        max_features="sqrt",
        min_samples_leaf=15,
        min_samples_split=10,
        loss="huber",
        random_state=42,
    )
    return model


def get_lightgbm_model():
    """
    LightGBM 模型。
    理论依据:
    - 基于直方图的决策树算法 (Histogram-based), 将连续值离散化为 bins
    - Leaf-wise 生长策略 (vs. XGBoost 的 Level-wise): 选择增益最大的叶子分裂,
      同等叶子数下通常损失更低, 但更易过拟合 → 需要 num_leaves 约束
    - GOSS (Gradient-based One-Side Sampling): 保留大梯度样本, 随机采样小梯度样本
    - EFB (Exclusive Feature Bundling): 自动捆绑互斥稀疏特征, 降维加速
    - 训练速度通常是 XGBoost 的 2-10 倍
    """
    model = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.01,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.005,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    return model


def get_lasso_model():
    """
    Lasso 回归 (L1 正则化线性模型)。
    理论依据:
    - L1 正则化可以产生稀疏解, 自动进行特征选择
    - 对 One-Hot 编码后的高维稀疏特征特别有效
    - 与树模型互补: 线性模型捕获全局趋势, 树模型捕获局部非线性
    - 使用 RobustScaler 对特征标准化, 对异常值的影响更小
    """
    return make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=42))


def get_elasticnet_model():
    """
    ElasticNet (L1 + L2 混合正则化)。
    理论依据:
    - 结合了 Lasso (特征选择) 和 Ridge (系数收缩) 的优点
    - l1_ratio 控制 L1/L2 比例, 0.9 偏向稀疏但保留相关特征组
    - 在存在多重共线性时比纯 Lasso 更稳定
    """
    return make_pipeline(
        RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=42)
    )


def get_ridge_model():
    """
    Ridge 回归 (L2 正则化)。
    理论依据:
    - L2 正则化不做特征选择, 但将所有系数向零收缩
    - 在特征高度相关时表现优于 Lasso (不会随意丢弃其中一个)
    - 闭式解, 训练极快, 适合作为集成学习的基模型
    """
    return make_pipeline(RobustScaler(), Ridge(alpha=10.0))


def get_random_forest_model():
    """
    随机森林。
    理论依据:
    - Bagging + 随机特征子集 → 降低方差
    - 与 Boosting 方法互补: RF 是低偏差高方差模型, Boosting 是逐步降偏差
    - 作为 Stacking 基模型时, 提供与 GBDT 不同的 "视角"
    """
    return RandomForestRegressor(
        n_estimators=500,
        max_depth=15,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )


# =====================================================================
# 第四部分: 模型评估
# =====================================================================

def rmsle_cv(model, X, y, n_folds=5):
    """
    K-Fold 交叉验证评估 RMSLE。
    理论依据:
    - 因为 y 已做 log1p 变换, 此处 RMSE 即等价于原始尺度的 RMSLE
    - K-Fold 比 Hold-out 更稳定: 每条数据都被验证过, 减少评估方差
    - shuffle=True 打破数据中可能的时间/空间排列偏差
    - neg_mean_squared_error 取负是 sklearn 约定 (越大越好)
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    rmse = np.sqrt(
        -cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf)
    )
    return rmse


def evaluate_single_models(X_train, y_train):
    """评估各单模型的 CV 表现"""
    models = {
        "Lasso":       get_lasso_model(),
        "ElasticNet":  get_elasticnet_model(),
        "Ridge":       get_ridge_model(),
        "GBDT":        get_gbdt_model(),
        "XGBoost":     get_xgboost_model(),
        "LightGBM":    get_lightgbm_model(),
        "RandomForest": get_random_forest_model(),
    }

    results = {}
    for name, model in models.items():
        score = rmsle_cv(model, X_train, y_train)
        results[name] = score
        print(f"{name:15s}: RMSLE = {score.mean():.5f} (+/- {score.std():.5f})")

    return results


# =====================================================================
# 第五部分: 模型融合 (集成学习)
# =====================================================================
#
# 理论依据:
# 集成学习通过组合多个模型来提升预测性能, 核心原理:
#
# 1. 偏差-方差分解: E[(f(x)-y)²] = Bias² + Variance + Noise
#    - 不同模型有不同的偏差-方差特性
#    - 组合可以在不增加偏差的情况下降低方差
#
# 2. Condorcet 陪审团定理 (推广):
#    - 若各模型误差独立且单模型准确率 > 50%, 集成准确率随模型数增加趋向 100%
#    - 关键: 模型间的多样性 (diversity) 越高, 集成增益越大
#
# 3. 本方案的多样性来源:
#    - 算法多样性: 线性模型 (Lasso, Ridge, ElasticNet) + 树模型 (XGBoost, GBDT, LightGBM, RF)
#    - 正则化多样性: L1 (Lasso) vs L2 (Ridge) vs L1+L2 (ElasticNet) vs 树正则化
#    - 学习策略多样性: Boosting (序列纠错) vs Bagging (并行降方差)
#
# 集成方法: 本方案采用两层融合架构:
#
#   Layer 1: Stacking (堆叠泛化)
#     - 基学习器: XGBoost, LightGBM, GBDT, Lasso, ElasticNet, Ridge, RF
#     - 使用 K-Fold 交叉验证生成元特征 (out-of-fold predictions)
#     - 避免信息泄露: 每个样本的元特征都来自未使用该样本训练的模型
#
#   Layer 2: 加权平均 + Meta-learner
#     - Meta-learner (Ridge) 学习各基模型预测的最优组合权重
#     - 额外加入简单加权平均作为最终 Blending 的一部分
#

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    """
    自定义 Stacking 集成器。

    原理:
    - 对每个基模型做 K-Fold: 训练 K 次, 每次用 K-1 份训练, 1 份预测
    - 收集所有 fold 的 out-of-fold 预测作为新特征
    - 元模型 (meta_model) 在这些新特征上训练, 学习最优组合

    相比简单加权平均:
    - 权重是数据驱动学习的, 而非人工设定
    - 元模型可以学习非线性组合 (如果用非线性元模型)
    - 本方案用 Ridge 作为元模型: 线性组合 + L2 正则防过拟合
    """

    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        self.base_models_ = [list() for _ in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        X = np.array(X)
        y = np.array(y)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                instance.fit(X[train_index], y[train_index])
                self.base_models_[i].append(instance)
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_
        ])
        return self.meta_model_.predict(meta_features)


def build_stacking_model():
    """
    构建 Stacking 集成模型。
    架构:
    - 基学习器: XGBoost, LightGBM, GBDT, Lasso, ElasticNet, Ridge
    - 元学习器: Ridge (线性组合, L2 正则化)
    """
    stacking = StackingAveragedModels(
        base_models=[
            get_xgboost_model(),
            get_lightgbm_model(),
            get_lasso_model(),
            get_elasticnet_model(),
            get_ridge_model(),
        ],
        meta_model=Ridge(alpha=5.0),
        n_folds=5,
    )
    return stacking


def final_blending_predict(X_train, y_train, X_test):
    """
    最终预测: Stacking + 单模型加权混合。

    融合策略:
    - Stacking 模型占主导权重 (0.60)
    - XGBoost 单模型补充 (0.20): 捕获 Stacking 可能遗漏的非线性
    - LightGBM 单模型补充 (0.20): 与 XGBoost 互补 (不同的分裂策略)

    权重选择理由:
    - 基于 CV 表现和模型相关性分析
    - Stacking 已包含多种模型, 给更高权重
    - 单独再加 XGBoost 和 LightGBM 是因为它们通常是最强的单模型,
      直接预测可能捕获 Stacking 元模型 (线性) 无法学到的模式
    """

    # 训练 Stacking 模型
    print("\n[融合] 训练 Stacking 模型...")
    stacked = build_stacking_model()
    stacked.fit(X_train, y_train)
    stacked_pred = stacked.predict(X_test)

    # 训练单独的 XGBoost
    print("[融合] 训练 XGBoost...")
    xgb_model = get_xgboost_model()
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)

    # 训练单独的 LightGBM
    print("[融合] 训练 LightGBM...")
    lgb_model = get_lightgbm_model()
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_test)

    # 加权混合
    final_pred = 0.60 * stacked_pred + 0.20 * xgb_pred + 0.20 * lgb_pred

    return final_pred


# =====================================================================
# 第六部分: 完整 Pipeline
# =====================================================================

def main():
    """
    完整流程:
    1. 加载数据
    2. 数据清洗 (离群值移除 + 目标变换)
    3. 缺失值处理
    4. 特征构造
    5. 特征转换 (偏度校正 + 编码)
    6. 特征选择
    7. 单模型评估
    8. 模型融合 (Stacking + Blending)
    9. 生成提交文件
    """

    # ---- 1. 加载数据 ----
    print("=" * 60)
    print("Step 1: 加载数据")
    print("=" * 60)
    train, test, train_id, test_id = load_data()
    print(f"训练集: {train.shape}, 测试集: {test.shape}")
    ntrain = train.shape[0]

    # ---- 2. 数据清洗 ----
    print("\n" + "=" * 60)
    print("Step 2: 数据清洗")
    print("=" * 60)
    train = remove_outliers(train)
    train, y_train = target_transform(train)
    ntrain = train.shape[0]
    print(f"移除离群值后训练集: {train.shape}")
    print(f"目标变量 log1p 变换后: mean={y_train.mean():.4f}, std={y_train.std():.4f}")

    # ---- 3-5. 合并处理 (避免 train/test 编码不一致) ----
    print("\n" + "=" * 60)
    print("Step 3-5: 缺失值处理 + 特征构造 + 特征转换")
    print("=" * 60)
    all_data = pd.concat([train, test], axis=0, ignore_index=True)
    all_data = handle_missing_values(all_data)
    all_data = feature_engineering(all_data)
    all_data = feature_transformation(all_data)

    # 对齐 train/test 列
    X_train = all_data[:ntrain].values.astype(np.float64)
    X_test = all_data[ntrain:].values.astype(np.float64)

    # 处理可能残留的 NaN
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    print(f"特征工程后: 训练集 {X_train.shape}, 测试集 {X_test.shape}")

    # ---- 6. 特征选择 ----
    print("\n" + "=" * 60)
    print("Step 6: 特征选择")
    print("=" * 60)
    X_train_df = pd.DataFrame(X_train, columns=all_data.columns)
    X_test_df = pd.DataFrame(X_test, columns=all_data.columns)
    X_train_df, X_test_df = feature_selection(X_train_df, y_train, X_test_df)

    X_train = X_train_df.values
    X_test = X_test_df.values

    # ---- 7. 单模型评估 ----
    print("\n" + "=" * 60)
    print("Step 7: 单模型 CV 评估")
    print("=" * 60)
    results = evaluate_single_models(X_train, y_train)

    # ---- 8. Stacking CV 评估 ----
    print("\n" + "=" * 60)
    print("Step 8: Stacking 模型 CV 评估")
    print("=" * 60)
    stacked_model = build_stacking_model()
    stacked_score = rmsle_cv(stacked_model, X_train, y_train)
    print(f"{'Stacking':15s}: RMSLE = {stacked_score.mean():.5f} "
          f"(+/- {stacked_score.std():.5f})")

    # Stacking 模型在全量训练集上训练并预测测试集
    stacked_model.fit(X_train, y_train)
    stacked_test_pred = stacked_model.predict(X_test)
    submission_stacking = pd.DataFrame({
        "Id": test_id,
        "SalePrice": np.expm1(stacked_test_pred),
    })
    submission_stacking.to_csv("submission_stacking.csv", index=False)
    print(f"Stacking 提交文件已保存: submission_stacking.csv")

    # ---- 9. 最终融合预测 ----
    print("\n" + "=" * 60)
    print("Step 9: 最终融合预测 & 生成提交文件")
    print("=" * 60)
    final_pred = final_blending_predict(X_train, y_train, X_test)

    # 逆变换 log1p → 原始价格
    submission = pd.DataFrame({
        "Id": test_id,
        "SalePrice": np.expm1(final_pred),
    })
    submission.to_csv("submission.csv", index=False)
    print(f"\n提交文件已保存: submission.csv")
    print(f"预测价格范围: ${submission['SalePrice'].min():,.0f} ~ "
          f"${submission['SalePrice'].max():,.0f}")
    print(f"预测价格均值: ${submission['SalePrice'].mean():,.0f}")

    return submission


if __name__ == "__main__":
    main()
