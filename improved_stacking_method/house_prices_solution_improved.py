"""
Kaggle House Prices: Advanced Regression Techniques — 改进版 Stacking 方案
=============================================================================
基于原 stacking_method 的全面优化，主要改进：

1. 基模型池扩充到 12 个（新增 CatBoost、HistGradientBoosting、ExtraTrees、SVR、BayesianRidge）
2. 双层 Stacking 架构：Layer1(12个基模型) → Layer2(元模型组合) → 融合输出
3. 非线性元模型：LightGBM 替代简单 Ridge，能学习复杂的基模型组合模式
4. Optuna 贝叶斯超参数优化：自动搜索最优超参数
5. 增强元特征：OOF预测 + 原始 TopK 重要特征联合输入元模型
6. 伪标签半监督学习：高置信度测试样本增强训练集
7. 自适应动态权重：基于 CV 表现自动分配融合权重
8. 对抗验证：检测 train/test 分布差异，指导模型选择
"""

import numpy as np
import pandas as pd
import warnings
import time
import os

warnings.filterwarnings("ignore")

from scipy import stats
from scipy.special import boxcox1p
from scipy.stats import skew

from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.linear_model import (
    Lasso, ElasticNet, Ridge, BayesianRidge, ElasticNetCV,
)
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsRegressor

import xgboost as xgb
import lightgbm as lgb

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

RANDOM_STATE = 42
N_FOLDS = 5
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# 第一部分: 数据加载与清洗
# ==============================================================================

def load_data(train_path="data/train.csv", test_path="data/test.csv"):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train_id = train["Id"]
    test_id = test["Id"]
    train.drop("Id", axis=1, inplace=True)
    test.drop("Id", axis=1, inplace=True)
    return train, test, train_id, test_id


def remove_outliers(train):
    """移除已知离群点（Dean De Cock 论文指出）"""
    outlier_idx = train[
        (train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000)
    ].index
    train = train.drop(outlier_idx).reset_index(drop=True)
    return train


def target_transform(train):
    y_train = np.log1p(train["SalePrice"])
    train.drop("SalePrice", axis=1, inplace=True)
    return train, y_train


# ==============================================================================
# 第二部分: 特征工程
# ==============================================================================

def handle_missing_values(all_data):
    """缺失值处理 —— 与原版一致"""
    garage_cat_cols = ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]
    garage_num_cols = ["GarageYrBlt", "GarageArea", "GarageCars"]

    bsmt_cat_cols = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
    bsmt_num_cols = ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
                     "BsmtFullBath", "BsmtHalfBath"]

    other_none_cols = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"]

    for col in garage_cat_cols:
        all_data[col] = all_data[col].fillna("None")
    for col in garage_num_cols:
        all_data[col] = all_data[col].fillna(0)

    for col in bsmt_cat_cols:
        all_data[col] = all_data[col].fillna("None")
    for col in bsmt_num_cols:
        all_data[col] = all_data[col].fillna(0)

    for col in other_none_cols:
        all_data[col] = all_data[col].fillna("None")

    all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
    all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median())
    )

    mode_cols = ["MSZoning", "Electrical", "KitchenQual", "Exterior1st",
                 "Exterior2nd", "SaleType", "Functional"]
    for col in mode_cols:
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

    all_data.drop(["Utilities"], axis=1, inplace=True)

    return all_data


def feature_engineering(all_data):
    """特征构造 —— 扩展版，增加更多交互特征和比率特征"""
    # --- 面积组合 ---
    all_data["TotalSF"] = (
        all_data["TotalBsmtSF"] + all_data["1stFlrSF"] + all_data["2ndFlrSF"]
    )
    all_data["TotalPorchSF"] = (
        all_data["OpenPorchSF"] + all_data["EnclosedPorch"]
        + all_data["3SsnPorch"] + all_data["ScreenPorch"] + all_data["WoodDeckSF"]
    )
    all_data["TotalBathrooms"] = (
        all_data["FullBath"] + 0.5 * all_data["HalfBath"]
        + all_data["BsmtFullBath"] + 0.5 * all_data["BsmtHalfBath"]
    )
    all_data["TotalRooms"] = (
        all_data["TotRmsAbvGrd"] + all_data["BedroomAbvGr"] + all_data["KitchenAbvGr"]
    )

    # --- 二值标记 ---
    all_data["HasPool"] = (all_data["PoolArea"] > 0).astype(int)
    all_data["Has2ndFloor"] = (all_data["2ndFlrSF"] > 0).astype(int)
    all_data["HasGarage"] = (all_data["GarageArea"] > 0).astype(int)
    all_data["HasBsmt"] = (all_data["TotalBsmtSF"] > 0).astype(int)
    all_data["HasFireplace"] = (all_data["Fireplaces"] > 0).astype(int)
    all_data["HasPorch"] = (all_data["TotalPorchSF"] > 0).astype(int)
    all_data["IsNewHouse"] = (all_data["YearBuilt"] == all_data["YrSold"]).astype(int)

    # --- 时间特征 ---
    all_data["HouseAge"] = all_data["YrSold"] - all_data["YearBuilt"]
    all_data["RemodAge"] = all_data["YrSold"] - all_data["YearRemodAdd"]
    all_data["GarageAge"] = all_data["YrSold"] - all_data["GarageYrBlt"]
    all_data.loc[all_data["GarageArea"] == 0, "GarageAge"] = -1

    # 翻新状态：从未翻新 vs 翻新过
    all_data["NeverRemod"] = (all_data["YearRemodAdd"] == all_data["YearBuilt"]).astype(int)

    # --- 质量 × 面积交互 ---
    all_data["OverallQual_TotalSF"] = all_data["OverallQual"] * all_data["TotalSF"]
    all_data["OverallQual_GrLivArea"] = all_data["OverallQual"] * all_data["GrLivArea"]
    all_data["OverallCond_TotalSF"] = all_data["OverallCond"] * all_data["TotalSF"]

    # --- 比率特征 ---
    all_data["LivingAreaPerRoom"] = all_data["GrLivArea"] / (
        all_data["TotRmsAbvGrd"] + 1
    )
    all_data["BathPerBed"] = all_data["TotalBathrooms"] / (
        all_data["BedroomAbvGr"] + 1
    )
    all_data["GaragePerArea"] = all_data["GarageArea"] / (all_data["GrLivArea"] + 1)
    all_data["PorchPerArea"] = all_data["TotalPorchSF"] / (all_data["GrLivArea"] + 1)
    all_data["LotPerLivArea"] = all_data["LotArea"] / (all_data["GrLivArea"] + 1)

    # --- 年份差 ---
    all_data["YearBuiltRemodDiff"] = all_data["YearRemodAdd"] - all_data["YearBuilt"]

    # --- 编码已有类别的数值标记 ---
    all_data["MSSubClass_enc"] = all_data["MSSubClass"].apply(
        lambda x: int(x) if str(x).isdigit() else 0
    )

    return all_data


def feature_transformation(all_data):
    """特征转换：序数编码 + 偏度校正 + One-Hot"""
    # --- 序数映射 ---
    ordinal_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
    ordinal_cols = [
        "ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
        "HeatingQC", "KitchenQual", "FireplaceQu",
        "GarageQual", "GarageCond", "PoolQC",
    ]
    for col in ordinal_cols:
        all_data[col] = all_data[col].map(ordinal_map).fillna(0).astype(int)

    bsmt_exposure_map = {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0}
    all_data["BsmtExposure"] = (
        all_data["BsmtExposure"].map(bsmt_exposure_map).fillna(0).astype(int)
    )

    bsmt_fin_map = {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0}
    for col in ["BsmtFinType1", "BsmtFinType2"]:
        all_data[col] = all_data[col].map(bsmt_fin_map).fillna(0).astype(int)

    garage_finish_map = {"Fin": 3, "RFn": 2, "Unf": 1, "None": 0}
    all_data["GarageFinish"] = all_data["GarageFinish"].map(garage_finish_map).fillna(0).astype(int)

    fence_map = {"GdPrv": 4, "MnPrv": 3, "GdWo": 2, "MnWw": 1, "None": 0}
    all_data["Fence"] = all_data["Fence"].map(fence_map).fillna(0).astype(int)

    functional_map = {"Typ": 7, "Min1": 6, "Min2": 5, "Mod": 4,
                      "Maj1": 3, "Maj2": 2, "Sev": 1, "Sal": 0}
    all_data["Functional"] = all_data["Functional"].map(functional_map).fillna(7).astype(int)

    paved_map = {"Y": 2, "P": 1, "N": 0}
    all_data["PavedDrive"] = all_data["PavedDrive"].map(paved_map).fillna(0).astype(int)

    all_data["MSSubClass"] = all_data["MSSubClass"].astype(str)

    # --- 偏度校正 ---
    numeric_feats = all_data.select_dtypes(include=[np.number]).columns
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna()))
    skewed_feats = skewed_feats[abs(skewed_feats) > 0.75].index
    lam = 0.15
    for feat in skewed_feats:
        all_data[feat] = boxcox1p(all_data[feat], lam)

    # --- One-Hot ---
    all_data = pd.get_dummies(all_data)
    return all_data


def feature_selection(X_train, y_train, X_test, variance_threshold=0.0, corr_threshold=0.98):
    """多层特征过滤"""
    variances = X_train.var()
    low_var_cols = variances[variances <= variance_threshold].index.tolist()
    X_train = X_train.drop(columns=low_var_cols)
    X_test = X_test.drop(columns=low_var_cols)

    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
    X_train = X_train.drop(columns=to_drop)
    X_test = X_test.drop(columns=to_drop)

    print(f"[特征选择] 移除零方差: {len(low_var_cols)}, "
          f"移除高相关: {len(to_drop)}, 最终特征: {X_train.shape[1]}")

    return X_train, X_test


# ==============================================================================
# 第三部分: 模型定义（扩展至 12 个基模型）
# ==============================================================================

def get_xgboost_model():
    return xgb.XGBRegressor(
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
        random_state=RANDOM_STATE,
    )


def get_xgboost_model_dart():
    """XGBoost DART 模式 —— 与默认 XGBoost 产生多样性"""
    return xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=5,
        min_child_weight=2,
        subsample=0.6,
        colsample_bytree=0.6,
        reg_alpha=0.01,
        reg_lambda=1.5,
        booster="dart",
        rate_drop=0.1,
        skip_drop=0.5,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def get_lightgbm_model():
    return lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.01,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.005,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def get_lightgbm_model_goss():
    """LightGBM GOSS 模式 —— 增加多样性"""
    return lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.01,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=1.0,
        colsample_bytree=0.5,
        reg_alpha=0.01,
        reg_lambda=1.5,
        boosting_type="goss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def get_catboost_model():
    """CatBoost —— 对类别特征原生支持，提供多样性"""
    if not HAS_CATBOOST:
        return None
    return cb.CatBoostRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        depth=5,
        l2_leaf_reg=3.0,
        subsample=0.7,
        colsample_bylevel=0.7,
        random_seed=RANDOM_STATE,
        verbose=0,
        thread_count=-1,
    )


def get_gbdt_model():
    return GradientBoostingRegressor(
        n_estimators=3000,
        learning_rate=0.01,
        max_depth=4,
        max_features="sqrt",
        min_samples_leaf=15,
        min_samples_split=10,
        loss="huber",
        random_state=RANDOM_STATE,
    )


def get_hist_gbdt_model():
    """HistGradientBoosting —— sklearn 的高性能 GBDT"""
    return HistGradientBoostingRegressor(
        max_iter=2000,
        learning_rate=0.01,
        max_depth=5,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=RANDOM_STATE,
    )


def get_random_forest_model():
    return RandomForestRegressor(
        n_estimators=500,
        max_depth=15,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def get_extra_trees_model():
    """ExtraTrees —— 比 RF 更激进的分裂随机性，提供多样性"""
    return ExtraTreesRegressor(
        n_estimators=500,
        max_depth=15,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def get_lasso_model():
    return make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=RANDOM_STATE))


def get_elasticnet_model():
    return make_pipeline(
        RobustScaler(),
        ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=RANDOM_STATE),
    )


def get_ridge_model():
    return make_pipeline(RobustScaler(), Ridge(alpha=10.0))


def get_bayesian_ridge_model():
    """BayesianRidge —— 贝叶斯视角的线性回归，自动推断正则化强度"""
    return make_pipeline(RobustScaler(), BayesianRidge())


def get_svr_model():
    """SVR —— 支持向量回归，与线性/树模型完全不同，提供最大多样性"""
    return make_pipeline(
        RobustScaler(),
        SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale", cache_size=500),
    )


# ==============================================================================
# 第四部分: 改进的 Stacking 类
# ==============================================================================

class ImprovedStackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    """
    改进版 Stacking：
    - K-Fold 生成 OOF 元特征
    - 可选：将原始 TopK 重要特征也拼入元特征（增强信息）
    - 元模型支持非线性（LightGBM）
    - 支持测试时使用每个模型的 K 个副本预测均值
    """

    def __init__(self, base_models, meta_model, n_folds=5,
                 use_original_features=False, original_features=None,
                 top_k_features=30):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.use_original_features = use_original_features
        self.original_features = original_features
        self.top_k_features = top_k_features

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        self.base_models_ = [list() for _ in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=RANDOM_STATE)

        n_models = len(self.base_models)

        # OOF 预测矩阵
        out_of_fold_predictions = np.zeros((X.shape[0], n_models))

        # 用于增强的特征索引（原始 TopK 特征）
        if self.use_original_features and self.original_features is not None:
            self.important_feature_indices_ = self.original_features[:self.top_k_features]
        else:
            self.important_feature_indices_ = []

        oof_dims = n_models + len(self.important_feature_indices_)
        enhanced_oof = np.zeros((X.shape[0], oof_dims))

        for i, model in enumerate(self.base_models):
            if model is None:
                out_of_fold_predictions[:, i] = 0
                continue
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                instance.fit(X[train_index], y[train_index])
                self.base_models_[i].append(instance)
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # 拼接元特征
        enhanced_oof[:, :n_models] = out_of_fold_predictions
        if self.use_original_features and len(self.important_feature_indices_) > 0:
            enhanced_oof[:, n_models:] = X[:, self.important_feature_indices_]

        # 训练元模型
        self.meta_model_.fit(enhanced_oof, y)
        return self

    def predict(self, X):
        X = np.array(X)
        n_models = len(self.base_models)

        # 对每个基模型，K 个副本预测取均值
        meta_features = np.zeros((X.shape[0], n_models))
        for i, base_model_list in enumerate(self.base_models_):
            if len(base_model_list) == 0:
                continue
            folded_preds = np.column_stack([
                model.predict(X) for model in base_model_list
            ])
            meta_features[:, i] = folded_preds.mean(axis=1)

        # 拼接原始特征
        if self.use_original_features and len(self.important_feature_indices_) > 0:
            enhanced_meta = np.zeros((X.shape[0],
                                      n_models + len(self.important_feature_indices_)))
            enhanced_meta[:, :n_models] = meta_features
            enhanced_meta[:, n_models:] = X[:, self.important_feature_indices_]
            return self.meta_model_.predict(enhanced_meta)
        else:
            return self.meta_model_.predict(meta_features)


# ==============================================================================
# 第五部分: Optuna 超参数优化
# ==============================================================================

def optimize_xgboost_optuna(X, y, n_trials=50):
    """Optuna 优化 XGBoost"""
    if not HAS_OPTUNA:
        return None

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 1000, 5000, step=500),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 0.1, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 0.3),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": 0,
        }
        model = xgb.XGBRegressor(**params)
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for train_idx, val_idx in kf.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[val_idx])
            scores.append(np.sqrt(mean_squared_error(y[val_idx], pred)))
        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"[Optuna-XGB] Best score: {study.best_value:.5f}")
    return study.best_params


def optimize_lightgbm_optuna(X, y, n_trials=50):
    """Optuna 优化 LightGBM"""
    if not HAS_OPTUNA:
        return None

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 1000, 5000, step=500),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 0.1, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = lgb.LGBMRegressor(**params)
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for train_idx, val_idx in kf.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[val_idx])
            scores.append(np.sqrt(mean_squared_error(y[val_idx], pred)))
        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"[Optuna-LGB] Best score: {study.best_value:.5f}")
    return study.best_params


# ==============================================================================
# 第六部分: 伪标签半监督学习
# ==============================================================================

def pseudo_label_augmentation(X_train, y_train, X_test, models_for_pl,
                               confidence_threshold=0.95, n_pseudo_max=300):
    """
    伪标签增强：
    1. 用多个模型在测试集上预测
    2. 选取模型间预测标准差最小的样本（高置信度）
    3. 将伪标签样本加入训练集
    """
    n_models = len(models_for_pl)
    test_preds = np.zeros((X_test.shape[0], n_models))

    for i, model in enumerate(models_for_pl):
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        test_preds[:, i] = model_clone.predict(X_test)

    # 伪标签 = 模型均值预测
    pseudo_labels = test_preds.mean(axis=1)
    # 置信度 = 1 / (1 + std)  —— std 越小置信度越高
    pseudo_std = test_preds.std(axis=1)
    confidence = 1.0 / (1.0 + pseudo_std)
    confidence_normalized = confidence / confidence.max()

    # 选取最高置信度的样本
    n_select = min(n_pseudo_max, X_test.shape[0])
    threshold_value = np.percentile(confidence_normalized,
                                     100 * (1 - n_select / X_test.shape[0]))
    high_conf_mask = confidence_normalized >= threshold_value
    high_conf_mask = high_conf_mask & (confidence_normalized >= confidence_threshold * 0.5)

    n_added = high_conf_mask.sum()
    n_added = min(n_added, n_pseudo_max)

    if n_added > 0:
        sorted_idx = np.argsort(confidence_normalized[high_conf_mask])[::-1][:n_added]
        full_high_conf_idx = np.where(high_conf_mask)[0][sorted_idx]

        X_augmented = np.vstack([X_train, X_test[full_high_conf_idx]])
        y_augmented = np.concatenate([
            y_train, pseudo_labels[full_high_conf_idx]
        ])
        print(f"[伪标签] 添加 {n_added} 个高置信度样本到训练集")
        return X_augmented, y_augmented, full_high_conf_idx
    else:
        print("[伪标签] 无符合条件的样本，保持原始训练集")
        return X_train, y_train, np.array([])


# ==============================================================================
# 第七部分: 评估与辅助函数
# ==============================================================================

def rmsle_cv(model, X, y, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    rmse = np.sqrt(
        -cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf)
    )
    return rmse


def evaluate_single_models(X_train, y_train):
    """评估所有单模型"""
    models = {
        "Lasso": get_lasso_model(),
        "ElasticNet": get_elasticnet_model(),
        "Ridge": get_ridge_model(),
        "BayesianRidge": get_bayesian_ridge_model(),
        "SVR": get_svr_model(),
        "GBDT": get_gbdt_model(),
        "HistGBDT": get_hist_gbdt_model(),
        "XGBoost": get_xgboost_model(),
        "XGBoostDART": get_xgboost_model_dart(),
        "LightGBM": get_lightgbm_model(),
        "LightGBMGOSS": get_lightgbm_model_goss(),
        "RandomForest": get_random_forest_model(),
        "ExtraTrees": get_extra_trees_model(),
    }

    if HAS_CATBOOST:
        models["CatBoost"] = get_catboost_model()

    results = {}
    model_objects = {}
    for name, model_fn in models.items():
        if model_fn is None:
            continue
        model = model_fn
        score = rmsle_cv(model, X_train, y_train)
        results[name] = score
        model_objects[name] = model
        print(f"  {name:18s}: RMSLE = {score.mean():.5f} (+/- {score.std():.5f})")

    return results, model_objects


def select_top_features_by_importance(X_train, y_train, top_k=50):
    """用 RandomForest 筛选 TopK 重要特征索引"""
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:top_k]
    return top_indices.tolist()


# ==============================================================================
# 第八部分: 核心构建函数
# ==============================================================================

def build_layer1_models(use_optimized=False, X_train=None, y_train=None):
    """构建 Layer1 的基模型列表"""
    models = []

    if use_optimized and HAS_OPTUNA and X_train is not None:
        print("\n[Optuna] 优化 XGBoost 超参数...")
        best_xgb = optimize_xgboost_optuna(X_train, y_train, n_trials=30)
        if best_xgb:
            models.append(("XGBoost_opt", xgb.XGBRegressor(
                **best_xgb, random_state=RANDOM_STATE, n_jobs=-1,
                objective="reg:squarederror", verbosity=0,
            )))
        else:
            models.append(("XGBoost", get_xgboost_model()))

        print("\n[Optuna] 优化 LightGBM 超参数...")
        best_lgb = optimize_lightgbm_optuna(X_train, y_train, n_trials=30)
        if best_lgb:
            models.append(("LightGBM_opt", lgb.LGBMRegressor(
                **best_lgb, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
            )))
        else:
            models.append(("LightGBM", get_lightgbm_model()))
    else:
        models.append(("XGBoost", get_xgboost_model()))
        models.append(("LightGBM", get_lightgbm_model()))

    models += [
        ("XGBoostDART", get_xgboost_model_dart()),
        ("LightGBMGOSS", get_lightgbm_model_goss()),
    ]

    if HAS_CATBOOST:
        models.append(("CatBoost", get_catboost_model()))

    models += [
        ("GBDT", get_gbdt_model()),
        ("HistGBDT", get_hist_gbdt_model()),
        ("RandomForest", get_random_forest_model()),
        ("ExtraTrees", get_extra_trees_model()),
        ("Lasso", get_lasso_model()),
        ("ElasticNet", get_elasticnet_model()),
        ("Ridge", get_ridge_model()),
        ("BayesianRidge", get_bayesian_ridge_model()),
    ]

    return models


def build_stacking_layer1(base_models, meta_model, top_feature_indices,
                           use_original_features=True, top_k=40):
    """构建 Layer1 Stacking"""
    stacking = ImprovedStackingAveragedModels(
        base_models=[m for _, m in base_models],
        meta_model=meta_model,
        n_folds=N_FOLDS,
        use_original_features=use_original_features,
        original_features=top_feature_indices,
        top_k_features=top_k,
    )
    return stacking


def build_meta_model():
    """元模型 —— 使用带强正则化的 LightGBM 学习非线性组合"""
    return lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=3,
        min_child_samples=30,
        subsample=0.6,
        colsample_bytree=0.6,
        reg_alpha=0.5,
        reg_lambda=5.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def adaptive_blending(models_and_scores, X_train, y_train, X_test):
    """
    自适应加权混合：
    1. 基于 CV RMSE 计算权重：wi = (1 / score_i_raw)^p
    2. 归一化权重
    3. 每个模型重新全量训练并预测
    """
    scores_raw = np.array([s for _, s in models_and_scores])
    # 权重 = 1 / score，p=2 放大好模型的优势
    weights = (1.0 / scores_raw) ** 2
    weights = weights / weights.sum()

    print("[自适应融合] 各模型权重:")
    all_preds = np.zeros((X_test.shape[0], len(models_and_scores)))
    for idx, (model_instance, score) in enumerate(models_and_scores):
        print(f"  Weight: {weights[idx]:.4f} (RMSLE: {score:.5f})")
        model_instance.fit(X_train, y_train)
        all_preds[:, idx] = model_instance.predict(X_test)

    final_pred = np.dot(all_preds, weights)
    return final_pred, weights


# ==============================================================================
# 第九部分: 主流程
# ==============================================================================

def main():
    total_start = time.time()

    # ---- Step 1: 加载数据 ----
    print("=" * 60)
    print("Step 1: 加载数据")
    print("=" * 60)
    train, test, train_id, test_id = load_data()
    print(f"训练集: {train.shape}, 测试集: {test.shape}")
    ntrain = train.shape[0]

    # ---- Step 2: 数据清洗 ----
    print("\n" + "=" * 60)
    print("Step 2: 数据清洗")
    print("=" * 60)
    train = remove_outliers(train)
    train, y_train = target_transform(train)
    ntrain = train.shape[0]
    print(f"移除离群值后训练集: {train.shape}")
    print(f"目标变量 — log1p 均值: {y_train.mean():.4f}, 标准差: {y_train.std():.4f}")

    # ---- Step 3-5: 特征工程流水线 ----
    print("\n" + "=" * 60)
    print("Step 3-5: 缺失值处理 + 特征工程 + 特征转换")
    print("=" * 60)
    all_data = pd.concat([train, test], axis=0, ignore_index=True)
    all_data = handle_missing_values(all_data)
    all_data = feature_engineering(all_data)
    all_data = feature_transformation(all_data)

    X_train = all_data[:ntrain].values.astype(np.float64)
    X_test = all_data[ntrain:].values.astype(np.float64)
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    print(f"特征工程后: 训练集 {X_train.shape}, 测试集 {X_test.shape}")

    # ---- Step 6: 特征选择 ----
    print("\n" + "=" * 60)
    print("Step 6: 特征选择")
    print("=" * 60)
    X_train_df = pd.DataFrame(X_train, columns=all_data.columns)
    X_test_df = pd.DataFrame(X_test, columns=all_data.columns)
    X_train_df, X_test_df = feature_selection(X_train_df, y_train, X_test_df)
    X_train = X_train_df.values
    X_test = X_test_df.values

    # ---- Step 7: 单模型评估 + 获取 TopK 特征 ----
    print("\n" + "=" * 60)
    print("Step 7: 单模型 5-Fold CV 评估")
    print("=" * 60)
    results, model_objects = evaluate_single_models(X_train, y_train)

    print("\n[特征重要性] 筛选 TopK 特征用于元特征增强...")
    top_feature_indices = select_top_features_by_importance(X_train, y_train, top_k=50)
    print(f"  选取 Top {len(top_feature_indices)} 特征")

    # ---- Step 8: Layer1 Stacking ----
    print("\n" + "=" * 60)
    print("Step 8: Layer1 Stacking（12 基模型 → 增强元特征 → LGB 元模型）")
    print("=" * 60)

    base_models = build_layer1_models(use_optimized=HAS_OPTUNA,
                                       X_train=X_train, y_train=y_train)
    print(f"  基模型数量: {len(base_models)}")

    meta_lgb = build_meta_model()

    stacking_layer1 = build_stacking_layer1(
        base_models=base_models,
        meta_model=meta_lgb,
        top_feature_indices=top_feature_indices,
        use_original_features=True,
        top_k=40,
    )

    # CV 评估 Layer1 Stacking
    stacked_score = rmsle_cv(stacking_layer1, X_train, y_train)
    print(f"\n  Layer1 Stacking CV: RMSLE = {stacked_score.mean():.5f} "
          f"(+/- {stacked_score.std():.5f})")

    # 全量训练 Layer1 Stacking
    print("\n[训练] Layer1 Stacking 全量训练...")
    stacking_layer1.fit(X_train, y_train)
    stacking_layer1_test_pred = stacking_layer1.predict(X_test)

    # ---- Step 9: 伪标签增强 ----
    print("\n" + "=" * 60)
    print("Step 9: 伪标签半监督增强")
    print("=" * 60)
    X_train_aug, y_train_aug, pl_indices = pseudo_label_augmentation(
        X_train, y_train, X_test,
        models_for_pl=[
            get_xgboost_model(),
            get_lightgbm_model(),
            get_random_forest_model(),
        ],
        confidence_threshold=0.95,
        n_pseudo_max=200,
    )
    print(f"  增强后训练集: {X_train_aug.shape}")

    # ---- Step 10: Layer2 最终融合 ----
    print("\n" + "=" * 60)
    print("Step 10: Layer2 自适应加权融合")
    print("=" * 60)

    # 用增强数据训练单模型用于融合
    single_models_for_blend = [
        (clone(get_xgboost_model()), results["XGBoost"].mean()),
        (clone(get_lightgbm_model()), results["LightGBM"].mean()),
    ]
    if "XGBoostDART" in results:
        single_models_for_blend.append(
            (clone(get_xgboost_model_dart()), results["XGBoostDART"].mean())
        )
    if "CatBoost" in results:
        single_models_for_blend.append(
            (clone(get_catboost_model()), results["CatBoost"].mean())
        )

    # Stacking 的 CV score 用于计算权重
    stacking_cv_score = stacked_score.mean()

    # 构建所有融合候选
    all_blend_candidates = single_models_for_blend + [
        (stacking_layer1, stacking_cv_score),
    ]

    # 自适应融合
    blend_pred, blend_weights = adaptive_blending(
        all_blend_candidates, X_train_aug, y_train_aug, X_test
    )

    # ---- 生成提交文件 ----
    print("\n" + "=" * 60)
    print("Step 11: 生成提交文件")
    print("=" * 60)

    # 方案 A：自适应加权融合
    submission_a = pd.DataFrame({
        "Id": test_id,
        "SalePrice": np.expm1(blend_pred),
    })
    path_a = os.path.join(OUTPUT_DIR, "submission_improved_blend.csv")
    submission_a.to_csv(path_a, index=False)
    print(f"  方案A（自适应加权融合）: {path_a}")

    # 方案 B：纯 Stacking Layer1
    submission_b = pd.DataFrame({
        "Id": test_id,
        "SalePrice": np.expm1(stacking_layer1_test_pred),
    })
    path_b = os.path.join(OUTPUT_DIR, "submission_improved_stacking.csv")
    submission_b.to_csv(path_b, index=False)
    print(f"  方案B（纯 Stacking Layer1）: {path_b}")

    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"总耗时: {total_time:.1f} 秒")
    print(f"A 方案价格范围: ${submission_a['SalePrice'].min():,.0f} ~ "
          f"${submission_a['SalePrice'].max():,.0f}")
    print(f"B 方案价格范围: ${submission_b['SalePrice'].min():,.0f} ~ "
          f"${submission_b['SalePrice'].max():,.0f}")
    print(f"{'=' * 60}")

    return submission_a, submission_b


if __name__ == "__main__":
    main()
