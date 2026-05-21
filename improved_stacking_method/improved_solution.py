"""
Kaggle House Prices: Advanced Regression Techniques
====================================================
改进版 Stacking 解决方案

核心改进:
1. 更丰富的特征工程 (综合面积、年代、质量交互、缺失标志)
2. 目标编码处理高基数类别特征
3. 基模型多样化 (加入 CatBoost, KNN)
4. 增强型 Stacking (元特征扩充: 非线性变换 + 差值特征 + 跳过连接)
5. 元学习器可选 (Ridge / Huber / ElasticNet / LightGBM)
6. 多折重复平均提升稳定性
7. 模型相关性分析去除冗余基模型
"""

import numpy as np
import pandas as pd
import warnings
import os
from copy import deepcopy

warnings.filterwarnings("ignore")

from scipy import stats
from scipy.special import boxcox1p
from scipy.stats import skew, boxcox

from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.linear_model import (
    Lasso, ElasticNet, Ridge, HuberRegressor, LassoCV, RidgeCV, BayesianRidge
)
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)

import xgboost as xgb
import lightgbm as lgb

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("[警告] CatBoost 未安装，将跳过 CatBoost 模型")

# =====================================================================
# 全局配置
# =====================================================================

RANDOM_SEED = 42
N_FOLDS = 10           # Stacking 交叉验证折数
CV_FOLDS = 5           # 评估用折数
TARGET_ENCODE_FOLDS = 5 # 目标编码折数


# =====================================================================
# 第一部分: 数据加载与清洗
# =====================================================================

def load_data(train_path="data/train.csv", test_path="data/test.csv"):
    """加载训练集和测试集"""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train_id = train["Id"].copy()
    test_id = test["Id"].copy()
    train.drop("Id", axis=1, inplace=True)
    test.drop("Id", axis=1, inplace=True)
    return train, test, train_id, test_id


def remove_outliers(train):
    """
    移除离群点:
    1. GrLivArea > 4000 且 SalePrice < 300000 (数据作者指定)
    2. TotalBsmtSF > 6000 (输入错误)
    """
    idx1 = train[(train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000)].index
    idx2 = train[train["TotalBsmtSF"] > 6000].index
    outlier_idx = idx1.union(idx2)
    train = train.drop(outlier_idx).reset_index(drop=True)
    print(f"  移除离群点 {len(outlier_idx)} 个")
    return train


def target_transform(train):
    """对目标变量做 log1p 变换"""
    y_train = np.log1p(train["SalePrice"])
    train.drop("SalePrice", axis=1, inplace=True)
    return train, y_train


# =====================================================================
# 第二部分: 特征工程
# =====================================================================

def handle_missing_values(all_data):
    """
    缺失值处理:
    - NA 有业务含义的填 'None'
    - 数值缺失填 0 / 中位数 / 众数
    - 为重要缺失特征添加 _is_missing 标志
    """
    original_index = all_data.index

    # --- 记录缺失标志 (在填补之前) ---
    missing_cols = [
        'LotFrontage', 'MasVnrArea', 'MasVnrType',
        'GarageYrBlt', 'GarageArea', 'GarageCars',
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
        'BsmtFullBath', 'BsmtHalfBath',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
    ]
    for col in missing_cols:
        if col in all_data.columns:
            missing_mask = all_data[col].isna()
            if missing_mask.any():
                all_data.loc[missing_mask, f'{col}_is_missing'] = 1
                # 确保非缺失行也有该列
                all_data.loc[~missing_mask, f'{col}_is_missing'] = 0

    # --- 类别1: NA = "没有该设施" ---
    none_cols = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'MasVnrType'
    ]
    for col in none_cols:
        if col in all_data.columns:
            all_data[col] = all_data[col].fillna('None')

    # --- 类别2: NA = 0 ---
    zero_cols = [
        'GarageYrBlt', 'GarageArea', 'GarageCars',
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
        'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
    ]
    for col in zero_cols:
        if col in all_data.columns:
            all_data[col] = all_data[col].fillna(0)

    # --- 类别3: 按 Neighborhood 中位数填充 LotFrontage ---
    if 'LotFrontage' in all_data.columns:
        all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )

    # --- 类别4: 众数填充 ---
    mode_cols = ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st',
                 'Exterior2nd', 'SaleType', 'Functional']
    for col in mode_cols:
        if col in all_data.columns:
            all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

    # --- 类别5: 去掉无区分度特征 ---
    if 'Utilities' in all_data.columns:
        all_data.drop(['Utilities'], axis=1, inplace=True)

    return all_data


def target_encode_feature(train_series, y_train, test_series, n_fold=5, random_state=42):
    """
    留一法目标编码 (LOO Target Encoding).
    对训练集用 K-Fold 内类别均值编码，测试集用全局类别均值。
    """
    train_encoded = np.zeros(len(train_series), dtype=np.float64)
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=random_state)

    global_mean = y_train.mean()

    for tr_idx, te_idx in kf.split(train_series):
        for cat in train_series.iloc[tr_idx].unique():
            mask = (train_series.iloc[te_idx] == cat)
            cat_mask = (train_series.iloc[tr_idx] == cat)
            cat_mean = y_train.iloc[tr_idx][cat_mask].mean()
            train_encoded[te_idx[mask.values]] = cat_mean if not np.isnan(cat_mean) else global_mean

    # 未在训练 fold 中出现过的类别用全局均值
    unseen_mask = (train_encoded == 0) & (~train_series.isna())
    # 只对实际出现过的用全局均值
    train_encoded[train_encoded == 0] = global_mean

    # 测试集: 用训练集全局类别均值
    cat_means = train_series.groupby(train_series.values).apply(
        lambda x: y_train.loc[x.index].mean()
    ).to_dict()
    test_encoded = test_series.map(cat_means).fillna(global_mean).values

    return train_encoded, test_encoded


def perform_target_encoding(train_df, test_df, y_train, high_card_cols=None):
    """
    对高基数类别特征做目标编码。
    返回编码后的 DataFrame 和新增列名列表。
    """
    if high_card_cols is None:
        # 检测高基数 (>6 个类别) 的有序/名义特征
        high_card_cols = []
        for col in train_df.columns:
            if train_df[col].dtype == 'object':
                n_unique = train_df[col].nunique()
                if n_unique > 6:
                    high_card_cols.append(col)

    encoded_col_names = []
    for col in high_card_cols:
        if col not in train_df.columns:
            continue
        te_name = f'{col}_target_enc'
        train_enc, test_enc = target_encode_feature(
            train_df[col], y_train, test_df[col],
            n_fold=TARGET_ENCODE_FOLDS, random_state=RANDOM_SEED
        )
        train_df[te_name] = train_enc
        test_df[te_name] = test_enc
        encoded_col_names.append(te_name)

    return train_df, test_df, encoded_col_names


def feature_engineering(all_data):
    """
    特征构造: 领域知识驱动的新特征
    """
    # --- 面积组合 ---
    all_data['TotalSF'] = (
        all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
    )
    all_data['TotalPorchSF'] = (
        all_data['OpenPorchSF'] + all_data['EnclosedPorch']
        + all_data['3SsnPorch'] + all_data['ScreenPorch']
        + all_data['WoodDeckSF']
    )
    all_data['TotalBathrooms'] = (
        all_data['FullBath'] + 0.5 * all_data['HalfBath']
        + all_data['BsmtFullBath'] + 0.5 * all_data['BsmtHalfBath']
    )
    # 每平方英尺价格 (仅训练集，在对数空间近似)
    all_data['LivAreaPerRoom'] = all_data['GrLivArea'] / (all_data['TotRmsAbvGrd'].replace(0, 1))
    all_data['BathPerSF'] = all_data['TotalBathrooms'] / (all_data['TotalSF'].replace(0, 1))

    # --- 二值特征 ---
    all_data['HasPool'] = (all_data['PoolArea'] > 0).astype(int)
    all_data['Has2ndFloor'] = (all_data['2ndFlrSF'] > 0).astype(int)
    all_data['HasGarage'] = (all_data['GarageArea'] > 0).astype(int)
    all_data['HasBsmt'] = (all_data['TotalBsmtSF'] > 0).astype(int)
    all_data['HasFireplace'] = (all_data['Fireplaces'] > 0).astype(int)
    all_data['HasPorch'] = (all_data['TotalPorchSF'] > 0).astype(int)

    # --- 年代特征 ---
    all_data['HouseAge'] = all_data['YrSold'].astype(int) - all_data['YearBuilt']
    all_data['RemodAge'] = all_data['YrSold'].astype(int) - all_data['YearRemodAdd']
    all_data['IsRemod'] = (all_data['YearRemodAdd'] != all_data['YearBuilt']).astype(int)
    all_data['IsNewHouse'] = (all_data['YearBuilt'] == all_data['YrSold'].astype(int)).astype(int)
    all_data['GarageAge'] = all_data['YrSold'].astype(int) - np.where(
        all_data['GarageYrBlt'] == 0, all_data['YearBuilt'], all_data['GarageYrBlt'])

    # 年代分箱 (每10年)
    all_data['HouseAgeBin'] = pd.cut(all_data['HouseAge'], bins=range(0, 150, 10),
                                     labels=False, right=False).fillna(0).astype(int)

    # --- 质量综合评分 ---
    all_data['OverallGrade'] = all_data['OverallQual'] * all_data['OverallCond']
    all_data['OverallQual_TotalSF'] = all_data['OverallQual'] * all_data['TotalSF']
    all_data['OverallQual_GrLivArea'] = all_data['OverallQual'] * all_data['GrLivArea']
    all_data['OverallQual_LotArea'] = all_data['OverallQual'] * np.log1p(all_data['LotArea'])
    all_data['ExterGrade'] = all_data['ExterQual'].map(
        {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}
    ).fillna(3).astype(int) * all_data['ExterCond'].map(
        {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}
    ).fillna(3).astype(int) if 'ExterQual' in all_data.columns and 'ExterCond' in all_data.columns else 0

    # --- 地段与面积交互 ---
    if 'Neighborhood' in all_data.columns:
        median_lot_by_nb = all_data.groupby('Neighborhood')['LotArea'].transform('median')
        all_data['LotArea_vs_Neighborhood'] = all_data['LotArea'] / median_lot_by_nb.replace(0, 1)
        # 更多 Neighborhood 分组聚合
        for col in ['GrLivArea', 'TotalSF', 'OverallQual', 'YearBuilt']:
            if col in all_data.columns:
                median_val = all_data.groupby('Neighborhood')[col].transform('median')
                all_data[f'{col}_vs_NbMedian'] = all_data[col] / median_val.replace(0, 1)

    # --- 聚类特征: 基于关键连续变量做 KMeans ---
    cluster_feats = ['LotArea', 'TotalSF', 'OverallQual', 'YearBuilt', 'TotalBathrooms']
    valid_cf = [f for f in cluster_feats if f in all_data.columns]
    if len(valid_cf) >= 3:
        cluster_df = all_data[valid_cf].fillna(0)
        cluster_scaled = RobustScaler().fit_transform(cluster_df)
        all_data['Cluster'] = KMeans(n_clusters=8, random_state=RANDOM_SEED, n_init=10).fit_predict(cluster_scaled)

    return all_data


def feature_transformation(all_data):
    """
    特征转换: 偏度校正、序数编码、One-Hot 编码
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
        if col in all_data.columns:
            all_data[col] = all_data[col].map(ordinal_map).fillna(0).astype(int)

    bsmt_exposure_map = {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0}
    if "BsmtExposure" in all_data.columns:
        all_data["BsmtExposure"] = all_data["BsmtExposure"].map(bsmt_exposure_map).fillna(0).astype(int)

    bsmt_fin_map = {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0}
    for col in ["BsmtFinType1", "BsmtFinType2"]:
        if col in all_data.columns:
            all_data[col] = all_data[col].map(bsmt_fin_map).fillna(0).astype(int)

    garage_finish_map = {"Fin": 3, "RFn": 2, "Unf": 1, "None": 0}
    if "GarageFinish" in all_data.columns:
        all_data["GarageFinish"] = all_data["GarageFinish"].map(garage_finish_map).fillna(0).astype(int)

    fence_map = {"GdPrv": 4, "MnPrv": 3, "GdWo": 2, "MnWw": 1, "None": 0}
    if "Fence" in all_data.columns:
        all_data["Fence"] = all_data["Fence"].map(fence_map).fillna(0).astype(int)

    functional_map = {
        "Typ": 7, "Min1": 6, "Min2": 5, "Mod": 4,
        "Maj1": 3, "Maj2": 2, "Sev": 1, "Sal": 0,
    }
    if "Functional" in all_data.columns:
        all_data["Functional"] = all_data["Functional"].map(functional_map).fillna(7).astype(int)

    paved_map = {"Y": 2, "P": 1, "N": 0}
    if "PavedDrive" in all_data.columns:
        all_data["PavedDrive"] = all_data["PavedDrive"].map(paved_map).fillna(0).astype(int)

    # MSSubClass 转为字符串
    if "MSSubClass" in all_data.columns:
        all_data["MSSubClass"] = all_data["MSSubClass"].astype(str)

    # --- 偏度校正: 每特征自动搜索最优 Box-Cox λ ---
    numeric_feats = all_data.select_dtypes(include=[np.number]).columns
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(pd.to_numeric(x, errors='coerce').dropna()))
    skewed_feats = skewed_feats[abs(skewed_feats) > 0.75].index

    for feat in skewed_feats:
        col_data = pd.to_numeric(all_data[feat], errors='coerce').dropna()
        if len(col_data) == 0 or (col_data <= 0).any() or col_data.nunique() < 10:
            # 有非正值或离散度过低，跳过或用 log1p
            all_data[feat] = np.log1p(all_data[feat].clip(lower=0))
        else:
            try:
                # 自动搜索最佳 λ
                transformed, lam = boxcox(col_data + 1e-6)
                mapping = dict(zip(col_data.index, transformed))
                all_data[feat] = all_data[feat].map(mapping).fillna(0)
            except Exception:
                # 回退到固定 λ=0.15
                all_data[feat] = boxcox1p(all_data[feat].clip(lower=0), 0.15)

    # --- 对数值型特征做 log1p 变换 (针对高度右偏的特征) ---
    log_transform_cols = ['LotArea', 'GrLivArea', '1stFlrSF']
    for col in log_transform_cols:
        if col in all_data.columns:
            all_data[f'Log{col}'] = np.log1p(all_data[col])

    # --- One-Hot 编码 ---
    all_data = pd.get_dummies(all_data)

    return all_data


# =====================================================================
# 第三部分: 特征选择
# =====================================================================

def feature_selection(X_train, y_train, X_test,
                      variance_threshold=0.0, corr_threshold=0.98):
    """方差过滤 + 高相关过滤"""
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

    print(f"  [特征选择] 移除零方差: {len(low_var_cols)}个, "
          f"高相关: {len(to_drop)}个, 最终: {X_train.shape[1]}个特征")

    return X_train, X_test


# =====================================================================
# 第四部分: 模型定义
# =====================================================================

def get_xgboost_model(**kwargs):
    params = {
        'n_estimators': 3000,
        'learning_rate': 0.01,
        'max_depth': 4,
        'min_child_weight': 3,
        'gamma': 0.0,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.005,
        'reg_lambda': 1.0,
        'objective': 'reg:squarederror',
        'n_jobs': -1,
        'random_state': RANDOM_SEED,
    }
    params.update(kwargs)
    return xgb.XGBRegressor(**params)


def get_lightgbm_model(**kwargs):
    params = {
        'n_estimators': 3000,
        'learning_rate': 0.01,
        'num_leaves': 31,
        'max_depth': -1,
        'min_child_samples': 20,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.005,
        'reg_lambda': 1.0,
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
        'verbose': -1,
    }
    params.update(kwargs)
    return lgb.LGBMRegressor(**params)


def get_catboost_model(**kwargs):
    if not HAS_CATBOOST:
        return None
    params = {
        'iterations': 3000,
        'learning_rate': 0.01,
        'depth': 4,
        'l2_leaf_reg': 3,
        'subsample': 0.8,
        'colsample_bylevel': 0.8,
        'eval_metric': 'RMSE',
        'random_seed': RANDOM_SEED,
        'logging_level': 'Silent',
        'allow_writing_files': False,
    }
    params.update(kwargs)
    return CatBoostRegressor(**params)


def get_gbdt_model(**kwargs):
    params = {
        'n_estimators': 3000,
        'learning_rate': 0.01,
        'max_depth': 4,
        'max_features': 'sqrt',
        'min_samples_leaf': 15,
        'min_samples_split': 10,
        'loss': 'huber',
        'random_state': RANDOM_SEED,
    }
    params.update(kwargs)
    return GradientBoostingRegressor(**params)


def get_lasso_model(**kwargs):
    return make_pipeline(
        RobustScaler(),
        Lasso(alpha=0.0005, random_state=RANDOM_SEED, **kwargs)
    )


def get_elasticnet_model(**kwargs):
    return make_pipeline(
        RobustScaler(),
        ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=RANDOM_SEED, **kwargs)
    )


def get_ridge_model(**kwargs):
    return make_pipeline(
        RobustScaler(),
        Ridge(alpha=10.0, **kwargs)
    )

def get_knn_model(**kwargs):
    return make_pipeline(
        RobustScaler(),
        KNeighborsRegressor(n_neighbors=7, weights='distance', n_jobs=-1, **kwargs)
    )

def get_svr_model(**kwargs):
    return make_pipeline(
        RobustScaler(),
        SVR(kernel='rbf', C=5.0, epsilon=0.01, gamma='scale', **kwargs)
    )

def get_mlp_model(**kwargs):
    return make_pipeline(
        RobustScaler(),
        MLPRegressor(
            hidden_layer_sizes=(128, 64), activation='relu',
            early_stopping=True, validation_fraction=0.1,
            max_iter=2000, random_state=RANDOM_SEED, **kwargs
        )
    )

def get_bayesian_ridge_model(**kwargs):
    return make_pipeline(
        RobustScaler(),
        BayesianRidge(max_iter=300, alpha_1=1e-6, alpha_2=1e-6,
                      lambda_1=1e-6, lambda_2=1e-6, **kwargs)
    )

def get_xgboost_shallow_model(**kwargs):
    """浅层 XGBoost 变体: 深度更小，学习率更大，提供不同偏差-方差剖面"""
    params = {
        'n_estimators': 2000, 'learning_rate': 0.03, 'max_depth': 2,
        'min_child_weight': 5, 'gamma': 0.0, 'subsample': 0.8,
        'colsample_bytree': 0.6, 'reg_alpha': 0.01, 'reg_lambda': 2.0,
        'objective': 'reg:squarederror', 'n_jobs': -1,
        'random_state': RANDOM_SEED,
    }
    params.update(kwargs)
    return xgb.XGBRegressor(**params)

def get_all_base_models():
    """返回所有基模型字典 {name: model}"""
    models = {
        'XGBoost': get_xgboost_model(),
        'XGBoost_shallow': get_xgboost_shallow_model(),
        'LightGBM': get_lightgbm_model(),
        'GBDT': get_gbdt_model(),
        'Ridge': get_ridge_model(),
        'Lasso': get_lasso_model(),
        'ElasticNet': get_elasticnet_model(),
        'BayesianRidge': get_bayesian_ridge_model(),
        'KNN': get_knn_model(),
        'SVR': get_svr_model(),
        'CatBoost': get_catboost_model(),
    }
    return {k: v for k, v in models.items() if v is not None}


# =====================================================================
# 第五部分: 模型评估
# =====================================================================

def rmsle_cv(model, X, y, n_folds=5):
    """K-Fold 交叉验证评估 RMSLE (因 y 已 log1p, RMSE = RMSLE)"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    rmse = np.sqrt(-cross_val_score(
        model, X, y, scoring="neg_mean_squared_error", cv=kf, n_jobs=1
    ))
    return rmse


def evaluate_single_models(X_train, y_train, models_dict):
    """评估各单模型的 CV 表现"""
    results = {}
    for name, model in models_dict.items():
        try:
            score = rmsle_cv(model, X_train, y_train, n_folds=CV_FOLDS)
            results[name] = score
            print(f"  {name:15s}: RMSLE = {score.mean():.5f} (std={score.std():.5f})")
        except Exception as e:
            print(f"  {name:15s}: 评估失败 - {e}")
    return results


# =====================================================================
# 第六部分: 增强型 Stacking
# =====================================================================

class EnhancedStackingRegressor(BaseEstimator, RegressorMixin):
    """
    增强型 Stacking 回归器。

    改进点:
    1. 标准 K-Fold out-of-fold 预测生成元特征
    2. 元特征扩充: 预测值的平方 / 开方 / 绝对值
    3. 基模型预测差值特征 (捕获分歧样本)
    4. 原始特征跳过连接 (可选)
    5. 模型相关性分析辅助去冗余
    """

    def __init__(self, base_models, meta_model, n_folds=10,
                 use_transforms=True, use_diffs=True,
                 use_raw_features=False, random_state=42):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.use_transforms = use_transforms
        self.use_diffs = use_diffs
        self.use_raw_features = use_raw_features
        self.random_state = random_state

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).ravel()
        n_models = len(self.base_models)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        # Out-of-fold predictions
        oof_preds = np.zeros((X.shape[0], n_models))
        self.trained_models_ = [[] for _ in range(n_models)]

        for i, model in enumerate(self.base_models):
            for train_idx, val_idx in kf.split(X):
                model_clone = clone(model)
                model_clone.fit(X[train_idx], y[train_idx])
                oof_preds[val_idx, i] = model_clone.predict(X[val_idx])
                self.trained_models_[i].append(model_clone)

        # 构造增强元特征
        meta_features = self._build_meta_features(oof_preds, X if self.use_raw_features else None)

        # 训练元学习器
        self.meta_model_ = clone(self.meta_model)
        self.meta_model_.fit(meta_features, y)

        # 保存元特征形状供预测使用
        self.meta_feat_n_ = meta_features.shape[1]

        return self

    def predict(self, X):
        X = np.array(X)
        n_models = len(self.base_models)

        # 每个基模型的 K 折子模型投票平均
        base_preds = np.column_stack([
            np.mean([model.predict(X) for model in models], axis=0)
            for models in self.trained_models_
        ])

        meta_features = self._build_meta_features(base_preds, X if self.use_raw_features else None)
        return self.meta_model_.predict(meta_features)

    def _build_meta_features(self, base_preds, raw_X=None):
        """构造增强元特征，并对元特征做标准化"""
        feats = [base_preds]

        if self.use_transforms:
            feats.append(base_preds ** 2)
            feats.append(np.sqrt(np.abs(base_preds)))

        if self.use_diffs and base_preds.shape[1] > 1:
            n = base_preds.shape[1]
            for i in range(n):
                for j in range(i + 1, n):
                    feats.append((base_preds[:, i] - base_preds[:, j]).reshape(-1, 1))

        if raw_X is not None:
            feats.append(raw_X)

        meta = np.column_stack(feats)
        # 标准化元特征，防止不同尺度主导元学习器
        if not hasattr(self, 'meta_scaler_'):
            self.meta_scaler_ = StandardScaler()
            meta = self.meta_scaler_.fit_transform(meta)
        else:
            meta = self.meta_scaler_.transform(meta)
        return meta


def compute_model_correlations(models_dict, X_train, y_train, n_folds=5):
    """
    计算各基模型预测值之间的相关性矩阵。
    用于去除相关系数 > 0.95 的冗余模型。
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    oof_preds = {}

    for name, model in models_dict.items():
        preds = np.zeros(len(y_train))
        for train_idx, val_idx in kf.split(X_train):
            model_clone = clone(model)
            model_clone.fit(X_train[train_idx], y_train[train_idx])
            preds[val_idx] = model_clone.predict(X_train[val_idx])
        oof_preds[name] = preds

    preds_df = pd.DataFrame(oof_preds)
    corr_matrix = preds_df.corr()

    print("\n  基模型预测相关性矩阵:")
    print(f"  {corr_matrix.to_string()}")

    # 找出相关 > 0.95 的模型对
    redundant_pairs = []
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    for col in upper.columns:
        high_corr = upper[col][upper[col] > 0.95]
        for idx in high_corr.index:
            redundant_pairs.append((idx, col, high_corr[idx]))

    if redundant_pairs:
        print(f"  高相关模型对 (>0.95):")
        for m1, m2, corr_val in redundant_pairs:
            print(f"    {m1} - {m2}: {corr_val:.4f}")

    return corr_matrix, redundant_pairs


# =====================================================================
# 第七部分: 超参数优化 (Optuna)
# =====================================================================

def tune_model_optuna(X_train, y_train, model_name, n_trials=50, cv_folds=5):
    """
    使用 Optuna 对单个模型做贝叶斯超参数搜索。

    参数:
        X_train, y_train: 训练数据
        model_name: 'XGBoost' / 'LightGBM' / 'CatBoost' / 'Ridge' / 'Lasso' / 'ElasticNet'
        n_trials: Optuna 试验次数
        cv_folds: 交叉验证折数

    返回:
        (best_params, best_score)
    """
    if not HAS_OPTUNA:
        print(f"  [警告] Optuna 未安装，跳过 {model_name} 调参")
        return {}, float('inf')

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)

    def objective(trial):
        if model_name == 'XGBoost':
            params = {
                'max_depth': 2,
                'learning_rate': 0.01931,
                'subsample': 0.69288,
                'colsample_bytree': 0.48587,
                'reg_alpha': 2.0293e-4,
                'reg_lambda': 2.9652,
                'min_child_weight': 6,
                'gamma': 0.01476,
                'n_estimators': 3000,
                'objective': 'reg:squarederror',
                'n_jobs': -1,
                'random_state': RANDOM_SEED,
                'verbosity': 0,
            }
            model = xgb.XGBRegressor(**params)

        elif model_name == 'LightGBM':
            params = {
                'num_leaves': trial.suggest_int('num_leaves', 8, 128),
                'learning_rate': trial.suggest_float('learning_rate', 0.003, 0.05, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 80),
                'subsample': trial.suggest_float('subsample', 0.5, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
                'n_estimators': 3000,
                'max_depth': -1,
                'random_state': RANDOM_SEED,
                'n_jobs': -1,
                'verbose': -1,
            }
            model = lgb.LGBMRegressor(**params)

        elif model_name == 'CatBoost':
            if not HAS_CATBOOST:
                return float('inf')
            params = {
                'depth': trial.suggest_int('depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.003, 0.05, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 0.9),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 0.9),
                'iterations': 3000,
                'eval_metric': 'RMSE',
                'random_seed': RANDOM_SEED,
                'logging_level': 'Silent',
                'allow_writing_files': False,
            }
            model = CatBoostRegressor(**params)

        elif model_name == 'Ridge':
            params = {
                'alpha': trial.suggest_float('alpha', 0.1, 50.0, log=True),
            }
            model = make_pipeline(RobustScaler(), Ridge(**params))

        elif model_name == 'Lasso':
            params = {
                'alpha': trial.suggest_float('alpha', 1e-5, 0.01, log=True),
                'max_iter': 100000,
                'random_state': RANDOM_SEED,
            }
            model = make_pipeline(RobustScaler(), Lasso(**params))

        elif model_name == 'ElasticNet':
            params = {
                'alpha': trial.suggest_float('alpha', 1e-5, 0.01, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.1, 0.95),
                'max_iter': 100000,
                'random_state': RANDOM_SEED,
            }
            model = make_pipeline(RobustScaler(), ElasticNet(**params))
        else:
            return float('inf')

        scores = -cross_val_score(model, X_train, y_train,
                                  scoring='neg_mean_squared_error',
                                  cv=kf, n_jobs=-1)
        return np.sqrt(scores.mean())

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=1)

    print(f"    {model_name}: best RMSLE={study.best_value:.6f}")
    return study.best_params, study.best_value


def tune_all_base_models(X_train, y_train, model_names, n_trials=50):
    """
    对指定的基模型列表逐一做 Optuna 调参，返回优化后的参数字典。

    返回: {model_name: best_params}
    """
    if not HAS_OPTUNA:
        print("[警告] Optuna 未安装，将使用默认超参数。")
        print("  安装: pip install optuna")
        return {}

    print("\n超参数优化 (Optuna Bayesian Search)")
    print("-" * 40)

    optimized_params = {}
    tunable_models = ['XGBoost', 'LightGBM', 'CatBoost', 'Ridge', 'Lasso', 'ElasticNet']

    for name in model_names:
        if name not in tunable_models:
            continue
        print(f"  优化 {name} ({n_trials} trials)...")
        try:
            best_params, best_score = tune_model_optuna(
                X_train, y_train, name, n_trials=n_trials
            )
            optimized_params[name] = best_params
        except Exception as e:
            print(f"    {name} 调参失败: {e}")

    return optimized_params


def build_optimized_models(model_names, optimized_params):
    """
    根据调优后的参数构建模型实例。

    返回: {name: model_instance}
    """
    optimized_models = {}
    for name in model_names:
        factory = _get_model_factory(name)
        if factory is None:
            continue
        if name in optimized_params and optimized_params[name]:
            optimized_models[name] = factory(**optimized_params[name])
        else:
            optimized_models[name] = factory()
    return optimized_models


# =====================================================================
# 第八部分: PCA 特征扩充
# =====================================================================

def add_pca_features(X_train, X_test, n_components=0.99):
    """
    对原始特征做 PCA，将主成分拼接到原特征矩阵末尾。
    为线性模型提供去共线性的特征表示。
    """
    scaler = RobustScaler()
    combined = np.vstack([X_train, X_test])
    scaled = scaler.fit_transform(combined)
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    pca_transformed = pca.fit_transform(scaled)
    n = X_train.shape[0]
    X_train_new = np.hstack([X_train, pca_transformed[:n]])
    X_test_new = np.hstack([X_test, pca_transformed[n:]])
    print(f"  [PCA] 增加 {pca.n_components_} 个主成分, "
          f"总解释方差: {pca.explained_variance_ratio_.sum():.4f}")
    return X_train_new, X_test_new

def final_blending_predict(X_train, y_train, X_test, selected_models_dict,
                           meta_model=None, n_seeds=1):
    """
    最终融合预测:
    - 使用增强型 Stacking
    - 可选多种子重复平均
    - 加权混合 Stacking + 最佳单模型
    """
    model_names = list(selected_models_dict.keys())
    print(f"\n  [融合] 基模型: {model_names}")

    if meta_model is None:
        meta_model = Ridge(alpha=5.0)

    # 多种子运行
    all_stacked_preds = []
    all_xgb_preds = []
    all_lgb_preds = []

    seeds = [RANDOM_SEED] if n_seeds == 1 else \
            [RANDOM_SEED + i * 100 for i in range(n_seeds)]

    for seed_idx, seed in enumerate(seeds):
        if n_seeds > 1:
            print(f"  [融合] 种子 {seed_idx + 1}/{n_seeds} (seed={seed})")

        # 用当前种子重新初始化模型
        base_models = []
        for name in model_names:
            factory = _get_model_factory(name)
            if factory:
                base_models.append(factory())

        # Stacking (增强型)
        stacking = EnhancedStackingRegressor(
            base_models=base_models,
            meta_model=clone(meta_model),
            n_folds=N_FOLDS,
            use_transforms=True,
            use_diffs=True,
            use_raw_features=False,
            random_state=seed,
        )
        stacking.fit(X_train, y_train)
        stacked_pred = stacking.predict(X_test)
        all_stacked_preds.append(stacked_pred)

        # 额外训练 XGBoost 和 LightGBM 作为补充
        if 'XGBoost' in model_names:
            xgb_model = get_xgboost_model(random_state=seed)
            xgb_model.fit(X_train, y_train)
            all_xgb_preds.append(xgb_model.predict(X_test))

        if 'LightGBM' in model_names:
            lgb_model = get_lightgbm_model(random_state=seed)
            lgb_model.fit(X_train, y_train)
            all_lgb_preds.append(lgb_model.predict(X_test))

    # 多种子平均
    stacked_pred = np.mean(all_stacked_preds, axis=0)

    # 加权融合: Stacking 主导 + 最佳单模型补充
    final_pred = 0.60 * stacked_pred

    if all_xgb_preds:
        final_pred += 0.20 * np.mean(all_xgb_preds, axis=0)

    if all_lgb_preds:
        final_pred += 0.20 * np.mean(all_lgb_preds, axis=0)

    return final_pred


def _get_model_factory(name):
    """根据模型名返回工厂函数"""
    factories = {
        'XGBoost': get_xgboost_model,
        'XGBoost_shallow': get_xgboost_shallow_model,
        'LightGBM': get_lightgbm_model,
        'CatBoost': get_catboost_model,
        'GBDT': get_gbdt_model,
        'Ridge': get_ridge_model,
        'Lasso': get_lasso_model,
        'ElasticNet': get_elasticnet_model,
        'BayesianRidge': get_bayesian_ridge_model,
        'KNN': get_knn_model,
        'SVR': get_svr_model,
        'MLP': get_mlp_model,
    }
    return factories.get(name)


# =====================================================================
# 第十部分: 完整 Pipeline
# =====================================================================

def run_pipeline(
    train_path="data/train.csv",
    test_path="data/test.csv",
    do_feature_selection=True,
    do_target_encoding=True,
    do_correlation_analysis=True,
    do_optuna_tuning=True,
    do_pca=True,
    meta_learner='ridge',
    n_seeds=3,
    n_optuna_trials=50,
):
    """
    完整 Pipeline 入口。

    参数:
        meta_learner: 'ridge', 'elasticnet', 'huber', 'lgb', 'bayesian'
        n_seeds: 多种子重复次数 (1=不重复)
        do_optuna_tuning: 是否启用 Optuna 超参数优化
        do_pca: 是否拼接 PCA 特征
        n_optuna_trials: Optuna 试验次数
    """

    print("=" * 60)
    print("改进版 Stacking 解决方案 - 完整 Pipeline")
    print("=" * 60)

    # ---- 1. 加载数据 ----
    print("\n加载数据")
    train, test, train_id, test_id = load_data(train_path, test_path)
    ntrain = train.shape[0]
    print(f"  训练集: {train.shape}, 测试集: {test.shape}")

    # ---- 2. 数据清洗 ----
    print("\n数据清洗")
    train = remove_outliers(train)
    train, y_train = target_transform(train)
    ntrain = train.shape[0]
    print(f"  清洗后训练集: {train.shape}")
    print(f"  目标 log1p: mean={y_train.mean():.4f}, std={y_train.std():.4f}")

    # ---- 3. 缺失值处理 ----
    print("\n缺失值处理 + 缺失标志")
    all_data = pd.concat([train, test], axis=0, ignore_index=True)
    all_data = handle_missing_values(all_data)

    # ---- 4. 目标编码 (在拆分 train/test 前保存原始类别列) ----
    # 注意: 目标编码需要区分 train/test 以及 y_train, 所以必须在合并前做
    print("\n目标编码")

    # 重新拆分
    train_processed = all_data.iloc[:ntrain].copy()
    test_processed = all_data.iloc[ntrain:].copy()

    if do_target_encoding:
        # 检测高基数类别
        high_card_cols = []
        for col in train_processed.columns:
            if train_processed[col].dtype == 'object':
                n_unique = train_processed[col].nunique()
                if n_unique > 6:
                    high_card_cols.append(col)
                    print(f"  高基数特征: {col} ({n_unique} 类别) → 目标编码")

        train_processed, test_processed, encoded_cols = perform_target_encoding(
            train_processed, test_processed, y_train, high_card_cols
        )
        print(f"  新增目标编码特征: {encoded_cols}")

    # ---- 5. 特征构造 ----
    print("\n特征构造")
    all_data = pd.concat([train_processed, test_processed], axis=0, ignore_index=True)
    all_data = feature_engineering(all_data)

    # ---- 6. 特征转换 ----
    print("\n特征转换 (编码 + Box-Cox + One-Hot)")
    all_data = feature_transformation(all_data)

    # 对齐 train/test
    X_train = all_data.iloc[:ntrain].values.astype(np.float64)
    X_test = all_data.iloc[ntrain:].values.astype(np.float64)
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    print(f"  特征工程后: 训练集 {X_train.shape}, 测试集 {X_test.shape}")

    # ---- 7. 特征选择 ----
    print("\n特征选择")
    feat_names = all_data.columns.tolist()
    X_train_df = pd.DataFrame(X_train, columns=feat_names)
    X_test_df = pd.DataFrame(X_test, columns=feat_names)

    if do_feature_selection:
        X_train_df, X_test_df = feature_selection(X_train_df, y_train, X_test_df)

    X_train = X_train_df.values
    X_test = X_test_df.values

    # ---- 7.5. PCA 特征拼接 ----
    if do_pca:
        print("\nPCA 特征扩充")
        X_train, X_test = add_pca_features(X_train, X_test)

    # ---- 8. 初始化所有基模型 ----
    print("\n初始化基模型")
    all_models = get_all_base_models()
    selected_names_all = list(all_models.keys())
    for name in selected_names_all:
        print(f"  + {name}")

    # ---- 8.5. Optuna 超参数优化 ----
    if do_optuna_tuning and HAS_OPTUNA:
        optimized_params = tune_all_base_models(
            X_train, y_train, selected_names_all, n_trials=n_optuna_trials
        )
        if optimized_params:
            all_models = build_optimized_models(selected_names_all, optimized_params)
    elif do_optuna_tuning and not HAS_OPTUNA:
        print("\n[警告] Optuna 未安装，使用默认超参数。安装: pip install optuna")

    # ---- 9. 单模型 CV 评估 ----
    print("\n单模型 CV 评估 (RMSLE)")
    results = evaluate_single_models(X_train, y_train, all_models)

    # ---- 10. 模型相关性分析 ----
    selected_models = all_models
    if do_correlation_analysis and len(all_models) > 2:
        print("\n模型相关性分析")
        corr_matrix, redundant_pairs = compute_model_correlations(
            all_models, X_train, y_train
        )

        # 自动移除冗余模型 (保留性能更好的)
        to_remove = set()
        if redundant_pairs:
            for m1, m2, _ in redundant_pairs:
                score1 = results.get(m1, [np.inf]).mean() if m1 in results else np.inf
                score2 = results.get(m2, [np.inf]).mean() if m2 in results else np.inf
                if score1 <= score2:
                    to_remove.add(m2)
                else:
                    to_remove.add(m1)

        if to_remove:
            print(f"  移除冗余模型: {to_remove}")
            selected_models = {k: v for k, v in all_models.items()
                               if k not in to_remove}

    print(f"\n  Stacking 基模型 ({len(selected_models)}个): "
          f"{list(selected_models.keys())}")

    # ---- 11. 选择元学习器 ----
    print(f"\n元学习器 = {meta_learner}")
    meta_map = {
        'ridge': Ridge(alpha=5.0),
        'bayesian': BayesianRidge(max_iter=300, alpha_1=1e-6, alpha_2=1e-6,
                                  lambda_1=1e-6, lambda_2=1e-6),
        'elasticnet': make_pipeline(RobustScaler(),
                                    ElasticNet(alpha=0.001, l1_ratio=0.5,
                                               random_state=RANDOM_SEED)),
        'huber': HuberRegressor(alpha=0.1, epsilon=1.35, max_iter=1000),
        'lgb': lgb.LGBMRegressor(
            num_leaves=8, max_depth=3, learning_rate=0.01,
            n_estimators=200, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, random_state=RANDOM_SEED,
            verbose=-1
        ),
    }
    meta_model = meta_map.get(meta_learner, Ridge(alpha=5.0))

    # ---- 12. 增强型 Stacking CV 评估 ----
    print("\n增强型 Stacking CV 评估")
    base_models_list = list(selected_models.values())

    stacking_cv = EnhancedStackingRegressor(
        base_models=base_models_list,
        meta_model=clone(meta_model),
        n_folds=N_FOLDS,
        use_transforms=True,
        use_diffs=True,
        use_raw_features=False,
        random_state=RANDOM_SEED,
    )

    # 用 CV 评估 Stacking
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    stacking_scores = []
    for train_idx, val_idx in kf.split(X_train):
        stacking_fold = EnhancedStackingRegressor(
            base_models=[clone(m) for m in base_models_list],
            meta_model=clone(meta_model),
            n_folds=min(N_FOLDS, 5),
            use_transforms=True,
            use_diffs=True,
            use_raw_features=False,
            random_state=RANDOM_SEED,
        )
        stacking_fold.fit(X_train[train_idx], y_train[train_idx])
        pred = stacking_fold.predict(X_train[val_idx])
        stacking_scores.append(np.sqrt(mean_squared_error(y_train[val_idx], pred)))

    stacking_scores = np.array(stacking_scores)
    print(f"  {'Stacking':15s}: RMSLE = {stacking_scores.mean():.5f} "
          f"(std={stacking_scores.std():.5f})")

    # ---- 13. 最终融合预测 ----
    print(f"\n最终融合预测 (n_seeds={n_seeds})")
    selected_names = [name for name in selected_models
                      if _get_model_factory(name) is not None]

    final_pred = final_blending_predict(
        X_train, y_train, X_test,
        selected_models_dict={name: selected_models[name] for name in selected_names},
        meta_model=meta_model,
        n_seeds=n_seeds,
    )

    # ---- 14. 逆变换 & 生成提交文件 ----
    print("\n生成提交文件")

    final_pred_original = np.expm1(final_pred)

    os.makedirs('result', exist_ok=True)

    submission = pd.DataFrame({
        "Id": test_id,
        "SalePrice": final_pred_original,
    })
    sub_path = "result/submission_improved.csv"
    submission.to_csv(sub_path, index=False)
    print(f"  [加权融合] 提交文件保存至: {sub_path}")
    print(f"  价格范围: ${submission['SalePrice'].min():,.0f} ~ "
          f"${submission['SalePrice'].max():,.0f}")
    print(f"  价格均值: ${submission['SalePrice'].mean():,.0f}")

    print("\n" + "=" * 60)
    print("Pipeline 完成!")
    print("=" * 60)

    return submission, submission_stack


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='改进版 Stacking 房价预测')
    parser.add_argument('--meta', type=str, default='ridge',
                        choices=['ridge', 'elasticnet', 'huber', 'lgb', 'bayesian'],
                        help='元学习器类型')
    parser.add_argument('--seeds', type=int, default=3,
                        help='多种子重复次数 (default: 3)')
    parser.add_argument('--no-feature-selection', action='store_true',
                        help='跳过特征选择')
    parser.add_argument('--no-target-encoding', action='store_true',
                        help='跳过目标编码')
    parser.add_argument('--no-correlation', action='store_true',
                        help='跳过模型相关性分析')
    parser.add_argument('--no-optuna', action='store_true',
                        help='跳过 Optuna 超参数优化')
    parser.add_argument('--no-pca', action='store_true',
                        help='跳过 PCA 特征拼接')
    parser.add_argument('--optuna-trials', type=int, default=50,
                        help='Optuna 试验次数 (default: 50)')
    args = parser.parse_args()

    run_pipeline(
        do_feature_selection=not args.no_feature_selection,
        do_target_encoding=not args.no_target_encoding,
        do_correlation_analysis=not args.no_correlation,
        do_optuna_tuning=not args.no_optuna,
        do_pca=not args.no_pca,
        meta_learner=args.meta,
        n_seeds=args.seeds,
        n_optuna_trials=args.optuna_trials,
    )
