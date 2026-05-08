"""
14_特征重要性分析 (Feature Importance Analysis)

功能:
分析P2（完整特征+SMOTE）与P4（SMOTE→FS）的特征重要性差异

输出:
- 特征重要性对比表
- 特征重要性对比图（中英双语）
- 被剔除特征分析
"""

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from imblearn.over_sampling import SMOTE

import plot_config
import config
from utils import (
    save_results, create_output_subdir, print_section_header, encode_labels_for_xgboost
)
from baseline import get_model
from smote_only import apply_smote
from feature_selection import filter_feature_selection, wrapper_feature_selection


def analyze_feature_importance(X_train, y_train, X_test, y_test, output_dir):
    """
    分析P2和P4的特征重要性差异
    
    步骤:
    1. 训练P2模型（完整特征+SMOTE）并提取特征重要性
    2. 训练P4模型（SMOTE→FS）并提取特征重要性
    3. 对比两组特征重要性
    """
    print_section_header(
        "特征重要性分析 (P2 vs P4)",
        "Feature Importance Analysis (P2 vs P4)"
    )
    
    fi_dir = create_output_subdir(output_dir, '14_feature_importance')
    
    model_name = 'xgboost'
    
    # ========== P2: 完整特征 + SMOTE ==========
    print("\n1. 训练P2模型 (完整特征+SMOTE)")
    print("="*60)
    
    X_resampled_p2, y_resampled_p2, _ = apply_smote(X_train, y_train)
    
    model_p2 = get_model(model_name)
    y_resampled_p2_enc, label_mapping_p2, inverse_mapping_p2 = encode_labels_for_xgboost(y_resampled_p2)
    model_p2.fit(X_resampled_p2, y_resampled_p2_enc)
    
    # 获取特征重要性
    feature_names = X_train.columns.tolist()
    importance_p2 = model_p2.feature_importances_
    
    fi_p2_df = pd.DataFrame({
        'feature': feature_names,
        'importance_p2': importance_p2,
    }).sort_values('importance_p2', ascending=False)
    
    print(f"P2模型训练完成，特征数: {len(feature_names)}")
    print(f"Top 5 特征:")
    for _, row in fi_p2_df.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance_p2']:.4f}")
    
    # ========== P4: SMOTE → FS (Wrapper, K=30) ==========
    print("\n2. 训练P4模型 (SMOTE→FS, Wrapper, K=30)")
    print("="*60)
    
    X_resampled_p4, y_resampled_p4, _ = apply_smote(X_train, y_train)
    
    # 特征选择
    X_train_p4, X_test_p4, selected_features, feature_ranks, _ = wrapper_feature_selection(
        X_resampled_p4, y_resampled_p4, X_test, k=30)
    
    model_p4 = get_model(model_name)
    y_resampled_p4_enc, label_mapping_p4, inverse_mapping_p4 = encode_labels_for_xgboost(y_resampled_p4)
    model_p4.fit(X_train_p4, y_resampled_p4_enc)
    
    # 获取特征重要性
    importance_p4 = model_p4.feature_importances_
    
    fi_p4_df = pd.DataFrame({
        'feature': selected_features,
        'importance_p4': importance_p4,
    }).sort_values('importance_p4', ascending=False)
    
    print(f"P4模型训练完成，特征数: {len(selected_features)}")
    print(f"Top 5 特征:")
    for _, row in fi_p4_df.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance_p4']:.4f}")
    
    # ========== 对比分析 ==========
    print("\n3. 特征重要性对比分析")
    print("="*60)
    
    # 合并两个DataFrame
    fi_comparison = fi_p2_df.merge(
        fi_p4_df[['feature', 'importance_p4']], 
        on='feature', 
        how='outer'
    )
    fi_comparison['importance_p4'] = fi_comparison['importance_p4'].fillna(0)
    fi_comparison['in_p4'] = fi_comparison['feature'].isin(selected_features)
    
    # 计算排名
    fi_comparison['rank_p2'] = fi_comparison['importance_p2'].rank(ascending=False)
    fi_comparison['rank_p4'] = fi_comparison['importance_p4'].rank(ascending=False)
    fi_comparison['rank_p4'] = fi_comparison['rank_p4'].fillna(999)
    
    # 保存对比表
    comparison_file = fi_dir / 'feature_importance_comparison.csv'
    fi_comparison.to_csv(comparison_file, index=False)
    print(f"特征重要性对比表已保存: {comparison_file}")
    
    # 分析被剔除的特征
    removed_features = fi_comparison[~fi_comparison['in_p4']].sort_values('importance_p2', ascending=False)
    
    print(f"\n被P4剔除的特征中，P2重要性Top 10:")
    for _, row in removed_features.head(10).iterrows():
        print(f"  {row['feature']}: P2重要性={row['importance_p2']:.4f}, P2排名={int(row['rank_p2'])}")
    
    removed_file = fi_dir / 'removed_features_analysis.csv'
    removed_features.to_csv(removed_file, index=False)
    print(f"被剔除特征分析已保存: {removed_file}")
    
    # 绘制对比图
    plot_feature_importance_comparison(fi_comparison, fi_dir, language='ch')
    plot_feature_importance_comparison(fi_comparison, fi_dir, language='en')
    
    plot_top_features_comparison(fi_p2_df, fi_p4_df, fi_dir, language='ch')
    plot_top_features_comparison(fi_p2_df, fi_p4_df, fi_dir, language='en')
    
    return fi_comparison


def plot_feature_importance_comparison(fi_comparison, output_dir, language='ch', top_n=20):
    """
    绘制P2 vs P4特征重要性对比散点图
    """
    if language == 'ch':
        labels = {
            'title': f'特征重要性对比 (P2 vs P4, Top-{top_n})',
            'xlabel': 'P2重要性 (完整特征+SMOTE)',
            'ylabel': 'P4重要性 (SMOTE→FS)',
            'in_p4': '被P4保留',
            'not_in_p4': '被P4剔除',
        }
        filename = f'fi_comparison_scatter_ch.png'
    else:
        labels = {
            'title': f'Feature Importance Comparison (P2 vs P4, Top-{top_n})',
            'xlabel': 'P2 Importance (Full Features + SMOTE)',
            'ylabel': 'P4 Importance (SMOTE→FS)',
            'in_p4': 'Retained in P4',
            'not_in_p4': 'Removed in P4',
        }
        filename = f'fi_comparison_scatter_en.png'
    
    # 取P2中Top-N特征
    top_features = fi_comparison.nlargest(top_n, 'importance_p2')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 分离保留和剔除的特征
    retained = top_features[top_features['in_p4']]
    removed = top_features[~top_features['in_p4']]
    
    # 绘制散点
    ax.scatter(retained['importance_p2'], retained['importance_p4'], 
              c='green', alpha=0.6, s=60, label=labels['in_p4'])
    ax.scatter(removed['importance_p2'], removed['importance_p4'], 
              c='red', alpha=0.6, s=60, label=labels['not_in_p4'])
    
    # 添加对角线（y=x）
    max_val = max(top_features['importance_p2'].max(), top_features['importance_p4'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1)
    
    ax.set_xlabel(labels['xlabel'], fontsize=12)
    ax.set_ylabel(labels['ylabel'], fontsize=12)
    ax.set_title(labels['title'], fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / filename
    plt.savefig(output_path, dpi=config.VISUALIZATION['dpi'],
                bbox_inches='tight', format=config.VISUALIZATION['format'])
    print(f"特征重要性对比图已保存: {output_path}")
    
    plt.close()


def plot_top_features_comparison(fi_p2_df, fi_p4_df, output_dir, language='ch', top_n=15):
    """
    绘制Top特征重要性对比柱状图
    """
    if language == 'ch':
        labels = {
            'title': f'Top-{top_n} 特征重要性对比',
            'xlabel': '重要性',
            'ylabel': '特征',
        }
        filename = f'top_features_comparison_ch.png'
    else:
        labels = {
            'title': f'Top-{top_n} Feature Importance Comparison',
            'xlabel': 'Importance',
            'ylabel': 'Feature',
        }
        filename = f'top_features_comparison_en.png'
    
    # 取P2和P4的Top-N
    p2_top = fi_p2_df.head(top_n)
    p4_top = fi_p4_df.head(top_n)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # P2
    y_pos = np.arange(len(p2_top))
    ax1.barh(y_pos, p2_top['importance_p2'].values, color='steelblue', alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(p2_top['feature'].values, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel(labels['xlabel'], fontsize=12)
    ax1.set_ylabel(labels['ylabel'], fontsize=12)
    ax1.set_title('P2 (完整特征+SMOTE)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # P4
    y_pos = np.arange(len(p4_top))
    ax2.barh(y_pos, p4_top['importance_p4'].values, color='coral', alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(p4_top['feature'].values, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel(labels['xlabel'], fontsize=12)
    ax2.set_ylabel(labels['ylabel'], fontsize=12)
    ax2.set_title('P4 (SMOTE→FS)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / filename
    plt.savefig(output_path, dpi=config.VISUALIZATION['dpi'],
                bbox_inches='tight', format=config.VISUALIZATION['format'])
    print(f"Top特征对比图已保存: {output_path}")
    
    plt.close()


def main():
    """主函数"""
    print_section_header(
        "阶段14：特征重要性分析",
        "Stage 14: Feature Importance Analysis"
    )
    
    # 创建输出目录
    output_dir = config.create_output_dir()
    print(f"输出目录: {output_dir}")
    
    # 加载数据
    print("\n加载数据...")
    from utils import load_nsl_kdd_data, map_attack_categories, preprocess_features
    
    train_df, test_df = load_nsl_kdd_data()
    
    # 调试模式采样
    if config.DEBUG_MODE:
        print(f"[调试模式] 采样 {config.DEBUG_TRAIN_SIZE} 训练样本")
        train_df = train_df.sample(n=min(config.DEBUG_TRAIN_SIZE, len(train_df)),
                                   random_state=config.RANDOM_STATE)
        test_df = test_df.sample(n=min(config.DEBUG_TEST_SIZE, len(test_df)),
                                random_state=config.RANDOM_STATE)
    
    # 预处理
    train_df = map_attack_categories(train_df)
    test_df = map_attack_categories(test_df)
    
    X_train, y_train = preprocess_features(train_df)
    X_test, y_test = preprocess_features(test_df)
    
    # 对齐特征
    common_features = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_features]
    X_test = X_test[common_features]
    
    print(f"训练集: {X_train.shape}")
    print(f"测试集: {X_test.shape}")
    
    # 分析特征重要性
    fi_comparison = analyze_feature_importance(X_train, y_train, X_test, y_test, output_dir)
    
    print_section_header(
        "特征重要性分析完成",
        "Feature Importance Analysis Completed"
    )
    
    return fi_comparison


if __name__ == "__main__":
    main()
