"""
13_计算开销分析 (Computational Cost Analysis)

功能:
汇总所有实验的时间数据，分析计算开销

分析维度:
1. 训练时间对比
2. 预处理时间（SMOTE、特征选择）
3. 效率-性能权衡

输出:
- 时间对比表
- 训练时间对比柱状图（中英双语）
- 效率-性能散点图（中英双语）
"""

import sys
from pathlib import Path
import time
import json

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plot_config
import config
from utils import (
    save_results, create_output_subdir, print_section_header
)


def collect_time_data(output_base_dir=None):
    """
    从已有实验结果中收集时间数据
    
    参数:
        output_base_dir: 输出根目录，默认使用config中的OUTPUT_DIR
    
    返回:
        time_data: 时间数据列表
    """
    if output_base_dir is None:
        output_base_dir = config.OUTPUT_DIR
    
    print(f"\n收集时间数据...")
    print(f"扫描目录: {output_base_dir}")
    
    time_data = []
    
    # 扫描所有实验输出目录
    output_path = Path(output_base_dir)
    if not output_path.exists():
        print(f"警告: 输出目录不存在: {output_path}")
        return time_data
    
    # 遍历所有时间戳子目录
    for timestamp_dir in sorted(output_path.iterdir()):
        if not timestamp_dir.is_dir():
            continue
        
        # 遍历所有实验子目录
        for exp_dir in timestamp_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            # 查找results.csv文件
            results_file = exp_dir / 'results.csv'
            if not results_file.exists():
                continue
            
            try:
                df = pd.read_csv(results_file)
                
                for _, row in df.iterrows():
                    # 提取时间相关字段
                    time_record = {
                        'timestamp_dir': timestamp_dir.name,
                        'exp_dir': exp_dir.name,
                    }
                    
                    # 提取流水线信息
                    if 'pipeline' in row:
                        time_record['pipeline'] = row['pipeline']
                    if 'model' in row:
                        time_record['model'] = row['model']
                    if 'fs_method' in row:
                        time_record['fs_method'] = row['fs_method']
                    if 'fs_k' in row:
                        time_record['fs_k'] = row['fs_k']
                    
                    # 提取时间字段
                    time_fields = ['train_time', 'predict_time', 'smote_time', 
                                  'fs_time', 'total_time']
                    for field in time_fields:
                        if field in row and not pd.isna(row[field]):
                            time_record[field] = row[field]
                    
                    # 提取性能字段
                    if 'f1_macro' in row:
                        time_record['f1_macro'] = row['f1_macro']
                    if 'accuracy' in row:
                        time_record['accuracy'] = row['accuracy']
                    
                    time_data.append(time_record)
                    
            except Exception as e:
                print(f"  读取失败 {results_file}: {e}")
                continue
    
    print(f"收集到 {len(time_data)} 条时间记录")
    
    return time_data


def analyze_time_cost(time_data, output_dir):
    """
    分析计算开销
    
    参数:
        time_data: 时间数据列表
        output_dir: 输出目录
    
    返回:
        analysis_df: 分析结果DataFrame
    """
    print_section_header(
        "计算开销分析",
        "Computational Cost Analysis"
    )
    
    if len(time_data) == 0:
        print("警告: 没有时间数据可分析")
        return None
    
    cost_dir = create_output_subdir(output_dir, '13_computational_cost')
    
    # 转换为DataFrame
    df = pd.DataFrame(time_data)
    
    # 保存原始数据
    raw_file = cost_dir / 'raw_time_data.csv'
    df.to_csv(raw_file, index=False)
    print(f"原始时间数据已保存: {raw_file}")
    
    # 1. 按流水线汇总
    print("\n1. 按流水线汇总时间开销")
    print("="*60)
    
    pipeline_summary = []
    
    # 定义流水线分组
    pipeline_groups = {
        'P1_基线': df[df['pipeline'].str.startswith('P1', na=False)],
        'P2_仅SMOTE': df[df['pipeline'].str.startswith('P2', na=False)],
        'P3_仅FS': df[df['pipeline'].str.startswith('P3', na=False)],
        'P4_SMOTE→FS': df[df['pipeline'].str.startswith('P4', na=False)],
        'P5_FS→SMOTE': df[df['pipeline'].str.startswith('P5', na=False)],
    }
    
    for pipeline_name, group_df in pipeline_groups.items():
        if len(group_df) == 0:
            continue
        
        summary = {
            'pipeline': pipeline_name,
            'n_experiments': len(group_df),
        }
        
        # 训练时间
        if 'train_time' in group_df.columns:
            summary['train_time_mean'] = group_df['train_time'].mean()
            summary['train_time_std'] = group_df['train_time'].std()
        
        # SMOTE时间
        if 'smote_time' in group_df.columns:
            smote_times = group_df['smote_time'].dropna()
            if len(smote_times) > 0:
                summary['smote_time_mean'] = smote_times.mean()
                summary['smote_time_std'] = smote_times.std()
        
        # FS时间
        if 'fs_time' in group_df.columns:
            fs_times = group_df['fs_time'].dropna()
            if len(fs_times) > 0:
                summary['fs_time_mean'] = fs_times.mean()
                summary['fs_time_std'] = fs_times.std()
        
        # 总时间
        if 'total_time' in group_df.columns:
            summary['total_time_mean'] = group_df['total_time'].mean()
            summary['total_time_std'] = group_df['total_time'].std()
        
        # 性能
        if 'f1_macro' in group_df.columns:
            summary['f1_macro_mean'] = group_df['f1_macro'].mean()
            summary['f1_macro_std'] = group_df['f1_macro'].std()
        
        pipeline_summary.append(summary)
        
        print(f"\n{pipeline_name}:")
        print(f"  实验数: {summary['n_experiments']}")
        if 'train_time_mean' in summary:
            print(f"  训练时间: {summary['train_time_mean']:.2f}±{summary.get('train_time_std', 0):.2f}s")
        if 'smote_time_mean' in summary:
            print(f"  SMOTE时间: {summary['smote_time_mean']:.2f}±{summary.get('smote_time_std', 0):.2f}s")
        if 'fs_time_mean' in summary:
            print(f"  FS时间: {summary['fs_time_mean']:.2f}±{summary.get('fs_time_std', 0):.2f}s")
        if 'total_time_mean' in summary:
            print(f"  总时间: {summary['total_time_mean']:.2f}±{summary.get('total_time_std', 0):.2f}s")
        if 'f1_macro_mean' in summary:
            print(f"  F1-macro: {summary['f1_macro_mean']:.4f}±{summary.get('f1_macro_std', 0):.4f}")
    
    # 保存汇总
    summary_df = pd.DataFrame(pipeline_summary)
    summary_file = cost_dir / 'pipeline_time_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"\n流水线时间汇总已保存: {summary_file}")
    
    # 2. 按模型汇总
    print("\n2. 按模型汇总训练时间")
    print("="*60)
    
    if 'model' in df.columns and 'train_time' in df.columns:
        model_summary = df.groupby('model')['train_time'].agg(['mean', 'std', 'count'])
        print(model_summary)
        
        model_file = cost_dir / 'model_time_summary.csv'
        model_summary.to_csv(model_file)
        print(f"模型时间汇总已保存: {model_file}")
    
    # 3. 特征选择方法对比
    print("\n3. Filter vs Wrapper 时间对比")
    print("="*60)
    
    if 'fs_method' in df.columns and 'fs_time' in df.columns:
        fs_summary = df[df['fs_method'].notna()].groupby('fs_method')['fs_time'].agg(['mean', 'std', 'count'])
        print(fs_summary)
        
        fs_file = cost_dir / 'fs_method_time_summary.csv'
        fs_summary.to_csv(fs_file)
        print(f"FS方法时间汇总已保存: {fs_file}")
    
    # 绘制图表
    plot_time_comparison(summary_df, cost_dir, language='ch')
    plot_time_comparison(summary_df, cost_dir, language='en')
    
    plot_efficiency_performance(df, cost_dir, language='ch')
    plot_efficiency_performance(df, cost_dir, language='en')
    
    return summary_df


def plot_time_comparison(summary_df, output_dir, language='ch'):
    """
    绘制训练时间对比柱状图
    """
    if language == 'ch':
        labels = {
            'title': '各流水线训练时间对比',
            'xlabel': '实验流水线',
            'ylabel': '时间 (秒)',
            'train_time': '训练时间',
            'smote_time': 'SMOTE时间',
            'fs_time': '特征选择时间',
        }
        filename = 'time_comparison_ch.png'
    else:
        labels = {
            'title': 'Pipeline Training Time Comparison',
            'xlabel': 'Experiment Pipeline',
            'ylabel': 'Time (seconds)',
            'train_time': 'Training Time',
            'smote_time': 'SMOTE Time',
            'fs_time': 'Feature Selection Time',
        }
        filename = 'time_comparison_en.png'
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pipelines = summary_df['pipeline'].values
    x = np.arange(len(pipelines))
    
    # 收集时间数据
    time_data = {}
    for col, label in [('train_time_mean', labels['train_time']), 
                        ('smote_time_mean', labels['smote_time']),
                        ('fs_time_mean', labels['fs_time'])]:
        if col in summary_df.columns:
            values = summary_df[col].fillna(0).values
            if np.sum(values) > 0:
                time_data[label] = values
    
    # 绘制堆叠柱状图
    bottom = np.zeros(len(pipelines))
    colors = ['steelblue', 'coral', 'green']
    
    for i, (label, values) in enumerate(time_data.items()):
        ax.bar(x, values, bottom=bottom, label=label, 
               color=colors[i % len(colors)], alpha=0.8)
        bottom += values
    
    ax.set_xlabel(labels['xlabel'], fontsize=12)
    ax.set_ylabel(labels['ylabel'], fontsize=12)
    ax.set_title(labels['title'], fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pipelines, rotation=30, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / filename
    plt.savefig(output_path, dpi=config.VISUALIZATION['dpi'],
                bbox_inches='tight', format=config.VISUALIZATION['format'])
    print(f"时间对比图已保存: {output_path}")
    
    plt.close()


def plot_efficiency_performance(df, output_dir, language='ch'):
    """
    绘制效率-性能权衡散点图
    """
    if language == 'ch':
        labels = {
            'title': '效率-性能权衡分析',
            'xlabel': '总时间 (秒)',
            'ylabel': 'F1-macro',
        }
        filename = 'efficiency_performance_ch.png'
    else:
        labels = {
            'title': 'Efficiency-Performance Trade-off',
            'xlabel': 'Total Time (seconds)',
            'ylabel': 'F1-macro',
        }
        filename = 'efficiency_performance_en.png'
    
    # 需要total_time和f1_macro
    if 'total_time' not in df.columns or 'f1_macro' not in df.columns:
        print("警告: 缺少total_time或f1_macro数据，跳过效率-性能图")
        return
    
    plot_df = df[['pipeline', 'total_time', 'f1_macro']].dropna()
    
    if len(plot_df) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 按流水线分组绘制
    pipelines = plot_df['pipeline'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(pipelines)))
    
    for i, pipeline in enumerate(pipelines):
        group = plot_df[plot_df['pipeline'] == pipeline]
        ax.scatter(group['total_time'], group['f1_macro'], 
                  label=pipeline, color=colors[i], alpha=0.6, s=50)
    
    ax.set_xlabel(labels['xlabel'], fontsize=12)
    ax.set_ylabel(labels['ylabel'], fontsize=12)
    ax.set_title(labels['title'], fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / filename
    plt.savefig(output_path, dpi=config.VISUALIZATION['dpi'],
                bbox_inches='tight', format=config.VISUALIZATION['format'])
    print(f"效率-性能图已保存: {output_path}")
    
    plt.close()


def main():
    """主函数"""
    print_section_header(
        "阶段13：计算开销分析",
        "Stage 13: Computational Cost Analysis"
    )
    
    # 创建输出目录
    output_dir = config.create_output_dir()
    print(f"输出目录: {output_dir}")
    
    # 收集时间数据
    time_data = collect_time_data()
    
    if len(time_data) > 0:
        # 分析计算开销
        summary_df = analyze_time_cost(time_data, output_dir)
    else:
        print("\n警告: 没有找到历史实验数据")
        print("请先运行其他实验脚本，或指定正确的输出目录")
    
    print_section_header(
        "计算开销分析完成",
        "Computational Cost Analysis Completed"
    )
    
    return time_data


if __name__ == "__main__":
    main()
