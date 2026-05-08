"""
面向类别不平衡入侵检测的SMOTE与特征选择联合流水线：可视化模块

功能：
1. 读取所有实验结果并汇总
2. 生成对比图表（柱状图、热力图、折线图等）
3. 生成论文所需的高质量图表
4. 所有图表使用中文标签

使用方法：
python 07_visualization.py
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "code"))

from config import OUTPUT_DIR, VISUALIZATION, PAPER_FIGURES, CLASS_LABELS
from plot_config import setup_chinese_font
from logger import get_logger

# 初始化日志
logger = get_logger("Visualization")

# 设置中文字体
setup_chinese_font()

# 设置seaborn样式
plt.style.use(VISUALIZATION['style'])
sns.set_palette(PAPER_FIGURES['color_palette'])


class ExperimentVisualizer:
    """实验结果可视化器"""
    
    def __init__(self, output_dir=None):
        """
        初始化可视化器
        
        Args:
            output_dir: 实验结果目录，如果为None则自动查找最新的
        """
        if output_dir is None:
            # 自动查找最新的输出目录（只查找目录，排除文件）
            output_dirs = sorted([d for d in OUTPUT_DIR.glob("*") if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
            if not output_dirs:
                raise ValueError("未找到实验结果目录，请先运行实验！")
            self.output_dir = output_dirs[0]
        else:
            self.output_dir = Path(output_dir)
        
        # 创建图表输出目录
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        
        logger.info(f"加载实验结果目录: {self.output_dir}")
        
        # 加载所有结果
        self.results_df = self._load_all_results()
        
    def _load_all_results(self):
        """加载所有实验结果"""
        logger.info("正在加载实验结果...")
        
        all_results = []
        
        # 遍历所有子目录
        for subdir in self.output_dir.iterdir():
            if subdir.is_dir():
                results_file = subdir / "results.csv"
                if results_file.exists():
                    try:
                        df = pd.read_csv(results_file)
                        # 添加实验标识
                        df['experiment'] = subdir.name
                        all_results.append(df)
                    except Exception as e:
                        logger.warning(f"读取 {results_file} 失败: {e}")
        
        if not all_results:
            raise ValueError("未找到任何实验结果！")
        
        combined = pd.concat(all_results, ignore_index=True)
        logger.info(f"成功加载 {len(all_results)} 个实验结果，共 {len(combined)} 条记录")
        
        return combined
    
    def plot_pipeline_comparison(self, metric='f1_macro', save=True):
        """
        绘制不同流水线的性能对比图
        
        Args:
            metric: 评估指标
            save: 是否保存图片
        """
        logger.info(f"绘制流水线对比图（指标: {metric}）...")
        
        # 准备数据
        df = self.results_df.copy()
        
        # 创建流水线标签
        def get_pipeline_label(row):
            if row['pipeline'].startswith('P1'):
                return 'P1: 基线'
            elif row['pipeline'].startswith('P2'):
                return 'P2: 仅SMOTE'
            elif row['pipeline'].startswith('P3'):
                return 'P3: 仅特征选择'
            elif row['pipeline'].startswith('P4'):
                return 'P4: SMOTE→FS'
            elif row['pipeline'].startswith('P5'):
                return 'P5: FS→SMOTE'
            return row['pipeline']
        
        df['pipeline_label'] = df.apply(get_pipeline_label, axis=1)
        
        # 按流水线和模型分组
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        models = ['decision_tree', 'random_forest', 'xgboost']
        model_names = ['决策树', '随机森林', 'XGBoost']
        
        for idx, (model, model_name) in enumerate(zip(models, model_names)):
            ax = axes[idx]
            model_data = df[df['model'] == model]
            
            if model_data.empty:
                ax.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(model_name)
                continue
            
            # 按流水线分组计算均值
            pipeline_means = model_data.groupby('pipeline_label')[metric].agg(['mean', 'std']).reset_index()
            
            # 排序保持流水线顺序
            order = ['P1: 基线', 'P2: 仅SMOTE', 'P3: 仅特征选择', 'P4: SMOTE→FS', 'P5: FS→SMOTE']
            pipeline_means['pipeline_label'] = pd.Categorical(pipeline_means['pipeline_label'], categories=order, ordered=True)
            pipeline_means = pipeline_means.sort_values('pipeline_label')
            
            # 绘制柱状图
            bars = ax.bar(range(len(pipeline_means)), pipeline_means['mean'], 
                         yerr=pipeline_means['std'], capsize=5, alpha=0.8)
            
            # 设置颜色
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_xticks(range(len(pipeline_means)))
            ax.set_xticklabels(pipeline_means['pipeline_label'], rotation=45, ha='right')
            ax.set_ylabel(metric.replace('_', ' ').upper())
            ax.set_title(f'{model_name} - 流水线对比', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # 添加数值标签
            for i, (mean_val, std_val) in enumerate(zip(pipeline_means['mean'], pipeline_means['std'])):
                ax.text(i, mean_val + std_val + 0.01, f'{mean_val:.4f}', 
                       ha='center', va='bottom', fontsize=9)
        
        plt.suptitle(f'不同流水线性能对比 ({metric})', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            output_path = self.figures_dir / f'pipeline_comparison_{metric}.png'
            plt.savefig(output_path, dpi=VISUALIZATION['dpi'], bbox_inches='tight')
            logger.info(f"图表已保存: {output_path}")
        
        return fig
    
    def plot_fs_method_comparison(self, metric='f1_macro', save=True):
        """
        绘制特征选择方法对比图
        
        Args:
            metric: 评估指标
            save: 是否保存图片
        """
        logger.info(f"绘制特征选择方法对比图（指标: {metric}）...")
        
        # 筛选包含特征选择的结果
        df = self.results_df[self.results_df['fs_method'].notna()].copy()
        
        if df.empty:
            logger.warning("未找到特征选择相关结果")
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        fs_methods = ['filter', 'wrapper']
        fs_names = ['Filter方法', 'Wrapper方法']
        models = ['decision_tree', 'random_forest', 'xgboost']
        model_names = ['决策树', '随机森林', 'XGBoost']
        
        for fs_idx, (fs_method, fs_name) in enumerate(zip(fs_methods, fs_names)):
            for model_idx, (model, model_name) in enumerate(zip(models, model_names)):
                ax = axes[fs_idx, model_idx]
                
                # 筛选数据
                data = df[(df['fs_method'] == fs_method) & (df['model'] == model)]
                
                if data.empty:
                    ax.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{fs_name} - {model_name}')
                    continue
                
                # 按特征数量和流水线分组
                pivot_data = data.pivot_table(values=metric, index='fs_k', columns='pipeline', aggfunc='mean')
                
                # 绘制折线图
                for pipeline in pivot_data.columns:
                    ax.plot(pivot_data.index, pivot_data[pipeline], marker='o', label=pipeline, linewidth=2)
                
                ax.set_xlabel('特征数量', fontsize=12)
                ax.set_ylabel(metric.replace('_', ' ').upper(), fontsize=12)
                ax.set_title(f'{fs_name} - {model_name}', fontsize=13, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.set_xticks([10, 20, 30])
        
        plt.suptitle(f'特征选择方法对比 ({metric})', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            output_path = self.figures_dir / f'fs_method_comparison_{metric}.png'
            plt.savefig(output_path, dpi=VISUALIZATION['dpi'], bbox_inches='tight')
            logger.info(f"图表已保存: {output_path}")
        
        return fig
    
    def plot_heatmap(self, metric='f1_macro', save=True):
        """
        绘制热力图：展示不同配置组合的性能
        
        Args:
            metric: 评估指标
            save: 是否保存图片
        """
        logger.info(f"绘制热力图（指标: {metric}）...")
        
        df = self.results_df.copy()
        
        # 创建透视表
        # 行：模型+特征选择方法，列：流水线+特征数量
        df['config'] = df['model'].astype(str) + '_' + df['fs_method'].fillna('none')
        df['pipeline_fs'] = df['pipeline'].astype(str) + '_' + df['fs_k'].fillna(0).astype(int).astype(str)
        
        pivot = df.pivot_table(values=metric, index='config', columns='pipeline_fs', aggfunc='mean')
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 绘制热力图
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd', 
                   cbar_kws={'label': metric.replace('_', ' ').upper()}, 
                   ax=ax, linewidths=0.5)
        
        ax.set_title(f'实验结果热力图 ({metric})', fontsize=16, fontweight='bold')
        ax.set_xlabel('流水线配置', fontsize=12)
        ax.set_ylabel('模型配置', fontsize=12)
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save:
            output_path = self.figures_dir / f'heatmap_{metric}.png'
            plt.savefig(output_path, dpi=VISUALIZATION['dpi'], bbox_inches='tight')
            logger.info(f"图表已保存: {output_path}")
        
        return fig
    
    def plot_time_comparison(self, save=True):
        """
        绘制时间对比图
        
        Args:
            save: 是否保存图片
        """
        logger.info("绘制时间对比图...")
        
        df = self.results_df.copy()
        
        # 检查是否有时间数据
        if 'train_time' not in df.columns:
            logger.warning("结果中未找到时间数据")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. 训练时间对比
        ax1 = axes[0]
        time_data = df.groupby('pipeline')['train_time'].mean().sort_values()
        
        bars = ax1.barh(range(len(time_data)), time_data.values, alpha=0.8)
        ax1.set_yticks(range(len(time_data)))
        ax1.set_yticklabels(time_data.index)
        ax1.set_xlabel('训练时间 (秒)', fontsize=12)
        ax1.set_title('各流水线平均训练时间', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # 添加数值标签
        for i, v in enumerate(time_data.values):
            ax1.text(v + 0.1, i, f'{v:.2f}s', va='center', fontsize=9)
        
        # 2. 特征选择时间（如果有）
        ax2 = axes[1]
        if 'fs_time' in df.columns:
            fs_data = df[df['fs_time'].notna()].groupby('pipeline')['fs_time'].mean().sort_values()
            
            if not fs_data.empty:
                bars = ax2.barh(range(len(fs_data)), fs_data.values, alpha=0.8, color='orange')
                ax2.set_yticks(range(len(fs_data)))
                ax2.set_yticklabels(fs_data.index)
                ax2.set_xlabel('特征选择时间 (秒)', fontsize=12)
                ax2.set_title('各流水线特征选择时间', fontsize=14, fontweight='bold')
                ax2.grid(axis='x', alpha=0.3)
                
                for i, v in enumerate(fs_data.values):
                    ax2.text(v + 0.1, i, f'{v:.2f}s', va='center', fontsize=9)
            else:
                ax2.text(0.5, 0.5, '无特征选择时间数据', ha='center', va='center', transform=ax2.transAxes)
        else:
            ax2.text(0.5, 0.5, '无特征选择时间数据', ha='center', va='center', transform=ax2.transAxes)
        
        plt.suptitle('实验时间对比', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save:
            output_path = self.figures_dir / 'time_comparison.png'
            plt.savefig(output_path, dpi=VISUALIZATION['dpi'], bbox_inches='tight')
            logger.info(f"图表已保存: {output_path}")
        
        return fig
    
    def plot_per_class_performance(self, pipeline='P5', model='decision_tree', save=True):
        """
        绘制每类性能对比图
        
        Args:
            pipeline: 流水线名称
            model: 模型名称
            save: 是否保存图片
        """
        logger.info(f"绘制每类性能对比图（{pipeline}, {model}）...")
        
        df = self.results_df.copy()
        
        # 筛选数据
        pipeline_data = df[df['pipeline'].str.startswith(pipeline) & (df['model'] == model)]
        
        if pipeline_data.empty:
            logger.warning(f"未找到 {pipeline} + {model} 的数据")
            return None
        
        # 获取每类的F1分数
        classes = ['dos', 'normal', 'probe', 'r2l', 'u2r']
        class_labels = ['DoS攻击', '正常流量', 'Probe攻击', 'R2L攻击', 'U2R攻击']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(classes))
        width = 0.15
        
        # 获取不同的特征选择配置
        if 'fs_k' in pipeline_data.columns:
            configs = pipeline_data['fs_k'].dropna().unique()
            configs = sorted(configs)
        else:
            configs = [0]
        
        for i, config in enumerate(configs):
            if config == 0:
                data = pipeline_data[pipeline_data['fs_k'].isna()]
                label = '无特征选择'
            else:
                data = pipeline_data[pipeline_data['fs_k'] == config]
                label = f'Top {int(config)} 特征'
            
            if data.empty:
                continue
            
            f1_scores = []
            for cls in classes:
                col = f'f1_{cls}'
                if col in data.columns:
                    f1_scores.append(data[col].mean())
                else:
                    f1_scores.append(0)
            
            offset = (i - len(configs)/2) * width
            ax.bar(x + offset, f1_scores, width, label=label, alpha=0.8)
        
        ax.set_xlabel('攻击类别', fontsize=12)
        ax.set_ylabel('F1分数', fontsize=12)
        ax.set_title(f'{pipeline} - {model} 每类性能对比', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_labels)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            output_path = self.figures_dir / f'per_class_performance_{pipeline}_{model}.png'
            plt.savefig(output_path, dpi=VISUALIZATION['dpi'], bbox_inches='tight')
            logger.info(f"图表已保存: {output_path}")
        
        return fig
    
    def generate_all_figures(self):
        """生成所有图表"""
        logger.info("=" * 60)
        logger.info("开始生成所有可视化图表...")
        logger.info("=" * 60)
        
        figures = []
        
        # 1. 流水线对比图
        try:
            fig = self.plot_pipeline_comparison(metric='f1_macro')
            figures.append(fig)
            plt.close(fig)
        except Exception as e:
            logger.error(f"生成流水线对比图失败: {e}")
        
        # 2. 特征选择方法对比
        try:
            fig = self.plot_fs_method_comparison(metric='f1_macro')
            if fig:
                figures.append(fig)
                plt.close(fig)
        except Exception as e:
            logger.error(f"生成特征选择对比图失败: {e}")
        
        # 3. 热力图
        try:
            fig = self.plot_heatmap(metric='f1_macro')
            figures.append(fig)
            plt.close(fig)
        except Exception as e:
            logger.error(f"生成热力图失败: {e}")
        
        # 4. 时间对比
        try:
            fig = self.plot_time_comparison()
            if fig:
                figures.append(fig)
                plt.close(fig)
        except Exception as e:
            logger.error(f"生成时间对比图失败: {e}")
        
        # 5. 每类性能（核心实验P5）
        try:
            for model in ['decision_tree', 'random_forest', 'xgboost']:
                fig = self.plot_per_class_performance(pipeline='P5', model=model)
                if fig:
                    figures.append(fig)
                    plt.close(fig)
        except Exception as e:
            logger.error(f"生成每类性能图失败: {e}")
        
        logger.info("=" * 60)
        logger.info(f"图表生成完成！共生成 {len(figures)} 张图表")
        logger.info(f"图表保存位置: {self.figures_dir}")
        logger.info("=" * 60)
        
        return figures


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("实验结果可视化")
    logger.info("=" * 60)
    
    try:
        # 创建可视化器
        visualizer = ExperimentVisualizer()
        
        # 生成所有图表
        visualizer.generate_all_figures()
        
        logger.info("可视化完成！")
        
    except Exception as e:
        logger.error(f"可视化过程出错: {e}")
        raise


if __name__ == "__main__":
    main()
