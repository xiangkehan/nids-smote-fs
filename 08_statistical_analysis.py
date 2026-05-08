"""
面向类别不平衡入侵检测的SMOTE与特征选择联合流水线：统计检验模块

功能：
1. 加载所有实验结果
2. 进行统计显著性检验（配对t检验、Wilcoxon符号秩检验）
3. 比较不同流水线的性能差异
4. 生成统计报告和表格

使用方法：
python 08_statistical_analysis.py
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import warnings

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "code"))

from config import OUTPUT_DIR, CLASS_LABELS
from logger import get_logger

# 初始化日志
logger = get_logger("StatisticalAnalysis")


class StatisticalAnalyzer:
    """统计检验分析器"""
    
    def __init__(self, output_dir=None):
        """
        初始化分析器
        
        Args:
            output_dir: 实验结果目录，如果为None则自动查找最新的
        """
        if output_dir is None:
            # 自动查找最新的输出目录（只查找目录）
            output_dirs = sorted([d for d in OUTPUT_DIR.glob("*") if d.is_dir()], 
                                key=lambda x: x.stat().st_mtime, reverse=True)
            if not output_dirs:
                raise ValueError("未找到实验结果目录，请先运行实验！")
            self.output_dir = output_dirs[0]
        else:
            self.output_dir = Path(output_dir)
        
        # 创建统计结果输出目录
        self.stats_dir = self.output_dir / "statistics"
        self.stats_dir.mkdir(exist_ok=True)
        
        logger.info(f"加载实验结果目录: {self.output_dir}")
        
        # 加载所有结果
        self.results_df = self._load_all_results()
        
    def _load_all_results(self):
        """加载所有实验结果"""
        logger.info("正在加载实验结果...")
        
        all_results = []
        
        # 遍历所有子目录
        for subdir in self.output_dir.iterdir():
            if subdir.is_dir() and subdir.name != 'figures' and subdir.name != 'statistics':
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
    
    def paired_t_test(self, group1_data, group2_data, metric='f1_macro'):
        """
        配对t检验
        
        Args:
            group1_data: 第一组数据
            group2_data: 第二组数据
            metric: 评估指标
            
        Returns:
            dict: 检验结果
        """
        # 确保两组数据长度相同
        min_len = min(len(group1_data), len(group2_data))
        g1 = group1_data[metric].values[:min_len]
        g2 = group2_data[metric].values[:min_len]
        
        # 配对t检验
        t_stat, p_value = stats.ttest_rel(g1, g2)
        
        # 计算效应量（Cohen's d）
        mean_diff = np.mean(g1 - g2)
        std_diff = np.std(g1 - g2, ddof=1)
        cohens_d = mean_diff / std_diff if std_diff != 0 else 0
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'mean_diff': mean_diff,
            'significant': p_value < 0.05,
            'n_samples': min_len
        }
    
    def wilcoxon_test(self, group1_data, group2_data, metric='f1_macro'):
        """
        Wilcoxon符号秩检验（非参数检验）
        
        Args:
            group1_data: 第一组数据
            group2_data: 第二组数据
            metric: 评估指标
            
        Returns:
            dict: 检验结果
        """
        # 确保两组数据长度相同
        min_len = min(len(group1_data), len(group2_data))
        g1 = group1_data[metric].values[:min_len]
        g2 = group2_data[metric].values[:min_len]
        
        # Wilcoxon检验
        try:
            w_stat, p_value = stats.wilcoxon(g1, g2)
        except ValueError:
            # 如果所有差异为0，则无法计算
            w_stat, p_value = 0, 1.0
        
        return {
            'w_statistic': w_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'n_samples': min_len
        }
    
    def compare_pipelines(self, pipeline1, pipeline2, metric='f1_macro'):
        """
        比较两个流水线的性能
        
        Args:
            pipeline1: 第一个流水线名称
            pipeline2: 第二个流水线名称
            metric: 评估指标
            
        Returns:
            dict: 比较结果
        """
        # 筛选数据
        df = self.results_df.copy()
        
        # 根据流水线名称筛选
        p1_data = df[df['pipeline'].str.startswith(pipeline1)]
        p2_data = df[df['pipeline'].str.startswith(pipeline2)]
        
        if p1_data.empty or p2_data.empty:
            logger.warning(f"流水线 {pipeline1} 或 {pipeline2} 无数据")
            return None
        
        # 按模型分组进行比较
        models = ['decision_tree', 'random_forest', 'xgboost']
        results = {}
        
        for model in models:
            m1_data = p1_data[p1_data['model'] == model]
            m2_data = p2_data[p2_data['model'] == model]
            
            if m1_data.empty or m2_data.empty:
                continue
            
            # 进行统计检验
            t_test_result = self.paired_t_test(m1_data, m2_data, metric)
            wilcoxon_result = self.wilcoxon_test(m1_data, m2_data, metric)
            
            results[model] = {
                'pipeline1_mean': m1_data[metric].mean(),
                'pipeline2_mean': m2_data[metric].mean(),
                'pipeline1_std': m1_data[metric].std(),
                'pipeline2_std': m2_data[metric].std(),
                't_test': t_test_result,
                'wilcoxon': wilcoxon_result,
                'improvement': ((m2_data[metric].mean() - m1_data[metric].mean()) / 
                               m1_data[metric].mean() * 100) if m1_data[metric].mean() != 0 else 0
            }
        
        return results
    
    def analyze_pipeline_order_effect(self, metric='f1_macro'):
        """
        分析处理顺序效应（核心分析）
        
        比较P4 (SMOTE→FS) 和 P5 (FS→SMOTE)
        
        Args:
            metric: 评估指标
            
        Returns:
            dict: 分析结果
        """
        logger.info("=" * 60)
        logger.info("分析处理顺序效应 | Analyzing Processing Order Effect")
        logger.info("=" * 60)
        
        results = self.compare_pipelines('P4', 'P5', metric)
        
        if results is None:
            logger.error("无法进行比较，数据不足")
            return None
        
        # 输出结果
        logger.info(f"\n处理顺序效应分析结果（指标: {metric}）:")
        logger.info("-" * 60)
        
        for model, result in results.items():
            model_name = {'decision_tree': '决策树', 'random_forest': '随机森林', 'xgboost': 'XGBoost'}[model]
            
            logger.info(f"\n{model_name}:")
            logger.info(f"  P4 (SMOTE→FS): {result['pipeline1_mean']:.4f} ± {result['pipeline1_std']:.4f}")
            logger.info(f"  P5 (FS→SMOTE): {result['pipeline2_mean']:.4f} ± {result['pipeline2_std']:.4f}")
            logger.info(f"  提升幅度: {result['improvement']:.2f}%")
            logger.info(f"  配对t检验: t={result['t_test']['t_statistic']:.4f}, p={result['t_test']['p_value']:.4f}")
            logger.info(f"  显著性: {'显著' if result['t_test']['significant'] else '不显著'} (α=0.05)")
            logger.info(f"  效应量(Cohen's d): {result['t_test']['cohens_d']:.4f}")
        
        return results
    
    def generate_comparison_table(self, metric='f1_macro'):
        """
        生成对比表格
        
        Args:
            metric: 评估指标
            
        Returns:
            pd.DataFrame: 对比表格
        """
        logger.info("生成对比表格...")
        
        df = self.results_df.copy()
        
        # 创建透视表
        pivot = df.pivot_table(
            values=metric,
            index='model',
            columns='pipeline',
            aggfunc=['mean', 'std']
        )
        
        # 保存表格
        output_path = self.stats_dir / f'comparison_table_{metric}.csv'
        pivot.to_csv(output_path)
        logger.info(f"对比表格已保存: {output_path}")
        
        return pivot
    
    def generate_statistical_report(self):
        """
        生成完整的统计报告
        
        Returns:
            str: 报告内容
        """
        logger.info("生成统计报告...")
        
        report = []
        report.append("=" * 80)
        report.append("统计检验报告 | Statistical Analysis Report")
        report.append("=" * 80)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"实验数据目录: {self.output_dir}")
        report.append(f"总实验记录数: {len(self.results_df)}")
        report.append("")
        
        # 1. 描述性统计
        report.append("-" * 80)
        report.append("1. 描述性统计 | Descriptive Statistics")
        report.append("-" * 80)
        
        for metric in ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']:
            if metric in self.results_df.columns:
                report.append(f"\n{metric.upper()}:")
                desc = self.results_df.groupby('pipeline')[metric].agg(['mean', 'std', 'min', 'max'])
                report.append(desc.to_string())
        
        # 2. 处理顺序效应分析
        report.append("\n" + "=" * 80)
        report.append("2. 处理顺序效应分析 | Processing Order Effect Analysis")
        report.append("=" * 80)
        
        order_results = self.analyze_pipeline_order_effect('f1_macro')
        
        if order_results:
            for model, result in order_results.items():
                model_name = {'decision_tree': '决策树', 'random_forest': '随机森林', 'xgboost': 'XGBoost'}[model]
                report.append(f"\n{model_name}:")
                report.append(f"  P4 (SMOTE→FS): {result['pipeline1_mean']:.4f} ± {result['pipeline1_std']:.4f}")
                report.append(f"  P5 (FS→SMOTE): {result['pipeline2_mean']:.4f} ± {result['pipeline2_std']:.4f}")
                report.append(f"  提升: {result['improvement']:.2f}%")
                report.append(f"  t检验: t={result['t_test']['t_statistic']:.4f}, p={result['t_test']['p_value']:.4f}")
                report.append(f"  显著性: {'显著' if result['t_test']['significant'] else '不显著'}")
        
        # 3. 特征选择方法对比
        report.append("\n" + "=" * 80)
        report.append("3. 特征选择方法对比 | Feature Selection Method Comparison")
        report.append("=" * 80)
        
        fs_results = self.compare_pipelines('P3', 'P5', 'f1_macro')
        if fs_results:
            report.append("\nFilter vs Wrapper (以P5为例):")
            # 这里可以添加更详细的分析
        
        # 4. 综合结论
        report.append("\n" + "=" * 80)
        report.append("4. 综合结论 | Conclusions")
        report.append("=" * 80)
        
        if order_results:
            significant_models = [m for m, r in order_results.items() if r['t_test']['significant']]
            if significant_models:
                report.append(f"\n在以下模型中，FS→SMOTE显著优于SMOTE→FS:")
                for model in significant_models:
                    model_name = {'decision_tree': '决策树', 'random_forest': '随机森林', 'xgboost': 'XGBoost'}[model]
                    improvement = order_results[model]['improvement']
                    report.append(f"  - {model_name}: 提升 {improvement:.2f}%")
            else:
                report.append("\n在当前实验条件下，两种处理顺序的差异未达到统计显著性水平。")
        
        report_text = "\n".join(report)
        
        # 保存报告
        output_path = self.stats_dir / 'statistical_report.txt'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"统计报告已保存: {output_path}")
        
        return report_text
    
    def run_all_analysis(self):
        """运行所有统计分析"""
        logger.info("=" * 60)
        logger.info("开始统计分析...")
        logger.info("=" * 60)
        
        # 1. 生成对比表格
        self.generate_comparison_table('f1_macro')
        
        # 2. 处理顺序效应分析
        self.analyze_pipeline_order_effect('f1_macro')
        
        # 3. 生成完整报告
        self.generate_statistical_report()
        
        logger.info("=" * 60)
        logger.info("统计分析完成！")
        logger.info(f"结果保存位置: {self.stats_dir}")
        logger.info("=" * 60)


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("实验结果统计分析")
    logger.info("=" * 60)
    
    try:
        # 创建分析器
        analyzer = StatisticalAnalyzer()
        
        # 运行所有分析
        analyzer.run_all_analysis()
        
        logger.info("统计分析完成！")
        
    except Exception as e:
        logger.error(f"统计分析过程出错: {e}")
        raise


if __name__ == "__main__":
    main()
