"""
可视化配置模块

设置matplotlib中文字体和样式
确保所有图表中的文字都是中文
"""

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
import warnings

import config


def setup_chinese_font():
    """
    设置matplotlib中文字体
    
    尝试多种中文字体，确保图表中的文字能正常显示
    """
    # 常见中文字体列表（按优先级）
    chinese_fonts = [
        'SimHei',           # 黑体（Windows）
        'Microsoft YaHei',  # 微软雅黑（Windows）
        'SimSun',           # 宋体（Windows）
        'Arial Unicode MS', # Mac
        'WenQuanYi Micro Hei', # Linux
        'Noto Sans CJK SC', # Linux
        'DejaVu Sans',      # 备用
    ]
    
    # 尝试设置中文字体
    font_found = False
    for font_name in chinese_fonts:
        try:
            # 检查字体是否可用
            font_path = font_manager.findfont(font_name, fallback_to_default=False)
            if font_path:
                plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
                plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
                font_found = True
                print(f"中文字体已设置 | Chinese font set: {font_name}")
                break
        except:
            continue
    
    if not font_found:
        warnings.warn("未找到中文字体，图表中的中文可能显示为方框 | Chinese font not found, Chinese characters may display as squares")
    
    # 设置全局样式
    plt.rcParams['figure.figsize'] = config.VISUALIZATION['figsize']
    plt.rcParams['figure.dpi'] = config.VISUALIZATION['dpi']
    plt.rcParams['font.size'] = config.VISUALIZATION['font_size']
    
    return font_found


def setup_plot_style():
    """
    设置绘图样式
    """
    try:
        plt.style.use(config.VISUALIZATION['style'])
    except:
        plt.style.use('seaborn-v0_8')
    
    # 设置中文字体
    setup_chinese_font()


# 初始化时自动设置
setup_plot_style()
