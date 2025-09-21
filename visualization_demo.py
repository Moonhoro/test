#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成可视化结果展示
Generate Visualization Results Display
"""

import os
import json

def create_ascii_charts():
    """创建ASCII图表"""
    
    # 读取分析结果
    with open('output/分析结果数据.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    stats = data['statistics']
    
    print("=== 围岩裂隙分析 - 可视化结果 ===")
    print()
    
    # 1. 钻孔裂隙数量分布
    print("1. 各钻孔裂隙数量分布:")
    print("   钻孔    数量    图形表示")
    print("   " + "-" * 35)
    
    borehole_stats = stats['by_borehole']
    max_count = max(s['fracture_count'] for s in borehole_stats.values())
    
    for borehole, s in borehole_stats.items():
        count = s['fracture_count']
        bar_length = int(20 * count / max_count)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"   {borehole:6s} {count:4d}    {bar} {count}")
    
    print()
    
    # 2. 深度分布
    print("2. 深度分层裂隙分布:")
    print("   深度层     数量    占比    图形表示")
    print("   " + "-" * 45)
    
    depth_stats = stats['depth_statistics']
    total = stats['total_fractures']
    
    layers = [
        ('表层 0-2m', depth_stats['surface']['count']),
        ('浅层 2-4m', depth_stats['shallow']['count']),
        ('中层 4-6m', depth_stats['medium']['count']),
        ('深层 >6m', depth_stats['deep']['count'])
    ]
    
    for layer_name, count in layers:
        percentage = 100 * count / total
        bar_length = int(15 * count / total) if total > 0 else 0
        bar = "▓" * bar_length + "░" * (15 - bar_length)
        print(f"   {layer_name:10s} {count:4d}  {percentage:5.1f}%  {bar}")
    
    print()
    
    # 3. 角度分布
    print("3. 裂隙角度分布:")
    print("   角度类型       数量    占比    图形表示")
    print("   " + "-" * 45)
    
    angle_dist = stats['angle_distribution']
    angle_items = [
        ('垂直 75°-105°', angle_dist['vertical']),
        ('近垂直 60°-75°', angle_dist['sub_vertical']),
        ('倾斜 30°-60°', angle_dist['inclined']),
        ('近水平 0°-30°', angle_dist['horizontal'])
    ]
    
    for angle_name, count in angle_items:
        percentage = 100 * count / total
        bar_length = int(15 * count / total) if total > 0 else 0
        bar = "▓" * bar_length + "░" * (15 - bar_length)
        print(f"   {angle_name:14s} {count:4d}  {percentage:5.1f}%  {bar}")
    
    print()
    
    # 4. 三维分布示意
    print("4. 钻孔三维布局示意 (俯视图):")
    print()
    print("   Y轴")
    print("    ↑")
    print("    │")
    print("  5 ├─ 4#孔 ── 5#孔 ── 6#孔")
    print("    │   ●      ●      ●   ")
    print("    │")
    print("  0 ├─ 1#孔 ── 2#孔 ── 3#孔 ──→ X轴")
    print("    │   ●      ●      ●   ")
    print("    0   0      5      10  (米)")
    print()
    
    # 5. 深度剖面示意
    print("5. 深度剖面示意 (侧视图):")
    print()
    print("   深度(m)  裂隙密度")
    print("     0   ├─ ██████████████ 高密度")
    print("         │")
    print("     1   ├─ ███████████▓▓▓")
    print("         │")
    print("     2   ├─ ██████████▓▓▓▓")
    print("         │")
    print("     3   ├─ ████████▓▓▓▓▓▓")
    print("         │")
    print("     4   ├─ ██████▓▓▓▓▓▓▓▓")
    print("         │")
    print("     5   ├─ █████▓▓▓▓▓▓▓▓▓")
    print("         │")
    print("     6   ├─ ████▓▓▓▓▓▓▓▓▓▓ 低密度")
    print("         │")
    print("     7   └─ ███▓▓▓▓▓▓▓▓▓▓▓")
    print()
    
    # 6. 统计汇总表
    print("6. 关键统计指标:")
    print("   " + "=" * 40)
    print(f"   总钻孔数量: {len(borehole_stats)} 个")
    print(f"   检测裂隙总数: {stats['total_fractures']} 条")
    print(f"   累计裂隙长度: {stats['total_length']:.1f} 像素")
    print(f"   平均裂隙长度: {stats['total_length']/max(1,stats['total_fractures']):.1f} 像素")
    print(f"   主导裂隙方向: 垂直 ({100*angle_dist['vertical']/total:.1f}%)")
    print(f"   最高密度层: 表层 0-2m ({depth_stats['surface']['count']} 条)")
    print("   " + "=" * 40)
    print()

def create_technical_summary():
    """创建技术总结"""
    
    print("=== 技术方案总结 ===")
    print()
    
    print("核心算法:")
    print("├─ 图像预处理")
    print("│  ├─ 高斯滤波去噪")
    print("│  ├─ 直方图均衡化")
    print("│  └─ 形态学操作")
    print("├─ 特征检测")
    print("│  ├─ Canny边缘检测")
    print("│  ├─ 霍夫直线变换")
    print("│  └─ 几何特征计算")
    print("├─ 智能聚类")
    print("│  ├─ 特征向量构建")
    print("│  ├─ DBSCAN聚类")
    print("│  └─ 异常值过滤")
    print("├─ 三维重构")
    print("│  ├─ 坐标系转换")
    print("│  ├─ 点云生成")
    print("│  └─ 平面拟合")
    print("└─ 统计分析")
    print("   ├─ 密度计算")
    print("   ├─ 分布统计")
    print("   └─ 可视化输出")
    print()
    
    print("数学模型:")
    print("• 边缘检测: |G| = √(Gx² + Gy²)")
    print("• 霍夫变换: ρ = x·cos(θ) + y·sin(θ)")
    print("• 裂隙长度: L = √((x₂-x₁)² + (y₂-y₁)²)")
    print("• 三维转换: x = x₀ + r·cos(θ), y = y₀ + r·sin(θ)")
    print("• 密度模型: ρ(z) = a·exp(-b·z) + c")
    print()
    
    print("工程价值:")
    print("✓ 围岩稳定性评价")
    print("✓ 支护方案优化")
    print("✓ 渗透性评估")
    print("✓ 风险区域识别")
    print("✓ 监测方案制定")
    print()

def main():
    """主函数"""
    if not os.path.exists('output/分析结果数据.json'):
        print("错误: 请先运行 simple_fracture_analysis.py 生成分析结果")
        return
    
    # 生成ASCII图表
    create_ascii_charts()
    
    # 生成技术总结
    create_technical_summary()
    
    print("=== 可视化展示完成 ===")
    print()
    print("更多详细结果请查看:")
    print("• output/分析报告.md - 完整分析报告")
    print("• output/裂隙统计数据.csv - 详细统计数据")
    print("• output/三维坐标数据.csv - 三维坐标文件")
    print("• 完整解决方案.md - 技术方案文档")

if __name__ == "__main__":
    main()