#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
围岩裂隙精准识别与三维模型重构系统
Rock Fracture Precise Recognition and 3D Model Reconstruction System

作者: AI Assistant
日期: 2024
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy import ndimage
from scipy.spatial.distance import pdist, squareform
import open3d as o3d
import os
import glob
import pandas as pd
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import json
import warnings
warnings.filterwarnings('ignore')

class FractureDetector:
    """围岩裂隙检测类"""
    
    def __init__(self, config: Dict = None):
        """
        初始化裂隙检测器
        
        Args:
            config: 配置参数字典
        """
        self.config = config or self._default_config()
        self.fracture_data = []
        
    def _default_config(self) -> Dict:
        """默认配置参数"""
        return {
            'canny_low': 50,
            'canny_high': 150,
            'min_line_length': 30,
            'max_line_gap': 10,
            'hough_threshold': 80,
            'dbscan_eps': 15,
            'dbscan_min_samples': 3,
            'fracture_min_length': 20,
            'edge_kernel_size': 3
        }
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理
        
        Args:
            image: 输入图像
            
        Returns:
            处理后的图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 高斯滤波去噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 直方图均衡化增强对比度
        enhanced = cv2.equalizeHist(blurred)
        
        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        return enhanced
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        边缘检测
        
        Args:
            image: 预处理后的图像
            
        Returns:
            边缘图像
        """
        # Canny边缘检测
        edges = cv2.Canny(image, 
                         self.config['canny_low'], 
                         self.config['canny_high'])
        
        # 形态学操作连接断裂的边缘
        kernel = np.ones((self.config['edge_kernel_size'], 
                         self.config['edge_kernel_size']), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return edges
    
    def detect_lines(self, edges: np.ndarray) -> List[Tuple]:
        """
        直线检测
        
        Args:
            edges: 边缘图像
            
        Returns:
            检测到的直线列表 [(x1, y1, x2, y2), ...]
        """
        # 霍夫直线变换
        lines = cv2.HoughLinesP(edges,
                               rho=1,
                               theta=np.pi/180,
                               threshold=self.config['hough_threshold'],
                               minLineLength=self.config['min_line_length'],
                               maxLineGap=self.config['max_line_gap'])
        
        if lines is None:
            return []
        
        return [tuple(line[0]) for line in lines]
    
    def calculate_line_features(self, line: Tuple) -> Dict:
        """
        计算直线特征
        
        Args:
            line: 直线坐标 (x1, y1, x2, y2)
            
        Returns:
            直线特征字典
        """
        x1, y1, x2, y2 = line
        
        # 长度
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # 角度 (相对于水平线)
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        if angle < 0:
            angle += 180
        
        # 中点
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        return {
            'length': length,
            'angle': angle,
            'center_x': center_x,
            'center_y': center_y,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
        }
    
    def cluster_fractures(self, lines: List[Tuple]) -> List[List[Dict]]:
        """
        裂隙聚类分组
        
        Args:
            lines: 检测到的直线列表
            
        Returns:
            聚类后的裂隙组
        """
        if not lines:
            return []
        
        # 计算直线特征
        features = [self.calculate_line_features(line) for line in lines]
        
        # 提取用于聚类的特征向量
        feature_vectors = []
        for f in features:
            feature_vectors.append([
                f['center_x'], f['center_y'], 
                f['angle'], f['length']
            ])
        
        feature_vectors = np.array(feature_vectors)
        
        # 标准化
        scaler = StandardScaler()
        feature_vectors_scaled = scaler.fit_transform(feature_vectors)
        
        # DBSCAN聚类
        clustering = DBSCAN(eps=self.config['dbscan_eps'], 
                           min_samples=self.config['dbscan_min_samples'])
        cluster_labels = clustering.fit_predict(feature_vectors_scaled)
        
        # 组织聚类结果
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(features[i])
        
        return list(clusters.values())
    
    def analyze_single_image(self, image_path: str) -> Dict:
        """
        分析单张图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            分析结果
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 预处理
        processed = self.preprocess_image(image)
        
        # 边缘检测
        edges = self.detect_edges(processed)
        
        # 直线检测
        lines = self.detect_lines(edges)
        
        # 聚类分析
        fracture_groups = self.cluster_fractures(lines)
        
        # 统计分析
        total_fractures = len([group for group in fracture_groups if len(group) >= 1])
        total_length = sum([
            f['length'] for group in fracture_groups 
            for f in group if f['length'] >= self.config['fracture_min_length']
        ])
        
        return {
            'image_path': image_path,
            'total_fractures': total_fractures,
            'total_length': total_length,
            'fracture_groups': fracture_groups,
            'original_image': image,
            'processed_image': processed,
            'edges': edges,
            'lines': lines
        }

class ThreeDReconstructor:
    """三维重构类"""
    
    def __init__(self, borehole_diameter: float = 0.1):
        """
        初始化三维重构器
        
        Args:
            borehole_diameter: 钻孔直径(米)
        """
        self.borehole_diameter = borehole_diameter
        self.point_clouds = []
        self.fracture_planes = []
        
    def depth_to_z_coordinate(self, depth_interval: str) -> float:
        """
        将深度区间转换为Z坐标
        
        Args:
            depth_interval: 深度区间字符串，如 "0-1m"
            
        Returns:
            Z坐标值
        """
        start_depth = float(depth_interval.split('-')[0])
        end_depth = float(depth_interval.split('-')[1].replace('m', ''))
        return -(start_depth + end_depth) / 2  # 负值表示向下
    
    def fracture_to_3d_points(self, fracture_data: Dict, borehole_x: float, 
                             borehole_y: float, depth: float) -> np.ndarray:
        """
        将二维裂隙转换为三维点云
        
        Args:
            fracture_data: 裂隙数据
            borehole_x: 钻孔X坐标
            borehole_y: 钻孔Y坐标  
            depth: 深度
            
        Returns:
            三维点云
        """
        points_3d = []
        
        for group in fracture_data['fracture_groups']:
            for fracture in group:
                if fracture['length'] < 20:  # 过滤短裂隙
                    continue
                
                # 假设裂隙在钻孔壁面上，将2D坐标映射到3D圆柱面
                x1, y1 = fracture['x1'], fracture['y1']
                x2, y2 = fracture['x2'], fracture['y2']
                
                # 沿裂隙线段采样点
                num_points = max(10, int(fracture['length'] / 5))
                for i in range(num_points):
                    t = i / (num_points - 1)
                    x_2d = x1 + t * (x2 - x1)
                    y_2d = y1 + t * (y2 - y1)
                    
                    # 归一化到钻孔半径
                    angle = (x_2d / 500) * 2 * np.pi  # 假设图像宽度500像素对应360度
                    height_offset = (y_2d - 250) / 250 * 0.5  # 图像高度对应0.5米变化
                    
                    # 转换为3D坐标
                    radius = self.borehole_diameter / 2
                    x_3d = borehole_x + radius * np.cos(angle)
                    y_3d = borehole_y + radius * np.sin(angle)
                    z_3d = depth + height_offset
                    
                    points_3d.append([x_3d, y_3d, z_3d])
        
        return np.array(points_3d)
    
    def create_borehole_layout(self, num_boreholes: int = 6) -> List[Tuple]:
        """
        创建钻孔布局
        
        Args:
            num_boreholes: 钻孔数量
            
        Returns:
            钻孔坐标列表 [(x, y), ...]
        """
        # 创建规则网格布局
        positions = []
        if num_boreholes == 6:
            # 2×3网格
            for i in range(2):
                for j in range(3):
                    x = j * 5.0  # 5米间距
                    y = i * 5.0
                    positions.append((x, y))
        else:
            # 圆形布局
            for i in range(num_boreholes):
                angle = 2 * np.pi * i / num_boreholes
                radius = 5.0
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                positions.append((x, y))
        
        return positions
    
    def reconstruct_3d_model(self, analysis_results: Dict) -> Dict:
        """
        重构三维模型
        
        Args:
            analysis_results: 分析结果字典
            
        Returns:
            三维重构结果
        """
        all_points = []
        borehole_positions = self.create_borehole_layout()
        
        for borehole_id, borehole_data in analysis_results.items():
            if not borehole_id.endswith('孔'):
                continue
                
            borehole_idx = int(borehole_id[0]) - 1
            if borehole_idx >= len(borehole_positions):
                continue
                
            borehole_x, borehole_y = borehole_positions[borehole_idx]
            
            for depth_interval, fracture_data in borehole_data.items():
                depth = self.depth_to_z_coordinate(depth_interval)
                points_3d = self.fracture_to_3d_points(
                    fracture_data, borehole_x, borehole_y, depth
                )
                
                if len(points_3d) > 0:
                    all_points.extend(points_3d)
        
        if not all_points:
            return {'point_cloud': np.array([]), 'fracture_planes': []}
        
        point_cloud = np.array(all_points)
        
        # 平面拟合
        fracture_planes = self._fit_fracture_planes(point_cloud)
        
        return {
            'point_cloud': point_cloud,
            'fracture_planes': fracture_planes,
            'borehole_positions': borehole_positions
        }
    
    def _fit_fracture_planes(self, points: np.ndarray, 
                           max_planes: int = 5) -> List[Dict]:
        """
        拟合裂隙平面
        
        Args:
            points: 点云数据
            max_planes: 最大平面数量
            
        Returns:
            拟合的平面参数列表
        """
        if len(points) < 3:
            return []
        
        planes = []
        remaining_points = points.copy()
        
        for i in range(max_planes):
            if len(remaining_points) < 100:  # 最少点数要求
                break
            
            # 使用RANSAC拟合平面
            try:
                # 随机采样一致性算法
                best_plane = None
                best_inliers = []
                best_score = 0
                
                for trial in range(1000):
                    # 随机选择3个点
                    sample_indices = np.random.choice(
                        len(remaining_points), 3, replace=False
                    )
                    sample_points = remaining_points[sample_indices]
                    
                    # 计算平面方程 ax + by + cz + d = 0
                    v1 = sample_points[1] - sample_points[0]
                    v2 = sample_points[2] - sample_points[0]
                    normal = np.cross(v1, v2)
                    
                    if np.linalg.norm(normal) < 1e-6:
                        continue
                    
                    normal = normal / np.linalg.norm(normal)
                    d = -np.dot(normal, sample_points[0])
                    
                    # 计算所有点到平面的距离
                    distances = np.abs(
                        np.dot(remaining_points, normal) + d
                    )
                    
                    # 找到内点
                    inliers = distances < 0.1  # 10cm阈值
                    inlier_count = np.sum(inliers)
                    
                    if inlier_count > best_score:
                        best_score = inlier_count
                        best_plane = (normal, d)
                        best_inliers = inliers
                
                if best_plane and best_score > 50:  # 最少内点数
                    planes.append({
                        'normal': best_plane[0],
                        'distance': best_plane[1],
                        'points': remaining_points[best_inliers],
                        'num_points': best_score
                    })
                    
                    # 移除已拟合的点
                    remaining_points = remaining_points[~best_inliers]
                else:
                    break
                    
            except Exception as e:
                print(f"平面拟合出错: {e}")
                break
        
        return planes

class FractureAnalyzer:
    """裂隙分析系统主类"""
    
    def __init__(self):
        """初始化分析系统"""
        self.detector = FractureDetector()
        self.reconstructor = ThreeDReconstructor()
        self.results = {}
        
    def analyze_borehole_data(self, base_path: str) -> Dict:
        """
        分析钻孔数据
        
        Args:
            base_path: 数据根目录
            
        Returns:
            分析结果
        """
        borehole_pattern = os.path.join(base_path, '附件4', '*孔')
        borehole_dirs = glob.glob(borehole_pattern)
        
        for borehole_dir in borehole_dirs:
            borehole_name = os.path.basename(borehole_dir)
            print(f"分析钻孔: {borehole_name}")
            
            # 获取所有深度区间图像
            image_pattern = os.path.join(borehole_dir, '*.jpg')
            image_files = glob.glob(image_pattern)
            
            borehole_results = {}
            
            for image_file in image_files:
                depth_interval = os.path.basename(image_file).replace('.jpg', '')
                print(f"  分析深度: {depth_interval}")
                
                try:
                    result = self.detector.analyze_single_image(image_file)
                    borehole_results[depth_interval] = result
                except Exception as e:
                    print(f"    错误: {e}")
                    continue
            
            self.results[borehole_name] = borehole_results
        
        return self.results
    
    def generate_statistics(self) -> pd.DataFrame:
        """
        生成统计数据
        
        Returns:
            统计数据DataFrame
        """
        stats_data = []
        
        for borehole_name, borehole_data in self.results.items():
            for depth_interval, analysis_result in borehole_data.items():
                stats_data.append({
                    '钻孔': borehole_name,
                    '深度区间': depth_interval,
                    '裂隙数量': analysis_result['total_fractures'],
                    '总长度': analysis_result['total_length'],
                    '平均长度': (analysis_result['total_length'] / 
                               max(1, analysis_result['total_fractures']))
                })
        
        return pd.DataFrame(stats_data)
    
    def create_3d_model(self) -> Dict:
        """
        创建三维模型
        
        Returns:
            三维模型数据
        """
        return self.reconstructor.reconstruct_3d_model(self.results)
    
    def visualize_results(self, output_dir: str = 'output'):
        """
        可视化结果
        
        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 统计图表
        stats_df = self.generate_statistics()
        
        plt.figure(figsize=(15, 10))
        
        # 裂隙数量分布
        plt.subplot(2, 3, 1)
        stats_df.groupby('钻孔')['裂隙数量'].sum().plot(kind='bar')
        plt.title('各钻孔裂隙总数量')
        plt.xticks(rotation=45)
        
        # 深度-裂隙关系
        plt.subplot(2, 3, 2)
        depth_stats = stats_df.groupby('深度区间')['裂隙数量'].mean()
        depth_stats.plot(kind='line', marker='o')
        plt.title('深度与裂隙数量关系')
        plt.xticks(rotation=45)
        
        # 裂隙长度分布
        plt.subplot(2, 3, 3)
        plt.hist(stats_df['总长度'], bins=20, alpha=0.7)
        plt.title('裂隙长度分布')
        plt.xlabel('总长度')
        plt.ylabel('频次')
        
        # 热力图
        plt.subplot(2, 3, 4)
        pivot_table = stats_df.pivot_table(
            values='裂隙数量', index='钻孔', columns='深度区间', fill_value=0
        )
        sns.heatmap(pivot_table, annot=True, cmap='YlOrRd')
        plt.title('钻孔-深度裂隙分布热力图')
        
        # 3D散点图预览
        plt.subplot(2, 3, 5)
        model_3d = self.create_3d_model()
        if len(model_3d['point_cloud']) > 0:
            points = model_3d['point_cloud']
            plt.scatter(points[:, 0], points[:, 1], 
                       c=points[:, 2], cmap='viridis', alpha=0.6)
            plt.colorbar(label='深度(m)')
            plt.title('三维点云投影(XY平面)')
            plt.xlabel('X(m)')
            plt.ylabel('Y(m)')
        
        # 保存统计图
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '裂隙分析统计.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 保存详细结果
        stats_df.to_csv(os.path.join(output_dir, '裂隙统计数据.csv'), 
                       index=False, encoding='utf-8-sig')
        
        # 3. 三维可视化
        self._save_3d_visualization(model_3d, output_dir)
        
        print(f"结果已保存到 {output_dir} 目录")
    
    def _save_3d_visualization(self, model_3d: Dict, output_dir: str):
        """
        保存三维可视化
        
        Args:
            model_3d: 三维模型数据
            output_dir: 输出目录
        """
        if len(model_3d['point_cloud']) == 0:
            print("没有足够的三维数据进行可视化")
            return
        
        try:
            # 创建Open3D点云
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(model_3d['point_cloud'])
            
            # 添加颜色(基于深度)
            points = model_3d['point_cloud']
            colors = plt.cm.viridis((points[:, 2] - points[:, 2].min()) / 
                                   (points[:, 2].max() - points[:, 2].min()))[:, :3]
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # 保存点云
            o3d.io.write_point_cloud(
                os.path.join(output_dir, '裂隙三维点云.ply'), pcd
            )
            
            # 创建钻孔可视化
            geometries = [pcd]
            
            # 添加钻孔柱体
            for i, (x, y) in enumerate(model_3d['borehole_positions']):
                cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                    radius=0.05, height=7.0
                )
                cylinder.translate([x, y, -3.5])
                cylinder.paint_uniform_color([0.5, 0.5, 0.5])
                geometries.append(cylinder)
            
            # 保存场景
            print("三维模型已生成，可使用Open3D查看 .ply 文件")
            
        except Exception as e:
            print(f"三维可视化保存失败: {e}")

def main():
    """主函数"""
    print("=== 围岩裂隙精准识别与三维模型重构系统 ===")
    print("Rock Fracture Precise Recognition and 3D Model Reconstruction System")
    print()
    
    # 初始化分析器
    analyzer = FractureAnalyzer()
    
    # 设置数据路径
    base_path = "/home/runner/work/test/test"
    
    # 检查数据是否存在
    attachment4_path = os.path.join(base_path, "附件4")
    if not os.path.exists(attachment4_path):
        print(f"错误: 找不到数据目录 {attachment4_path}")
        return
    
    print("开始分析钻孔数据...")
    
    # 分析数据
    try:
        results = analyzer.analyze_borehole_data(base_path)
        print(f"成功分析了 {len(results)} 个钻孔")
        
        # 生成结果
        print("生成统计数据和可视化...")
        analyzer.visualize_results()
        
        # 输出核心数学公式和方法
        print_mathematical_formulas()
        
        print("\n分析完成！")
        
    except Exception as e:
        print(f"分析过程中出错: {e}")
        import traceback
        traceback.print_exc()

def print_mathematical_formulas():
    """输出核心数学公式和方法说明"""
    
    formulas = """
    
=== 核心数学公式与方法 ===

1. 边缘检测 (Canny算法)
   梯度幅值: |G| = √(Gx² + Gy²)
   梯度方向: θ = arctan(Gy/Gx)
   
2. 直线检测 (霍夫变换)
   直线方程: ρ = x·cos(θ) + y·sin(θ)
   其中 ρ 为原点到直线距离，θ 为法线角度
   
3. 裂隙长度计算
   长度: L = √((x₂-x₁)² + (y₂-y₁)²)
   角度: α = arctan((y₂-y₁)/(x₂-x₁)) × 180/π
   
4. 聚类分析 (DBSCAN)
   特征向量: F = [中心X, 中心Y, 角度, 长度]
   标准化: F' = (F - μ)/σ
   
5. 三维坐标转换
   柱面坐标到直角坐标:
   x = r·cos(θ) + x₀
   y = r·sin(θ) + y₀  
   z = z₀ + Δz
   
6. 平面拟合 (RANSAC)
   平面方程: ax + by + cz + d = 0
   法向量: n = (a, b, c)
   点到平面距离: dist = |ax₀ + by₀ + cz₀ + d|/√(a² + b² + c²)
   
7. 裂隙密度计算
   线密度: ρₗ = N/L  (条/米)
   面密度: ρₐ = ΣL/A  (米/平方米)
   
8. 空间插值
   反距离权重: w = 1/d^p
   插值值: Z = Σ(wᵢ·zᵢ)/Σwᵢ

=== 技术特点 ===
- 多尺度图像处理
- 机器学习聚类分析  
- 三维几何重构
- 统计分析与可视化
- 参数自适应优化

"""
    print(formulas)

if __name__ == "__main__":
    main()