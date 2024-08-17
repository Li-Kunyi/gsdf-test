import open3d as o3d
import numpy as np

# 读取点云数据
pcd = o3d.io.read_point_cloud("/home/kunyi/work/data/TNT_GOF/TrainingSet/Barn/sparse/0/points3D.ply")

# 可视化原始点云
o3d.visualization.draw_geometries([pcd], window_name="Original Point Cloud")

# 统计滤波
cl, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
filtered_pcd = pcd.select_by_index(ind)

# 尝试不同参数进行半径滤波
nb_points = 32
radius = 1.0

# 半径滤波
cl, ind = filtered_pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
filtered_pcd = filtered_pcd.select_by_index(ind)

# 可视化过滤后的点云
o3d.visualization.draw_geometries([filtered_pcd], window_name="Filtered Point Cloud")

# 计算点云包围盒（bounding box）
aabb = filtered_pcd.get_axis_aligned_bounding_box()
print(f"Axis-aligned bounding box: \nCenter: {aabb.get_center()} \nExtents: {aabb.get_extent()}")

# 可视化包围盒
o3d.visualization.draw_geometries([filtered_pcd, aabb], window_name="Bounding Box")