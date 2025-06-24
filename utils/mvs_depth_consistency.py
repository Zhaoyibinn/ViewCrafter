import numpy as np
import cv2

# Filter hyperparameter Settings  
########################
s = 1
dist_base = 1/8
rel_diff_base = 1/10

# s = 1
# dist_base = 1.0
# rel_diff_base = 1.0
########################


def interpolate_depth_map(known_coords, known_depths, output_shape, invalid_value=np.nan):
    """
    基于已知的非整数像素坐标及其深度值，插值生成完整的深度图
    
    参数:
    known_coords (np.ndarray): 已知深度的像素坐标，形状为 (n_points, 2)，每行为 [x, y]
    known_depths (np.ndarray): 对应的深度值，形状为 (n_points,)
    output_shape (tuple): 输出深度图的形状 (height, width)
    invalid_value: 无法插值区域的填充值
    
    返回:
    depth_map (np.ndarray): 插值后的完整深度图
    """
    # 确保输入是numpy数组
    known_coords = np.asarray(known_coords, dtype=np.float32)
    known_depths = np.asarray(known_depths, dtype=np.float32)
    
    # 创建输出深度图，初始化为无效值
    depth_map = np.full(output_shape, invalid_value, dtype=np.float32)
    
    # 获取输出图像的尺寸
    height, width = output_shape
    
    # 创建网格坐标
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    
    # 将网格坐标展平为一维数组
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    
    # 对每个输出像素，找到最近的4个已知坐标点进行双线性插值
    for i in range(len(x_flat)):
        x, y = x_flat[i], y_flat[i]
        
        # 找到周围的4个整数坐标点
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        # x1, y1 = x0 + 1, y0 + 1
        
        # 检查坐标是否在有效范围内
        # if x0 < 0 or x1 >= width or y0 < 0 or y1 >= height:
        #     continue
            
        # 找到这4个点对应的已知深度值
        # 这里使用欧氏距离找到最近的已知点
        distances = np.sqrt(
            (known_coords[:, 0] - x0)**2 + (known_coords[:, 1] - y0)**2
        )
        idx00 = np.argmin(distances)
        d00 = known_depths[idx00]
        
        # distances = np.sqrt(
        #     (known_coords[:, 0] - x1)**2 + (known_coords[:, 1] - y0)**2
        # )
        # idx10 = np.argmin(distances)
        # d10 = known_depths[idx10]
        
        # distances = np.sqrt(
        #     (known_coords[:, 0] - x0)**2 + (known_coords[:, 1] - y1)**2
        # )
        # idx01 = np.argmin(distances)
        # d01 = known_depths[idx01]
        
        # distances = np.sqrt(
        #     (known_coords[:, 0] - x1)**2 + (known_coords[:, 1] - y1)**2
        # )
        # idx11 = np.argmin(distances)
        # d11 = known_depths[idx11]
        
        # 计算双线性插值权重
        # wx = x - x0
        # wy = y - y0
        
        # 双线性插值
        # interpolated_depth = (1 - wx) * (1 - wy) * d00 + \
        #                      wx * (1 - wy) * d10 + \
        #                      (1 - wx) * wy * d01 + \
        #                      wx * wy * d11
        interpolated_depth = d00
        # 将插值结果放入深度图
        depth_map[y, x] = interpolated_depth
    
    return np.nan_to_num(depth_map, nan=0.0)

# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    depth_src_origin = depth_src
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    
    # import open3d as o3d
    # point_cloud_2d = xyz_ref.T
    # downsampled_cloud = point_cloud_2d[::4]
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(downsampled_cloud)
    # o3d.io.write_point_cloud("test.ply", pcd)
    # zyb可视化点云

    # xyz_ref_world = np.matmul(np.linalg.inv(extrinsics_ref),np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # x_src, y_src = np.meshgrid(np.arange(0, width), np.arange(0, height))
    # x_src, y_src = x_ref.reshape([-1]), y_ref.reshape([-1])
    # xyz_src = np.matmul(np.linalg.inv(intrinsics_src),np.vstack((x_src, y_src, np.ones_like(x_src))) * depth_src.reshape([-1]))
    # xyz_src_world = np.matmul(np.linalg.inv(extrinsics_src),np.vstack((xyz_src, np.ones_like(x_src))))[:3]
    # import open3d as o3d
    # downsampled_cloud_ref = xyz_ref_world.T[::4]
    # pcd_ref = o3d.geometry.PointCloud()
    # pcd_ref.points = o3d.utility.Vector3dVector(downsampled_cloud_ref)

    # downsampled_cloud_src = xyz_src_world.T[::4]
    # pcd_src = o3d.geometry.PointCloud()
    # pcd_src.points = o3d.utility.Vector3dVector(downsampled_cloud_src)

    # pcd_ref.paint_uniform_color([1, 0, 0])
    # pcd_src.paint_uniform_color([0, 1, 0])
    # o3d.io.write_point_cloud("test.ply", pcd_ref + pcd_src)
    # # zyb可视化点云

    # xyz_ref计算在当前视角下的三维点坐标合集
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    

    # xyz_src xyz_ref转到src视角下
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]
    # 归一化计算回像素坐标 

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32) # 投影之后的点的x，y坐标
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    depth_src = K_xyz_src[2:3].reshape([height, width]).astype(np.float32)



    
    # reconstructed_depth = interpolate_depth_map(
    #     np.stack([x_src.flatten(), y_src.flatten()], axis=1), depth_src.flatten(), (height, width)
    # )
    # 这里是深度图渲染 很慢所以注释了
    
    valid = (x_src >= 0) & (x_src < width) & (y_src >= 0) & (y_src < height) & (depth_src > 0)
    valid_proj_x = x_src[valid].astype(np.int32)
    valid_proj_y = y_src[valid].astype(np.int32)
    valid_depth = depth_src[valid]
    depth_dst = np.zeros((height, width), dtype=np.float32)
    depth_dst[valid_proj_y, valid_proj_x] = valid_depth

    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # 这里相当于知道了src视图下对应像素的坐标变换 就可以重新采样回来 也就是获得了src视图下的这些对应点的深度
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # 和上面一样 再反投影回去 xyz_reprojected是反投影之后在当前坐标下的三维坐标 因为上面是采样的就不要重新渲染了 原本就是规整的
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    K_xyz_reprojected[2:3][K_xyz_reprojected[2:3]==0] += 0.00001
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    # 返回值
    # 前面三个就是反投影的深度,xy,其中depth_reprojected是已经和当前视图像素对齐了的
    # 后面两个是当前视图投影到参考视图之后的xy坐标(z已经归一化了,所以是像素坐标)
    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src

def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                                                                                                 depth_src, intrinsics_src, extrinsics_src)
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)
    # 反投影之后 xy坐标的差异

    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref 
    # 反投影之后,深度的变化比率

    mask = None
    masks = []
    masks_per = []
    for i in range(s,500):
        mask = np.logical_or(np.logical_and(dist < i * dist_base, relative_depth_diff < i * rel_diff_base),dist>100.0,relative_depth_diff==np.inf)
        masks.append(mask)
        masks_per.append(np.mean(mask))
    depth_reprojected[~mask] = 0

    return masks, masks_per,depth_reprojected, x2d_src, y2d_src