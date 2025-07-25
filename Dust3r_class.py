
import os
import cv2
import open3d as o3d
import numpy as np
import torch
import copy
import trimesh
import shutil

import sys
sys.path.append("extern/dust3r")


from dust3r.inference import load_model

from dust3r.inference import inference, load_model
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy

from utils.mvs_depth_consistency import check_geometric_consistency
def load_initial_images(image_dir):
    ## load images
    ## dict_keys(['img', 'true_shape', 'idx', 'instance', 'img_ori']),张量形式
    images = load_images(image_dir, size=512,force_1024 = True)
    img_ori = (images[0]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2. # [576,1024,3] [0,1]

    if len(images) == 1:
        images = [images[0], copy.deepcopy(images[0])]
        images[1]['idx'] = 1

    return images, img_ori

def get_pc(imgs, pts3d, mask, mask_pc=False, reduce_pc=False):
    imgs = to_numpy(imgs)
    pts3d = to_numpy(pts3d)
    mask = to_numpy(mask)
    
    if mask_pc:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
    else:
        pts = np.concatenate([p for p in pts3d])
        col = np.concatenate([p for p in imgs])

    if reduce_pc:
        pts = pts.reshape(-1, 3)[::3]
        col = col.reshape(-1, 3)[::3]
    else:
        pts = pts.reshape(-1, 3)
        col = col.reshape(-1, 3)
    
    #mock normals:
    normals = np.tile([0, 1, 0], (pts.shape[0], 1))
    
    pct = trimesh.PointCloud(pts, colors=col)
    # debug
    # pct.export('output.ply')
    # print('exporting output.ply')
    pct.vertices_normal = normals  # Manually add normals to the point cloud
    
    return pct#, pts

def save_pointcloud_with_normals(imgs, pts3d, msk, save_path, mask_pc, reduce_pc,downsample_voxel = None):
    pc = get_pc(imgs, pts3d, msk,mask_pc,reduce_pc)  # Assuming get_pc is defined elsewhere and returns a trimesh point cloud

    # Define a default normal, e.g., [0, 1, 0]
    default_normal = [0, 1, 0]

    # Prepare vertices, colors, and normals for saving
    vertices = pc.vertices
    colors = pc.colors
    normals = np.tile(default_normal, (vertices.shape[0], 1))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(vertices))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors)[:,:3] / 255)
    pcd.normals = o3d.utility.Vector3dVector(np.array(normals))

    if downsample_voxel:
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=downsample_voxel)
        _, ind = downsampled_pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0
        )

        filtered_pcd = downsampled_pcd.select_by_index(ind)
        print(f"Dust3R进行了滤波和降采样 voxel = {downsample_voxel} 点云个数从{np.array(pcd.points).shape[0]}变成了{np.array(filtered_pcd.points).shape[0]}")
    else:
        downsampled_pcd = pcd
        _, ind = downsampled_pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0
        )
        filtered_pcd = downsampled_pcd.select_by_index(ind)
        print(f"Dust3R进行了滤波  点云个数从{np.array(pcd.points).shape[0]}变成了{np.array(filtered_pcd.points).shape[0]}")

    o3d.io.write_point_cloud(
            save_path,
            filtered_pcd,
        )

    print(f"Dust3R将滤波后的点云保存在{save_path}")

#     vertices = np.array(filtered_pcd.points)
#     vertices = np.array(filtered_pcd.points)
#     # Construct the header of the PLY file
#     header = """ply
# format ascii 1.0
# element vertex {}
# property float x
# property float y
# property float z
# property uchar red
# property uchar green
# property uchar blue
# property float nx
# property float ny
# property float nz
# end_header
# """.format(len(vertices))

#     # Write the PLY file
#     with open(save_path, 'w') as ply_file:
#         ply_file.write(header)
#         for vertex, color, normal in zip(vertices, colors, normals):
#             ply_file.write('{} {} {} {} {} {} {} {} {}\n'.format(
#                 vertex[0], vertex[1], vertex[2],
#                 int(color[0]), int(color[1]), int(color[2]),
#                 normal[0], normal[1], normal[2]
#             ))



def load_colmap_text(txt_path):
    """
    Load COLMAP points3D from a text file.
    The text file should be in the format used by COLMAP.
    """
    points_3d = []
    points_rgb = []
    
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        lines = lines[3:]
        for line in lines:
            parts = line.strip().split()

            
            point_id = int(parts[0])
            x, y, z = map(float, parts[1:4])
            r, g, b = map(int, parts[4:7])
            
            points_3d.append([x, y, z])
            points_rgb.append([r, g, b])
    
    return np.array(points_3d), np.array(points_rgb)

if __name__ == "__main__":
    path_root = "test/DTU_3/scan37"
    img_path_root = os.path.join(path_root, "images")
    sparse_colmap_path_root = os.path.join(path_root, "sparse","0")
    gt_extrinsic_path = os.path.join(sparse_colmap_path_root, "images.txt")
    gt_intrinsic_path = os.path.join(sparse_colmap_path_root, "cameras.txt")
    
    if os.path.exists(os.path.join(sparse_colmap_path_root, "points3D_colmap.ply")):
        gt_ply_path =  os.path.join(sparse_colmap_path_root, "points3D_colmap.ply")
        gt_ply = o3d.io.read_point_cloud(gt_ply_path)
    elif os.path.exists(os.path.join(sparse_colmap_path_root, "points3D.ply")):
        gt_ply_path =  os.path.join(sparse_colmap_path_root, "points3D.ply")
        gt_ply = o3d.io.read_point_cloud(gt_ply_path)
    elif os.path.exists(os.path.join(sparse_colmap_path_root, "points3D.txt")):
        xyz,rgb = load_colmap_text(os.path.join(sparse_colmap_path_root, "points3D.txt"))
        gt_ply = o3d.geometry.PointCloud()
        gt_ply.points = o3d.utility.Vector3dVector(xyz)
        gt_ply.colors = o3d.utility.Vector3dVector(rgb)

    img_path_list = sorted(os.listdir(img_path_root))
    img_path_list = [os.path.join(img_path_root, img_path) for img_path in img_path_list]

    img_list = []
    for img_path in img_path_list:
        img = cv2.imread(img_path)
        img_list.append(img)



    dust3r = load_model("./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth", "cuda")
    
    images, img_ori = load_initial_images(image_dir=img_path_list)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, dust3r, "cuda", batch_size=1)
    
    mode = GlobalAlignerMode.PointCloudOptimizer #if len(self.images) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device="cuda", mode=mode,dtu_path = sparse_colmap_path_root)
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=300, schedule='linear', lr=0.01)
    clean_pc = True
    if clean_pc:
        scene = scene.clean_pointcloud()
    else:
        scene = scene


    pcd = [i.detach() for i in scene.get_pts3d(clip_thred=1)] # a list of points of size whc
    depth = [i.detach() for i in scene.get_depthmaps()]


    # self.scene.min_conf_thr = float(self.scene.conf_trf(torch.tensor(self.opts.min_conf_thr)))
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(3.0)))
    masks = scene.get_masks()
    more_mask = scene.get_more_masks(1.0)
    # 这里传参的数字就代表了mask过滤的严格性 越大mask越严格
    depth = scene.get_depthmaps()
    bgs_mask = [dpt > 0.0*(torch.max(dpt[40:-40,:])+torch.min(dpt[40:-40,:])) for dpt in depth]
    masks_new = [m+mb for m, mb in zip(masks,bgs_mask)] 
    masks_filtered_new = [m&mb for m, mb in zip(more_mask,bgs_mask)] 
    masks = to_numpy(masks_new)
    masks_filtered = to_numpy(masks_filtered_new)
    mask_pc = True

    imgs = np.array(scene.imgs)

    save_pointcloud_with_normals(imgs, pcd, msk=masks, save_path="test.ply", mask_pc=mask_pc, reduce_pc=False)
    save_pointcloud_with_normals(imgs, pcd, msk=masks_filtered, save_path="test_filterted.ply", mask_pc=mask_pc, reduce_pc=False)
    print(f"Dust3r cal OK")
    # points_3d,points_rgb,extrinsic,intrinsic = demo_colmap.run_demo(img_list,gt_extrinsic = gt_extrinsic_path,gt_intrinsic = gt_intrinsic_path)


class Dust3r:
    def __init__(self):
        father_dir = os.path.dirname(os.path.abspath(__file__))
        self.dust3r_model = load_model(os.path.join(father_dir,"checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"), "cuda")

    def load_data(self, img_path_list, sparse_colmap_path_root=None):
        
        if os.path.exists(os.path.join(sparse_colmap_path_root, "points3D.txt")):
            xyz,rgb = load_colmap_text(os.path.join(sparse_colmap_path_root, "points3D.txt"))
            gt_pcd = o3d.geometry.PointCloud()
            gt_pcd.points = o3d.utility.Vector3dVector(xyz)
            gt_pcd.colors = o3d.utility.Vector3dVector(rgb)
        elif os.path.exists(os.path.join(sparse_colmap_path_root, "points3D_colmap.ply")):
            gt_ply_path =  os.path.join(sparse_colmap_path_root, "points3D_colmap.ply")
            gt_pcd = o3d.io.read_point_cloud(gt_ply_path)
        elif os.path.exists(os.path.join(sparse_colmap_path_root, "points3D.ply")):
            gt_ply_path =  os.path.join(sparse_colmap_path_root, "points3D.ply")
            gt_pcd = o3d.io.read_point_cloud(gt_ply_path)

        self.gt_pcd = gt_pcd
        self.img_path_list = img_path_list
        self.sparse_colmap_path_root = sparse_colmap_path_root

    def run_dust3r(self,resolution_scale = 1):
        # assert self.sparse_colmap_path_root is not None, "未检测到Colmap格式的内外参"
        images, img_ori = load_initial_images(image_dir=self.img_path_list)

        Hs , Ws = [], []
        for img_path in self.img_path_list:
            origin_image = cv2.imread(img_path)
            H, W, _ = origin_image.shape
            Hs.append(H)
            Ws.append(W)
        assert len(set(Hs)) == 1 and len(set(Ws)) == 1, "所有图片的尺寸必须一致"
        
        self.origin_H = int(Hs[0] / resolution_scale)
        self.origin_W = int(Ws[0] / resolution_scale)

            

        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, self.dust3r_model, "cuda", batch_size=1)
        
        mode = GlobalAlignerMode.PointCloudOptimizer #if len(self.images) > 2 else GlobalAlignerMode.PairViewer
        scene = global_aligner(output, device="cuda", mode=mode,dtu_path = self.sparse_colmap_path_root)
        if mode == GlobalAlignerMode.PointCloudOptimizer:
            loss = scene.compute_global_alignment(init='mst', niter=300, schedule='linear', lr=0.01)
        clean_pc = True
        if clean_pc:
            scene = scene.clean_pointcloud()
        else:
            scene = scene

        self.scene = scene

        pcd = [i.detach() for i in scene.get_pts3d(clip_thred=1)] # a list of points of size whc
        pcd_camera = [i.detach() for i in scene.get_pts3d_camera(clip_thred=1)] #在相机坐标系下的点集 就是像素坐标和深度
        depth = [i.detach() for i in scene.get_depthmaps(raw=True)]

        black_bg_mask = [((image['img'][0][0]!=-1)&(image['img'][0][1]!=-1)&(image['img'][0][2]!=-1)).cuda() for image in images]
        # self.scene.min_conf_thr = float(self.scene.conf_trf(torch.tensor(self.opts.min_conf_thr)))
        scene.min_conf_thr = float(scene.conf_trf(torch.tensor(3.0)))
        masks = scene.get_masks()
        masks = [mb & bb for mb,bb in zip(masks,black_bg_mask)]
        more_mask = scene.get_more_masks(1.0)
        more_mask = [mb & bb for mb,bb in zip(more_mask,black_bg_mask)]
        # 这里传参的数字就代表了mask过滤的严格性 越大mask越严格
        depth = scene.get_depthmaps()
        bgs_mask = [dpt > 0.0*(torch.max(dpt[40:-40,:])+torch.min(dpt[40:-40,:])) for dpt in depth]
        # 纯黑色的都作为背景被过滤
        bgs_mask = [mb & bb for mb,bb in zip(bgs_mask,black_bg_mask)]

        masks_new = [m+mb for m, mb in zip(masks,bgs_mask)] 
        masks_filtered_new = [m&mb for m, mb in zip(more_mask,bgs_mask)] 
        masks = to_numpy(masks_new)
        masks_filtered = to_numpy(masks_filtered_new)
        mask_pc = True

        imgs = np.array(scene.imgs)

        self.masks = masks
        self.masks_filtered = masks_filtered
        self.pcd = pcd
        self.depth = depth
        self.imgs = imgs
        self.mvs_filter_masks = []
        self.masks_filtered_with_mvs = []
        
        self.conf_img = [(im_conf * (bgs*1)).cpu().detach().numpy() for im_conf,bgs in zip(scene.im_conf,bgs_mask)]
        self.H,self.W = imgs.shape[1],imgs.shape[2]

        self.resized2origin_size()
        self.mvs_filter()


    # def cameras_trans_for_mvs(self,camera):
    #     current_R , current_t = camera.R,camera.T
    #     current_ext_mat = np.eye(4)
    #     current_ext_mat[:3,:3] = current_R
    #     current_ext_mat[:3,3] = current_t
    #     return current_ext_mat # C2W
        
    def run_only_model(self,imgs_list): 
        images, img_ori = load_initial_images(image_dir=imgs_list)
        pairs = make_pairs(images, scene_graph='pic2_pair', prefilter=None, symmetrize=True)
        output = inference(pairs, self.dust3r_model, "cuda", batch_size=1)

        point_cloud_np = output['pred1']['pts3d'].squeeze(0).reshape(-1, 3).cpu().numpy()

        mode = GlobalAlignerMode.PointCloudOptimizer #if len(self.images) > 2 else GlobalAlignerMode.PairViewer
        scene = global_aligner(output, device="cuda", mode=mode)
        if mode == GlobalAlignerMode.PointCloudOptimizer:
            loss = scene.compute_global_alignment(init='mst', niter=300, schedule='linear', lr=0.01)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(point_cloud_np)
        # downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.002)
        # print(f"降采样后点云点数: {len(downsampled_pcd.points)}")
        # o3d.io.write_point_cloud("test.ply", downsampled_pcd)
        return output,point_cloud_np,scene

    def mvs_filter(self):
        
        all_ref_idxs = []
        for ii in range(len(self.imgs)):
            range_num = 2
            ref_tart = max(0, ii - range_num)
            ref_end = min(len(self.imgs)-1, ii + range_num)
            ref_idxs = list(range(ref_tart, ref_end + 1))
            ref_idxs.remove(ii)
            all_ref_idxs.append(ref_idxs)
            dy_range = len(ref_idxs)

        for ii in range(len(self.imgs)):
            # current_camera = train_cameras[ii]
            current_ext_mat = self.scene.colmap_pose[ii].cpu().detach().numpy()
            current_focal = self.scene.colmap_focal[ii]
            current_in_mat = np.eye(3)
            current_in_mat[0][0] = current_in_mat[1][1] = current_focal
            current_in_mat[0][2] = self.imgs[ii].shape[1]/2
            current_in_mat[1][2] = self.imgs[ii].shape[0]/2
            current_depth = self.depth[ii].cpu().detach().numpy()
            ref_idxs = all_ref_idxs[ii]

            geo_mask_sum = 0
            
            for ref_idx in ref_idxs:
                ref_ext_mat = self.scene.colmap_pose[ref_idx].cpu().detach().numpy()
                ref_focal = self.scene.colmap_focal[ref_idx]
                ref_in_mat = np.eye(3)
                ref_in_mat[0][0] = ref_in_mat[1][1] = ref_focal
                ref_in_mat[0][2] = self.imgs[ii].shape[1]/2
                ref_in_mat[1][2] = self.imgs[ii].shape[0]/2

                ref_depth = self.depth[ref_idx].cpu().detach().numpy()

                masks, masks_per,depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(current_depth,current_in_mat,current_ext_mat,ref_depth,ref_in_mat,ref_ext_mat)
                
                geo_mask_sum += masks[140].astype(np.int32)
            geo_mask = geo_mask_sum >= dy_range * 0.9
            self.mvs_filter_masks.append(geo_mask)
            self.masks_filtered_with_mvs.append(np.logical_and(self.masks_filtered[ii],geo_mask))
            print(f"Dust3R 相机{ii} 过滤后还剩{np.mean(geo_mask)}的点")
            print("ok")

    
    def resized2origin_size(self):
        self.masks_filtered_resized = []
        self.depth_resized = []
        self.imgs_resized = []
        self.conf_img_resized = []
        for ii in range(len(self.imgs)):
            mask_resized_1024_576 = cv2.resize(self.masks_filtered[ii].astype(np.uint8),(1024,576),cv2.INTER_NEAREST)
            scale_factor = self.origin_W / 1024
            resized_height = int(576 * scale_factor)
            mask_resized_2width = cv2.resize(mask_resized_1024_576,(self.origin_W, resized_height),cv2.INTER_NEAREST)
            pad_total = self.origin_H - resized_height  # 1162 - 765 = 397
            pad_top = pad_total // 2 # 上侧填充: 198
            # pad_bottom = pad_total - pad_top  # 下侧填充: 199

            padded_mask = np.zeros((self.origin_H, self.origin_W), dtype=mask_resized_2width.dtype)
            padded_mask[pad_top:pad_top + resized_height, :] = mask_resized_2width
            self.masks_filtered_resized.append(padded_mask)

        for ii in range(len(self.imgs)):
            mask_resized_1024_576 = cv2.resize(self.depth[ii].cpu().detach().numpy(),(1024,576))
            scale_factor = self.origin_W / 1024
            resized_height = int(576 * scale_factor)
            mask_resized_2width = cv2.resize(mask_resized_1024_576,(self.origin_W, resized_height))
            pad_total = self.origin_H - resized_height  # 1162 - 765 = 397
            pad_top = pad_total // 2 # 上侧填充: 198
            # pad_bottom = pad_total - pad_top  # 下侧填充: 199

            padded_mask = np.zeros((self.origin_H, self.origin_W), dtype=mask_resized_2width.dtype)
            padded_mask[pad_top:pad_top + resized_height, :] = mask_resized_2width
            self.depth_resized.append(padded_mask)


        for ii in range(len(self.imgs)):
            mask_resized_1024_576 = cv2.resize((self.imgs[ii] * 255).astype(np.uint8),(1024,576))
            scale_factor = self.origin_W / 1024
            resized_height = int(576 * scale_factor)
            mask_resized_2width = cv2.resize(mask_resized_1024_576,(self.origin_W, resized_height))
            pad_total = self.origin_H - resized_height  # 1162 - 765 = 397
            pad_top = pad_total // 2 # 上侧填充: 198
            # pad_bottom = pad_total - pad_top  # 下侧填充: 199

            padded_mask = np.zeros((self.origin_H, self.origin_W , 3), dtype=mask_resized_2width.dtype)
            padded_mask[pad_top:pad_top + resized_height, :] = mask_resized_2width
            self.imgs_resized.append(padded_mask)

        for ii in range(len(self.imgs)):
            mask_resized_1024_576 = cv2.resize(self.conf_img[ii],(1024,576))
            scale_factor = self.origin_W / 1024
            resized_height = int(576 * scale_factor)
            mask_resized_2width = cv2.resize(mask_resized_1024_576,(self.origin_W, resized_height))
            pad_total = self.origin_H - resized_height  # 1162 - 765 = 397
            pad_top = pad_total // 2 # 上侧填充: 198
            # pad_bottom = pad_total - pad_top  # 下侧填充: 199

            padded_mask = np.zeros((self.origin_H, self.origin_W), dtype=mask_resized_2width.dtype)
            padded_mask[pad_top:pad_top + resized_height, :] = mask_resized_2width
            self.conf_img_resized.append(padded_mask)


    def save_pointcloud_with_normals(self,filter = False,save_path = "test.ply",mask_pc = True,downsample_voxel = None):
        if filter:
            save_pointcloud_with_normals(self.imgs, self.pcd, msk=self.masks_filtered_with_mvs, save_path=save_path, mask_pc=mask_pc, reduce_pc=False,downsample_voxel = downsample_voxel)
        else:
            save_pointcloud_with_normals(self.imgs, self.pcd, msk=self.masks_filtered, save_path=save_path, mask_pc=mask_pc, reduce_pc=False,downsample_voxel = downsample_voxel)
    
    def save_dust3r_depth(self):
        depth_save_dir = os.path.join(self.sparse_colmap_path_root.rsplit("/",2)[0],"depth_dust3r")
        if os.path.exists(depth_save_dir):
            # 如果存在则删除整个文件夹（包括里面的文件）
            shutil.rmtree(depth_save_dir)
        os.makedirs(depth_save_dir)
        for idx,depth_resized_1 in enumerate(self.depth_resized):
            img_path = self.img_path_list[idx]
            img_name = img_path.rsplit('/',1)[-1].split('.')[0]
            cv2.imwrite(f"{depth_save_dir}/{img_name}.tiff",depth_resized_1)
        
        pass

    def save_pointcloud_with_gt(self,save_path = "test_withgt.ply",mask_pc = True):
        pc = get_pc(self.imgs, self.pcd, self.masks_filtered,mask_pc=mask_pc,reduce_pc=False) 
        vertices = pc.vertices
        colors = pc.colors
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(vertices[:, :3])
        colors1 = np.tile( [1.0, 0.0, 0.0], (len(pcd1.points), 1))
        pcd1.colors = o3d.utility.Vector3dVector(colors1)


        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(np.array(self.gt_pcd.points)[:, :3])
        colors2 = np.tile( [0.0, 1.0, 0.0], (len(pcd2.points), 1))
        pcd2.colors = o3d.utility.Vector3dVector(colors2)


        # Save the combined point cloud to a PLY file
        o3d.io.write_point_cloud(save_path, pcd1+pcd2)
        print(f"PLY with GT saved to {save_path}")