
import os
import cv2
import open3d as o3d
import numpy as np
import torch
import copy
import trimesh

import sys
sys.path.append("extern/dust3r")

from dust3r.inference import load_model

from dust3r.inference import inference, load_model
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy

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

def save_pointcloud_with_normals(imgs, pts3d, msk, save_path, mask_pc, reduce_pc):
    pc = get_pc(imgs, pts3d, msk,mask_pc,reduce_pc)  # Assuming get_pc is defined elsewhere and returns a trimesh point cloud

    # Define a default normal, e.g., [0, 1, 0]
    default_normal = [0, 1, 0]

    # Prepare vertices, colors, and normals for saving
    vertices = pc.vertices
    colors = pc.colors
    normals = np.tile(default_normal, (vertices.shape[0], 1))

    # Construct the header of the PLY file
    header = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property float nx
property float ny
property float nz
end_header
""".format(len(vertices))

    # Write the PLY file
    with open(save_path, 'w') as ply_file:
        ply_file.write(header)
        for vertex, color, normal in zip(vertices, colors, normals):
            ply_file.write('{} {} {} {} {} {} {} {} {}\n'.format(
                vertex[0], vertex[1], vertex[2],
                int(color[0]), int(color[1]), int(color[2]),
                normal[0], normal[1], normal[2]
            ))



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

    def run_dust3r(self):
        # assert self.sparse_colmap_path_root is not None, "未检测到Colmap格式的内外参"
        images, img_ori = load_initial_images(image_dir=self.img_path_list)

        Hs , Ws = [], []
        for img_path in self.img_path_list:
            origin_image = cv2.imread(img_path)
            H, W, _ = origin_image.shape
            Hs.append(H)
            Ws.append(W)
        assert len(set(Hs)) == 1 and len(set(Ws)) == 1, "所有图片的尺寸必须一致"
        
        self.origin_H = Hs[0]
        self.origin_W = Ws[0]

            

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

        self.masks = masks
        self.masks_filtered = masks_filtered
        self.pcd = pcd
        self.depth = depth
        self.imgs = imgs

        self.resized2origin_size()


    def resized2origin_size(self):
        self.masks_filtered_resized = []
        self.imgs_filtered_resized = []
        for ii in range(len(self.imgs)):
            mask_resized_1024_576 = cv2.resize(self.masks_filtered[ii].astype(np.uint8),(1024,576),cv2.IMREAD_UNCHANGED)
            scale_factor = self.origin_W / 1024
            resized_height = int(576 * scale_factor)
            mask_resized_2width = cv2.resize(mask_resized_1024_576,(self.origin_W, resized_height),cv2.IMREAD_UNCHANGED)
            pad_total = self.origin_H - resized_height  # 1162 - 765 = 397
            pad_top = pad_total // 2 # 上侧填充: 198
            # pad_bottom = pad_total - pad_top  # 下侧填充: 199

            padded_mask = np.zeros((self.origin_H, self.origin_W), dtype=mask_resized_2width.dtype)
            padded_mask[pad_top:pad_top + resized_height, :] = mask_resized_2width
            self.masks_filtered_resized.append(padded_mask)


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
            self.imgs_filtered_resized.append(padded_mask)

    def save_pointcloud_with_normals(self,filter = False,save_path = "test.ply",mask_pc = True):
        if filter:
            save_pointcloud_with_normals(self.imgs, self.pcd, msk=self.masks_filtered, save_path=save_path, mask_pc=mask_pc, reduce_pc=False)
        else:
            save_pointcloud_with_normals(self.imgs, self.pcd, msk=self.masks, save_path=save_path, mask_pc=mask_pc, reduce_pc=False)
    
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