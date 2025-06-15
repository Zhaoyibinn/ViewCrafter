
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