import os
import torch
import numpy as np
import imageio.v2 as imageio
import json
import torch.nn.functional as F
import cv2
from scipy.spatial import transform
import math
from PIL import Image


def blender_quat2rot(quaternion):
  """Convert quaternion to rotation matrix.


  Equivalent to, but support batched case:

  ```python
  rot3x3 = mathutils.Quaternion(quaternion).to_matrix()
  ```

  Args:
    quaternion:

  Returns:
    rotation matrix
  """

  # Note: Blender first cast to double values for numerical precision while
  # we're using float32.
  q = np.sqrt(2) * quaternion

  q0 = q[..., 0]
  q1 = q[..., 1]
  q2 = q[..., 2]
  q3 = q[..., 3]

  qda = q0 * q1
  qdb = q0 * q2
  qdc = q0 * q3
  qaa = q1 * q1
  qab = q1 * q2
  qac = q1 * q3
  qbb = q2 * q2
  qbc = q2 * q3
  qcc = q3 * q3

  # Note: idx are inverted as blender and numpy convensions do not
  # match (x, y) -> (y, x)
  rotation = np.empty((*quaternion.shape[:-1], 3, 3), dtype=np.float32)
  rotation[..., 0, 0] = 1.0 - qbb - qcc
  rotation[..., 1, 0] = qdc + qab
  rotation[..., 2, 0] = -qdb + qac

  rotation[..., 0, 1] = -qdc + qab
  rotation[..., 1, 1] = 1.0 - qaa - qcc
  rotation[..., 2, 1] = qda + qbc

  rotation[..., 0, 2] = qdb + qac
  rotation[..., 1, 2] = -qda + qbc
  rotation[..., 2, 2] = 1.0 - qaa - qbb
  return rotation

def make_transform_matrix(positions,rotations):
  """Create the 4x4 transformation matrix.

  Note: This function uses numpy.

  Args:
    positions: Translation applied after the rotation.
      Last column of the transformation matrix
    rotations: Rotation. Top-left 3x3 matrix of the transformation matrix.

  Returns:
    transformation_matrix:
  """
  # Create the 4x4 transformation matrix
  rot_pos = np.broadcast_to(np.eye(4), (*positions.shape[:-1], 4, 4)).copy()
  rot_pos[..., :3, :3] = rotations
  rot_pos[..., :3, 3] = positions
  return rot_pos

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1, use_saliency = False):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    all_saliency = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        print("meta: ")
        print("camera_angle_x", type(meta["camera_angle_x"]), meta["camera_angle_x"])
        print("trans matrix", type(meta["frames"][0]['transform_matrix']))
        imgs = []
        poses = []
        saliency = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
            if use_saliency:
                saliency.append(np.load(fname[0:-4] + '_heat.npy'))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        print("imgs: ", imgs.shape)
        poses = np.array(poses).astype(np.float32)
        print("poese: ", poses.shape)
        if use_saliency:
            saliency = np.array(saliency / 255.).astype(np.float32)        
        counts.append(counts[-1] + imgs.shape[0])
        print("counts: ", counts[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
        if use_saliency:
            all_saliency.append(saliency)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    if use_saliency:
        saliency = np.concatenate(all_saliency, 0)
    print("imgs, poses, i_split: ", imgs.shape, poses.shape, len(i_split))
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
    print("i_splits:", i_split)
    for i in range(10):
        print (poses[i])
    if use_saliency:
        return imgs, poses, render_poses, [H, W, focal], i_split, saliency
    else:
        return imgs, poses, render_poses, [H, W, focal], i_split

def load_Nesf_data(basedir, half_res=False, testskip=1):
    with open(os.path.join(basedir,"metadata.json"), 'r') as fp:
            file = json.load(fp)
    splits = ['train', 'val', 'test']
    metas = {}
    H = file["metadata"]['height']
    W = file["metadata"]['width']
    focal = file["camera"]['focal_length']
    n_frames = file["metadata"]['num_frames']
    train_id = file["split_ids"]["train"]
    test_id = file["split_ids"]["test"]
    all_imgs = []
    all_poses = []
    counts = [0]
    near = 10000
    far = 0
    minBounds = file["scene_boundaries"]["min"]
    maxBounds = file["scene_boundaries"]["max"]
    K = file["camera"]["K"]

    imgs = []
    poses = []
    for i in train_id:
        fname = os.path.join(basedir, "rgba_"+"%05d" % i+".png")
        # print(fname)
        imgs.append(imageio.imread(fname))
        pos = file["camera"]["positions"][i]
        quat = file["camera"]["quaternions"][i]
        dis1 = math.sqrt(pow((minBounds[0]-pos[0]),2)+pow((minBounds[1]-pos[1]),2)+pow((minBounds[1]-pos[1]),2))
        dis2 = math.sqrt(pow((maxBounds[0]-pos[0]),2)+pow((maxBounds[1]-pos[1]),2)+pow((maxBounds[1]-pos[1]),2))
        # rotations = transform.Rotation.from_quat(quat).as_matrix()
        quat = np.asarray(quat)
        pos = np.asarray(pos)
        rotation = blender_quat2rot(quat)
        # rotation = transform.Rotation.from_quat(quat).as_matrix()

        pose = make_transform_matrix(pos, rotation)
        poses.append(pose)

        minDis = min(dis1, dis2)
        maxDis = max(dis1, dis2)
        near = min(near,minDis)
        far = max(far, maxDis)

    imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
    print("imgs: ", imgs.shape)
    poses = np.array(poses).astype(np.float32)
    print("poese: ", poses.shape)
    counts.append(counts[-1] + imgs.shape[0])
    all_imgs.append(imgs)
    all_poses.append(poses)

    imgs = []
    poses = []
    for i in test_id:
        fname = os.path.join(basedir, "rgba_"+"%05d" % i+".png")
        # print(fname)
        imgs.append(imageio.imread(fname))
        pos = file["camera"]["positions"][i]
        quat = file["camera"]["quaternions"][i]
        dis1 = math.sqrt(pow((minBounds[0]-pos[0]),2)+pow((minBounds[1]-pos[1]),2)+pow((minBounds[1]-pos[1]),2))
        dis2 = math.sqrt(pow((maxBounds[0]-pos[0]),2)+pow((maxBounds[1]-pos[1]),2)+pow((maxBounds[1]-pos[1]),2))
        quat = np.asarray(quat)
        pos = np.asarray(pos)
        # rotation = transform.Rotation.from_quat(quat).as_matrix()
        rotation = blender_quat2rot(quat)
        pose = make_transform_matrix(pos, rotation)
        poses.append(pose)

        minDis = min(dis1, dis2)
        maxDis = max(dis1, dis2)
        near = min(near,minDis)
        far = max(far, maxDis)

    imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
    print("imgs: ", imgs.shape)
    poses = np.array(poses).astype(np.float32)
    print("poese: ", poses.shape)
    counts.append(counts[-1] + imgs.shape[0])
    all_imgs.append(imgs)
    all_poses.append(poses)

    imgs = []
    poses = []
    for i in test_id:
        fname = os.path.join(basedir, "rgba_"+"%05d" % i+".png")
        # print(fname)
        imgs.append(imageio.imread(fname))
        pos = file["camera"]["positions"][i]
        quat = file["camera"]["quaternions"][i]
        dis1 = math.sqrt(pow((minBounds[0]-pos[0]),2)+pow((minBounds[1]-pos[1]),2)+pow((minBounds[1]-pos[1]),2))
        dis2 = math.sqrt(pow((maxBounds[0]-pos[0]),2)+pow((maxBounds[1]-pos[1]),2)+pow((maxBounds[1]-pos[1]),2))

        quat = np.asarray(quat)
        pos = np.asarray(pos)
        rotation = blender_quat2rot(quat)
        # rotation = transform.Rotation.from_quat(quat).as_matrix()

        pose = make_transform_matrix(pos, rotation)
        poses.append(pose)
        minDis = min(dis1, dis2)
        maxDis = max(dis1, dis2)
        near = min(near,minDis)
        far = max(far, maxDis)

    imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
    print("imgs: ", imgs.shape)
    poses = np.array(poses).astype(np.float32)
    print("poese: ", poses.shape)
    counts.append(counts[-1] + imgs.shape[0])
    all_imgs.append(imgs)
    all_poses.append(poses)
    #-------------------------------loaded all 3 sets of data--------------------------------------
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    print("imgs, poses, i_split: ", imgs.shape, poses.shape, len(i_split))
    render_poses = torch.stack([pose_spherical(angle, -30.0, 10.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

    print("near, far: ", near, far)
    # print(poses[0])
    # print(quat)
    # print(pos)
    # print(pose)
    for i in range(10):
        print (poses[i])

    return imgs, poses, render_poses, [H, W, focal], i_split, near, far, K



if __name__=='__main__':
    print("-------------------------load blender data---------------------------------------")
    load_blender_data("./data/nerf_synthetic/chair")
    print("-----------------------load nesf data--------------------------------------------")
    load_Nesf_data("/gpfs/data/ssrinath/ychen485/implicitSearch/implicitObjDetection/toybox-13/0")
    



