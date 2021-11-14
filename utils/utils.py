import numpy as np
import torch
print("import torch")
from knn_cuda import KNN

def knn_point(group_size, point_cloud, query_cloud, transpose_mode=False):
    knn_obj = KNN(k=group_size, transpose_mode=transpose_mode) # transpose_mode=False表示最后两维不转置
    dist, idx = knn_obj(point_cloud, query_cloud) # 找query_points（）最近的K个点
    return dist, idx

def nonuniform_sampling(num, sample_num):  # num为总数，sample_num为采样数
    sample = set()  # 每次随机一个index加入集合，保证不重复
    loc = np.random.rand() * 0.8 + 0.1  # 均匀采样一个0.1~0.9的随机数作为高斯随机的均值
    while len(sample) < sample_num:
        a = int(np.random.normal(loc=loc, scale=0.3) * num)  # 以loc为均值，scale为方差的高斯分布中随机一个数并乘以总数作为采样index
        # 判断采样的index是否在范围内，不在范围内则丢弃
        if a < 0 or a >= num:
            continue
        sample.add(a)
    return list(sample)

def save_xyz_file(numpy_array, xyz_dir):
    num_points = numpy_array.shape[0]
    with open(xyz_dir, 'w') as f:
        for i in range(num_points):
            line = "%f %f %f\n" % (numpy_array[i, 0], numpy_array[i, 1], numpy_array[i, 2])
            f.write(line)
    return


def rotate_point_cloud_and_gt(input_data, gt_data=None):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction （依次沿着每个轴旋转一定的角度）
        Input:
          Nx3 array, original point cloud
        Return:
          Nx3 array, rotated point cloud
    """
    angles = np.random.uniform(size=(3)) * 2 * np.pi  # 随机三个在0~360度的角度值，作为欧拉角
    # 将欧拉角转换成旋转矩阵
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]),  np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0,  np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]),  np.cos(angles[2]), 0],
                   [0, 0, 1]])
    rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))

    input_data[:, :3] = np.dot(input_data[:, :3], rotation_matrix)
    if input_data.shape[1] > 3:  # 如果包含法向量，法向量也要做相同的旋转
        input_data[:, 3:] = np.dot(input_data[:, 3:], rotation_matrix)
    
    if gt_data is not None:
        # gt和input做相同的旋转
        gt_data[:, :3] = np.dot(gt_data[:, :3], rotation_matrix)
        if gt_data.shape[1] > 3:  # 法向量
            gt_data[:, 3:] = np.dot(gt_data[:, 3:], rotation_matrix)

    return input_data, gt_data


def random_scale_point_cloud_and_gt(input_data, gt_data=None, scale_low=0.5, scale_high=2):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            Nx3 array, original point cloud
        Return:
            Nx3 array, scaled point cloud
    """
    scale = np.random.uniform(scale_low, scale_high)
    input_data[:, :3] *= scale
    if gt_data is not None:
        gt_data[:, :3] *= scale

    return input_data, gt_data, scale


def shift_point_cloud_and_gt(input_data, gt_data=None, shift_range=0.3):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          Nx3 array, original point cloud
        Return:
          Nx3 array, shifted point cloud
    """
    shifts = np.random.uniform(-shift_range, shift_range, 3)
    input_data[:, :3] += shifts
    if gt_data is not None:
        gt_data[:, :3] += shifts
    return input_data, gt_data


def jitter_perturbation_point_cloud(input_data, sigma=0.005, clip=0.02):
    """ Randomly jitter points. jittering is per point.
        Input:
          Nx3 array, original point cloud
        Return:
          Nx3 array, jittered point cloud
    """
    assert(clip > 0)
    # jitter的形成：首先对input的每个点随机一个随机值，然后将这些值clip到（-clip,clip）之间，防止抖动太大
    jitter = np.clip(sigma * np.random.randn(*input_data.shape), -1 * clip, clip)
    jitter[:, 3:] = 0  # 只jitter坐标，不jitter法向量
    input_data += jitter  # 对每个输入点做一个小抖动
    return input_data


def rotate_perturbation_point_cloud(input_data, angle_sigma=0.03, angle_clip=0.09):
    """ Randomly perturb the point clouds by small rotations
        Input:
          Nx3 array, original point cloud
        Return:
          Nx3 array, rotated point cloud
    """
    angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1, 0, 0],
                   [0,np.cos(angles[0]), -np.sin(angles[0])],
                   [0,np.sin(angles[0]),  np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,  np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]),  np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    input_data[:, :3] = np.dot(input_data[:, :3], R)
    if input_data.shape[1] > 3:
        input_data[:, 3:] = np.dot(input_data[:, 3:], R)
    return input_data