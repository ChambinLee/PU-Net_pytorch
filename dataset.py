import torch.utils.data as torch_data
import h5py
import numpy as np
from utils import utils
from glob import glob
import os

class PUNET_Dataset_Whole(torch_data.Dataset):
    def __init__(self, data_dir='./datas/test_data/our_collected_data/MC_5k'):
        super().__init__()

        file_list = os.listdir(data_dir)
        self.names = [x.split('.')[0] for x in file_list]
        self.sample_path = [os.path.join(data_dir, x) for x in file_list]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        points = np.loadtxt(self.sample_path[index])
        return points


class PUNET_Dataset_WholeFPS_1k(torch_data.Dataset):
    def __init__(self, data_dir='./datas/test_data/obj_1k', use_norm=True):
        super().__init__()
        self.use_norm = use_norm

        folder_1k = os.path.join(data_dir, 'data_1k')
        folder_4k = os.path.join(data_dir, 'data_4k')
        file_list = os.listdir(folder_1k)
        self.names = [x.split('_')[0] for x in file_list]
        self.path_1k = [os.path.join(folder_1k, x) for x in os.listdir(folder_1k)]
        self.path_4k = [os.path.join(folder_4k, x) for x in os.listdir(folder_4k)]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        points = np.load(self.path_1k[index])
        gt = np.load(self.path_4k[index])

        if self.use_norm:
            centroid = np.mean(gt[:, :3], axis=0, keepdims=True) # 1, 3
            furthest_distance = np.amax(np.sqrt(np.sum((gt[:, :3] - centroid) ** 2, axis=-1)), axis=0, keepdims=True)

            gt[:, :3] -= centroid
            gt[:, :3] /= np.expand_dims(furthest_distance, axis=-1)
            points[:, :3] -= centroid
            points[:, :3] /= np.expand_dims(furthest_distance, axis=-1)
            return points, gt, np.array([1.0])
        else:
            raise NotImplementedError


class PUNET_Dataset(torch_data.Dataset):
    def __init__(self, h5_file_path='./datas/Patches_noHole_and_collected.h5', 
                    skip_rate=1, npoint=1024, use_random=True, use_norm=True, split='train', is_training=True):
        super().__init__()
        
        self.npoint = npoint
        self.use_random = use_random
        self.use_norm = use_norm
        self.is_training = is_training

        h5_file = h5py.File(h5_file_path)
        # h5_file保存了所有数据，总共是从40个mesh中切的4000个patch，test_list.txt和train_list.txt列出了训练和测试数据在其中的序号
        self.gt = h5_file['poisson_4096'][:] # [:] h5_obj => nparray
        self.input = h5_file['poisson_4096'][:] if use_random \
                            else h5_file['montecarlo_1024'][:]
        
        if split in ['train', 'test']:
            with open('./datas/{}_list.txt'.format(split), 'r') as f:  # 根据split选择训练或测试数据作为网络输入，分别包含3200、800个patch
                split_choice = [int(x) for x in f]
            self.gt = self.gt[split_choice, ...]
            self.input = self.input[split_choice, ...]
        elif split != 'all':  # 这个判断逻辑什么意思呢，如果是all是想把所有的数据都输入到网络里吗，tf版没有这个逻辑
            raise NotImplementedError("split must be 'train' or 'test' or 'all'")

        assert len(self.input) == len(self.gt), 'invalid data'
        self.data_npoint = self.input.shape[1]  # 每个patch包含的点数

        centroid = np.mean(self.gt[..., :3], axis=1, keepdims=True)  # 每个patch的centroid，:3代表点的坐标, patch_num*1*3
        # (self.gt[..., :3] - centroid) ** 2计算每个点到所属的centroid的欧氏距离(广播了)，(patch_num,4096,3),(x1-x2)^2, (y1-y2)^2, (z1-z2)^2
        # np.sum将最后一维的数据加起来，(patch_num,4096)  (x1-x2)^2+(y1-y2)^2+(z1-z2)^2
        # np.sqrt得到每个patch中4096个点到达各自centroid的欧氏距离,(patch_num,4096)
        # np.amax得到每个patch中距离centroid最远点的欧式距离,keepdims=True表示在求最值的那一维虽然只剩一个元素了，但是也保留哪一维，（patch_num,1）
        furthest_distance = np.amax(np.sqrt(np.sum((self.gt[..., :3] - centroid) ** 2, axis=-1)), axis=1, keepdims=True)
        self.radius = furthest_distance[:, 0]  # not very sure?  # 使用每个patch中所有点距离中心点的最远距离作为patch的半径,(patch_num,)

        if use_norm:  # 是否将每个patch归一化到单位球中
            self.radius = np.ones(shape=(len(self.input)))  # 归一化后所有patch的半径都为1，len函数只会返回array的第一维大小，（patch_num,）
            self.gt[..., :3] -= centroid
            # expand_dims表示在最后一维上插入一个新的轴，保证除法时广播机制正常
            self.gt[..., :3] /= np.expand_dims(furthest_distance, axis=-1)  # 这里将所有点除以所在patch的半径，将所有点归一化到单位球中
            self.input[..., :3] -= centroid
            self.input[..., :3] /= np.expand_dims(furthest_distance, axis=-1)

        self.input = self.input[::skip_rate]
        self.gt = self.gt[::skip_rate]
        self.radius = self.radius[::skip_rate]

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, index):
        input_data = self.input[index]
        gt_data = self.gt[index]
        radius_data = np.array([self.radius[index]])

        # 在此之前，input和gt是一样的（如果没使用MonteCarlo），下面就在input上随机采样，作为上采样网络的输入
        sample_idx = utils.nonuniform_sampling(self.data_npoint, sample_num=self.npoint)
        input_data = input_data[sample_idx, :]

        if self.use_norm:  # 是否将每个patch归一化到单位球中
            if not self.is_training:  # 不是训练过程，就不数据增强了，直接返回
                return input_data, gt_data, radius_data
            # 对训练数据做数据增强
            # for data aug，对gt和input
            input_data, gt_data = utils.rotate_point_cloud_and_gt(input_data, gt_data)  # 旋转
            input_data, gt_data, scale = utils.random_scale_point_cloud_and_gt(input_data, gt_data,
                                                                               scale_low=0.9, scale_high=1.1)  # 缩放
            input_data, gt_data = utils.shift_point_cloud_and_gt(input_data, gt_data, shift_range=0.1)  # 平移
            radius_data = radius_data * scale

            # for input aug，仅对input做,做这两个操作会使得input和gt的点不是一一对应的，二者只能保持语义上的物体表面的一致性
            if np.random.rand() > 0.5:  # 50%的概率对输入数据轻微抖动
                input_data = utils.jitter_perturbation_point_cloud(input_data, sigma=0.025, clip=0.05)
            if np.random.rand() > 0.5:  # 50%的概率对输入数据轻微旋转
                input_data = utils.rotate_perturbation_point_cloud(input_data, angle_sigma=0.03, angle_clip=0.09)
        else:  # 这里没有写不normalize数据的部分，tf版有
            raise NotImplementedError

        return input_data, gt_data, radius_data  # 返回指定index的patch

            
if __name__ == '__main__':
    test_choice = np.random.choice(4000, 800, replace=False)
    # f_test = open('test_list.txt', 'w')
    # f_train = open('train_list.txt', 'w')
    # train_list = []
    # test_list = []
    # for i in range(4000):
    #     if i in test_choice:
    #         test_list.append(i)
    #     else:
    #         train_list.append(i)
    # f_test.close()
    # f_train.close()

    # dst = PUNET_Dataset_WholeFPS_1k()
    # for batch in dst:
    #     pcd, gt, r = batch
    #     print(pcd.shape)
    #     print(gt.shape)
    #     print(r.shape)
    #     import pdb
    #     pdb.set_trace()

    ## test <PUNET_Dataset>
    # dst = PUNET_Dataset()
    # print(len(dst))
    # for batch in dst:
    #     pcd, gt, r = batch
    #     print(pcd.shape)
    #     import pdb
    #     pdb.set_trace()

    ## test <PUNET_Dataset_Whole>
    # dst = PUNET_Dataset_Whole()
    # points, name = dst[0]
    # print(points, name)