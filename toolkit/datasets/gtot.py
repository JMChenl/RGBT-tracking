from __future__ import absolute_import, print_function

import os
import glob
import numpy as np
import six

class GTOT(object):
    r"""
    Publication:
        RGB-t234:RGB-T Object Tracking:Benchmark and Baseline
    Args:
        root_dir:absolute path
        ...
    """

    def __init__(self, root_dir, list = 'gtot.txt'):
        super(GTOT, self).__init__()
        assert isinstance(root_dir, six.string_types) # 插入断点 判断数据类型
        assert isinstance(list, six.string_types)
        self.root_dir = root_dir # 得到图片路径
        self._check_integrity(root_dir, list)
        list_file = os.path.join(root_dir, list) # ../GTOT/GTOT.txt
        with open(list_file, 'r') as f: # 打开../GTOT/GTOT.txt文件
            self.seq_names = f.read().strip().split('\n') # 读取GTOT.txt文件中所有视频序列名字 空格发隔每个图像名
        self.seq_dirs_rgb = [os.path.join(root_dir, s, 'v') # RGB视频序列路径
                             for s in self.seq_names]
        self.seq_dirs_t = [os.path.join(root_dir, s, 'i') # 热红外视频序列路径
                           for s in self.seq_names]
        self.anno_files = [os.path.join(root_dir, s, 'init.txt') # 红外和rgb大致的目标位置标签路径
                           for s in self.seq_names]
        self.anno_files_rgb = [os.path.join(root_dir, s, 'groundTruth_v.txt') # rgb图像的目标锚框标签路径
                           for s in self.seq_names]
        self.anno_files_t = [os.path.join(root_dir, s, 'groundTruth_i.txt') # 热红外图像锚框标签路径
                           for s in self.seq_names]

    def __len__(self):
        return len(self.seq_names)

    def __getitem__(self, index):
        if isinstance(index, six.string_types):
            if not index in self.seq_names:
                raise Exception('Sequence {} not found.'.format(index))
            index =self.seq_names.index(index)

        img_files_rgb = sorted(glob.glob(os.path.join(
            self.seq_dirs_rgb[index], '*.*')))
        img_files_t = sorted(glob.glob(os.path.join(
            self.seq_dirs_t[index], '*.*')))
        anno = np.loadtxt(self.anno_files[index], delimiter='\t')
        anno_rgb = np.loadtxt(self.anno_files_rgb[index], delimiter=' ')
        anno_t = np.loadtxt(self.anno_files_t[index], delimiter=' ')
        assert len(img_files_rgb) == len(img_files_t) and len(img_files_t) == len(anno) \
               and len(anno) == len(anno_rgb) and len(anno_rgb) == len(anno_t)

        return img_files_rgb, img_files_t, anno, anno_rgb, anno_t

    def _check_integrity(self, root_dir, list = 'GTOT.txt'):
        list_file = os.path.join(root_dir, list)
        if os.path.isfile(list_file):
            with open(list_file, 'r') as f:
                seq_names = f.read().strip().split('\n') # 读取GTOT.txt文件中所有视频序列名字
            for seq_name in seq_names:
                seq_dir = os.path.join(root_dir, seq_name) # 找到视频序列名字对应的图片文件夹
                if not os.path.isdir(seq_dir):
                    print('Warning: sequence %s not exists.' % seq_name)
        else:
            raise Exception('Dataset not found or corrupted.')

if __name__ == "__main__":
    root_dir = '/root/cjm/object_tracker/My-/SiamCAR-master/test_dataset/GTOT'
    dataset = GTOT(root_dir = root_dir, list = 'gtot.txt')
    dataset.__getitem__(49)