import json
import os

from tqdm import tqdm

from .dataset import Dataset
from .video import Video

class LasHeRVideo(Video):
    """
    Args:
        name: video name
        root: dataset root
        video_dir: video directory
        init_rect: init rectangle
        img_names: image names
        gt_rect: groundtruth rectangle
        attr: attribute of video
    """
    def __init__(self, name, root, video_dir, init_rect_v, img_names_v, img_names_i,
            gt_rect_v, init_rect_i, gt_rect_i, init_init, init_rect, attr, load_img=False):
        super(LasHeRVideo, self).__init__(name, root, video_dir,
                init_rect_v, img_names_v, img_names_i, gt_rect_v, init_rect_i, gt_rect_i, init_init, init_rect, attr, load_img)


class LasHeRDataset(Dataset):
    """
    Args:
        name:  dataset name, should be "NFS30" or "NFS240"
        dataset_root, dataset root dir
    """
    def __init__(self, name, dataset_root, load_img=False):
        super(LasHeRDataset, self).__init__(name, dataset_root)
        with open(os.path.join(dataset_root, 'LasHeR_visible.json'), 'r') as f:
            meta_data_v = json.load(f)

        with open(os.path.join(dataset_root, 'LasHeR_infrared.json'), 'r') as f:
            meta_data_i = json.load(f)

        with open(os.path.join(dataset_root, 'LasHeR_init.json'), 'r') as f:
            meta_data_init = json.load(f)

        # load videos
        pbar = tqdm(meta_data_v.keys(), desc='loading '+name, ncols=100)
        self.videos = {}
        for video in pbar:
            pbar.set_postfix_str(video)
            self.videos[video] = LasHeRVideo(video,
                                          dataset_root,
                                          meta_data_v[video]['video_dir'],
                                          meta_data_v[video]['init_rect'],
                                          meta_data_v[video]['img_names'],
                                          meta_data_i[video]['img_names'],
                                          meta_data_v[video]['gt_rect'],
                                          meta_data_i[video]['init_rect'],
                                          meta_data_i[video]['gt_rect'],
                                          meta_data_init[video]['init_rect'],
                                          meta_data_init[video]['gt_rect'],
                                          None)
        self.attr = {}
        self.attr['ALL'] = list(self.videos.keys())
