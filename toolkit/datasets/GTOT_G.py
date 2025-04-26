import json
import os

from tqdm import tqdm

from .dataset import Dataset
from .video import Video

class GTOTVideo(Video):
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
        super(GTOTVideo, self).__init__(name, root, video_dir,
                init_rect_v, img_names_v, img_names_i, gt_rect_v, init_rect_i, gt_rect_i, init_init, init_rect, attr, load_img)

    # def load_tracker(self, path, tracker_names=None):
    #     """
    #     Args:
    #         path(str): path to result
    #         tracker_name(list): name of tracker
    #     """
    #     if not tracker_names:
    #         tracker_names = [x.split('/')[-1] for x in glob(path)
    #                 if os.path.isdir(x)]
    #     if isinstance(tracker_names, str):
    #         tracker_names = [tracker_names]
    #     # self.pred_trajs = {}
    #     for name in tracker_names:
    #         traj_file = os.path.join(path, name, self.name+'.txt')
    #         if os.path.exists(traj_file):
    #             with open(traj_file, 'r') as f :
    #                 self.pred_trajs[name] = [list(map(float, x.strip().split(',')))
    #                         for x in f.readlines()]
    #             if len(self.pred_trajs[name]) != len(self.gt_traj):
    #                 print(name, len(self.pred_trajs[name]), len(self.gt_traj), self.name)
    #         else:

    #     self.tracker_names = list(self.pred_trajs.keys())

class GTOTDataset(Dataset):
    """
    Args:
        name:  dataset name, should be "NFS30" or "NFS240"
        dataset_root, dataset root dir
    """
    def __init__(self, name, dataset_root, load_img=False):
        super(GTOTDataset, self).__init__(name, dataset_root)
        with open(os.path.join(dataset_root, 'gtot_v.json'), 'r') as f:
            meta_data_v = json.load(f)

        with open(os.path.join(dataset_root, 'gtot_i.json'), 'r') as f:
            meta_data_i = json.load(f)

        with open(os.path.join(dataset_root, 'gtot_init.json'), 'r') as f:
            meta_data_init = json.load(f)

        # load videos
        pbar = tqdm(meta_data_v.keys(), desc='loading '+name, ncols=100)
        self.videos = {}
        for video in pbar:
            pbar.set_postfix_str(video)
            self.videos[video] = GTOTVideo(video,
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
