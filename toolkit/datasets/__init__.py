from .otb import OTBDataset
from .uav import UAVDataset
from .lasot import LaSOTDataset
from .got10k import GOT10kDataset
from .GTOT_G import GTOTDataset
from .RGBT234 import RGBT234Dataset
from .LasHer import LasHeRDataset
from .data import dataDataset
from .VTUAV import VTUAVDataset

class DatasetFactory(object):
    @staticmethod
    def create_dataset(**kwargs):
        """
        Args:
            name: dataset name 'OTB2015', 'LaSOT', 'UAV123', 'NFS240', 'NFS30',
                'VOT2018', 'VOT2016', 'VOT2018-LT'
            dataset_root: dataset root
            load_img: wether to load image
        Return:
            dataset
        """
        assert 'name' in kwargs, "should provide dataset name"
        name = kwargs['name']
        if 'OTB' in name:
            dataset = OTBDataset(**kwargs)
        elif 'GTOT' == name:
            dataset = GTOTDataset(**kwargs)
        elif 'RGB_T234' == name:
            dataset = RGBT234Dataset(**kwargs)
        elif 'LasHeR' == name:
            dataset = LasHeRDataset(**kwargs)
        elif 'data' == name:
            dataset = dataDataset(**kwargs)
        # elif 'UAV' in name:
        #     dataset = UAVDataset(**kwargs)
        elif 'GOT-10k' == name:
            dataset = GOT10kDataset(**kwargs)
        elif 'VTUAV' == name:
            dataset = VTUAVDataset(**kwargs)
        else:
            raise Exception("unknow dataset {}".format(kwargs['name']))
        return dataset

