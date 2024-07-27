import os
import scipy.io as scio
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
# np.set_printoptions(threshold=np.inf)


def load_data_coloradar(radarpath, lidarpath, seqname):
    
    print("seqname", seqname)
    files = os.listdir(radarpath)
    files.sort()
    radar = []
    for i in files:
        path = os.path.join(radarpath, i)
        radar_img = Image.open(path).convert('L')
        # print("adc_i", adc_i.shape)
        radar.append(radar_img) 
    # print("seqname", seqname)
    # print("len(radarpath)", len(radar))
    files = os.listdir(lidarpath)
    files.sort()
    lidar = []
    for i in files:
        path = os.path.join(lidarpath, i)
        lidar_img = Image.open(path).convert('L')
        # print("adc_i", adc_i.shape)
        lidar.append(lidar_img)
    # print("len(lidar)", len(lidar))
    name_list = []
    for i in files:
        # print("asdf", seqname + "_" + i.split('.')[0])
        name_list.append(seqname + "_" + i.split('.')[0])
    # print("len(name_list)", len(name_list))
    assert(len(radar)==len(lidar)==len(name_list))

    return radar, lidar, name_list

def load_data_benchmark(adcpath, datapath, labelpath, seqname):
    
    norm = lambda x: (x - x.min())/(x.max() - x.min())

    files = os.listdir(adcpath)
    files.sort()
    adc = []
    for i in files:
        path = os.path.join(adcpath, i)
        adc_i = np.load(path)['rdm_multi']
        adc_i = adc_i.reshape((3, 4, 128, 128)) # warning zrb
        # print("adc_i", adc_i.shape)
        adc.append(adc_i)
    
    files = os.listdir(datapath)
    files.sort()
    data = [norm(scio.loadmat(os.path.join(datapath, i))['rdm'][np.newaxis]) for i in files]

    if not labelpath:
        return data
    
    files = os.listdir(labelpath)
    files.sort()
    label = [scio.loadmat(os.path.join(labelpath, i))['label'][np.newaxis] for i in files]
    
    
    assert(len(data)==len(label)==len(adc))

    name_list = []
    for i in files:
        # print("asdf", seqname + "_" + i.split('.')[0])
        name_list.append(seqname + "_" + i.split('.')[0])
        
    # print("data", data[0].dtype)
    # print("data", data[0].shape)
    # print("label", type(label[0]))
    # print("label", len(label))
    # print("label", label[0].shape)
    # print("label", label[0].dtype)
    # print("label", label[0].sum())
                      

    return adc, data, label, name_list

def load_data(datapath, labelpath = None):
    
    norm = lambda x: (x - x.min())/(x.max() - x.min())
    
    files = os.listdir(datapath)
    files.sort()
    data = [norm(scio.loadmat(os.path.join(datapath, i))['rdm'][np.newaxis]) for i in files]

    if not labelpath:
        return data
    
    files = os.listdir(labelpath)
    files.sort()
    label = [scio.loadmat(os.path.join(labelpath, i))['label'][np.newaxis] for i in files]
    
    assert(len(data)==len(label))
    # print("data", data[0].dtype)
    # print("data", data[0].shape)
    # print("label", type(label[0]))
    # print("label", len(label))
    # print("label", label[0].shape)
    # print("label", label[0].dtype)
    # print("label", label[0].sum())
                      

    return data, label

def init_dataset(config, dataset_path, transform, mode):


    Radar, Lidar, Name = [], [], []
    if mode == "train": 
        for i in config.data.train:
            # print("i", i)
            radar, lidar, name = load_data_coloradar(dataset_path + "/{}/range_azimuth_heatmap/".format(i), dataset_path + "{}/lidar_pcl_bev_polar_img/".format(i), i)
            Radar += radar
            Lidar += lidar
            Name += name
        dataset = myDataset_coloradar(Radar, Lidar, Name, transform)
        print("Using {} to train".format(config.data.train))
        print("Train data - {}".format(dataset.len))
    elif mode == "test": 
        for i in config.data.test:
            radar, lidar, name = load_data_coloradar(dataset_path + "{}/range_azimuth_heatmap/".format(i), dataset_path + "{}/lidar_pcl_bev_polar_img/".format(i), i)
            Radar += radar
            Lidar += lidar
            Name += name
        dataset = myDataset_coloradar(Radar, Lidar, Name, transform)
        print("Using {} to test".format(config.data.test))
        print("Test data - {}".format(dataset.len))

    return dataset


class myDataset_coloradar(Dataset):
    def __init__(self, radar, lidar, name, transform):

        self.radar = radar
        self.lidar = lidar
        self.name = name
        self.len = len(radar)
        self.transform = transform

    def __getitem__(self, index):
        
        # if self.label:
        #     return self.data[index], self.label[index]
        radar = self.radar[index]
        lidar = self.lidar[index]
        name = self.name[index]
        

        if self.transform:
            radar = self.transform(radar)
            lidar = self.transform(lidar)

        return (radar, lidar, name)

    def __len__(self):

        return self.len
    

class myDataset_adc(Dataset):
    def __init__(self, adc, data, label, name = None):

        self.data = data
        self.label = label
        self.adc = adc
        self.name = name
        self.len = len(data)

    def __getitem__(self, index):
        
        # if self.label:
        #     return self.data[index], self.label[index]
        return self.adc[index], self.data[index], self.label[index], self.name[index]

    def __len__(self):

        return self.len


class myDataset(Dataset):
    def __init__(self, data, label = None):

        self.data = data
        self.label = label
        self.len = len(data)

    def __getitem__(self, index):
        
        if self.label:
            return self.data[index], self.label[index]
        return self.data[index]

    def __len__(self):

        return self.len
