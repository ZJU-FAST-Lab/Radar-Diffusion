"""Entrypoint of the package."""
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt
import yaml
from easydict import EasyDict as edict
import open3d as o3d
import pandas as pd
# import cupy as cp
from scipy import io
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3' 
from PIL import Image
from multiprocessing import Pool
import pypatchworkpp
import cv2
import math
import radar_preprocessing

COLO_BASE_PATH = "/home/ruibin/Research_code/radar_ws/Coloradar_Dataset_Generation_New/Coloradar_Raw/"
STORE_BASE_PATH = "./COLO_RPD_Dataset/"

MAX_RANGE = 16.0
IMAGE_SHAPE = 160
RANGE_RESOLUTION = 0.125
RESIZE_SHAPE = 128

def pcl_to_cartesian_image (lidar_pcl, save_path):
    X_pixel = int((MAX_RANGE - 0)/ RANGE_RESOLUTION)
    Y_pixel = int((MAX_RANGE - (-MAX_RANGE)) / (RANGE_RESOLUTION * 2))
    lidar_bev_image = np.zeros((X_pixel, Y_pixel))

    x_grid = np.linspace(0, MAX_RANGE, X_pixel + 1)
    # print("x_grid", x_grid)
    y_grid = np.linspace(-MAX_RANGE, MAX_RANGE, Y_pixel + 1)

    if lidar_pcl.shape[0] == 0:
        im = Image.fromarray(np.fliplr((np.flipud(lidar_bev_image)*255)).astype(np.uint8))
        im = im.resize((IMAGE_SHAPE * 2, IMAGE_SHAPE))
        im.save(save_path)
        return 
    
    x = lidar_pcl[:, 0]
    y = lidar_pcl[:, 1]

    for i in range(lidar_pcl.shape[0]):
        x_diff = np.abs(x_grid - x[i])
        y_diff = np.abs(y_grid - y[i])
        # print("y_diff", y_grid - y[i])
        x_i = np.argmin(x_diff)
        y_i = np.argmin(y_diff)

        if(x_i >= lidar_bev_image.shape[0]) or (y_i >= lidar_bev_image.shape[1]):
            continue 
        lidar_bev_image[x_i,y_i] = 1
    # print("lidar_bev_image", lidar_bev_image.shape)
    im = Image.fromarray(np.fliplr((np.flipud(lidar_bev_image)*255)).astype(np.uint8))
    im = im.resize((IMAGE_SHAPE * 2, IMAGE_SHAPE))
    im.save(save_path)
    # return im
    
def load_radar_data(radar_config, radar_index, coloradar_path):
    radar_base_path = COLO_BASE_PATH + coloradar_path + "/single_chip/adc_samples/data/frame_" + str(radar_index) + ".bin"
    loaded_data = np.fromfile(radar_base_path,dtype = "int16")
    print("loaded_data", loaded_data.shape)
    print(radar_config.numTxChan, radar_config.numRxChan,radar_config.numChirpsPerFrame, radar_config.numAdcSamples)
    loaded_data_numpy_adc = loaded_data.reshape((radar_config.numTxChan, radar_config.numRxChan, 
                                                 radar_config.numChirpsPerFrame, radar_config.numAdcSamples, 2))
    print(loaded_data_numpy_adc.shape)
    print("loaded_data_numpy_adc", type(loaded_data_numpy_adc))
    
    I = loaded_data_numpy_adc[:, :, :, :, 0]
    Q = loaded_data_numpy_adc[:, :, :, :, 1]
    loaded_data_numpy_adc= I + 1j * Q
    # print("loaded_data_numpy_adc", type(loaded_data_numpy_adc))
    
    # loaded_data_numpy_adc = np.transpose(loaded_data_numpy_adc, (3, 2, 1, 0))  ###R, D, Rx, TX
    # print("loaded_data_numpy_adc", loaded_data_numpy_adc.shape)
    return loaded_data_numpy_adc



def load_lidar_data(lidar_index, coloradar_path):
    lidar_base_path = COLO_BASE_PATH + coloradar_path + "/lidar/pointclouds/lidar_pointcloud_" + str(lidar_index) + ".bin"
    pcd = np.fromfile(lidar_base_path, np.float32)
    pcd = np.reshape(pcd, (-1, 4))  ## x, y, z, intensity
    # print("pcd", pcd.shape)

    #coordinate!

    x = pcd[:, 0]
    y = pcd[:, 1]
    z = pcd[:, 2]
    intensity = pcd[:, 3]
    x = -x  ##coordinate fixing!!  warning zrb!!!
    y = -y

    pcd = np.array([x, y, z, intensity])
    pcd = np.transpose(pcd, (1, 0))
    # print("pcd", pcd.shape)
    pcd_bev = pcd[:, :2]
    pcd_3d = pcd[:, :3]


    # print("pcd_bev", pcd_bev.shape)
    return pcd, pcd_3d, pcd_bev


def remove_ceiling(point_cloud,  percentage = 10.0, divide_num = 3.0, ) -> np.array:

    sorted_indices = np.argsort(point_cloud[:, 2])
    sorted_point_cloud = point_cloud[sorted_indices]

 
    median_height = np.median(sorted_point_cloud[:, 2])
    # max_height = np.max(sorted_point_cloud[:, 2])
    max_height =  np.percentile(sorted_point_cloud[:, 2], 100 - percentage)
    min_height =  np.percentile(sorted_point_cloud[:, 2], percentage)


    ceiling_threshold = (max_height - min_height) / divide_num + min_height


    ceiling_indices = np.where(sorted_point_cloud[:, 2] > ceiling_threshold)[0] 

    filtered_point_cloud = np.delete(sorted_point_cloud, ceiling_indices, axis=0)

    return filtered_point_cloud




def lidar_preprocessing(lidar_pcd, radar_config, remove_ceiling_flag: bool = True, num_points_target: int = 4000, voxel_size: float = 0.02, num_neighbors: int = 50, radius: float = 1.0, std_dev_factor: float = 0.5, ):
    #filtering points out of FOV
    #Y
    #
    #
    #
    #########X
    angles_DOA_az = radar_config.angles_DOA_az
    angles_DOA_ele = radar_config.angles_DOA_ele
    max_range = radar_config.max_range
    filtered_pcd_list = []

    x = lidar_pcd[:, 0]
    y = lidar_pcd[:, 1]
    z = lidar_pcd[:, 2]

    for i in range(lidar_pcd.shape[0]):

        if (x[i] == 0) or (y[i] == 0):
            continue

        if math.sqrt(x[i]*x[i] + y[i]*y[i] + z[i]*z[i]) > max_range:
            continue

        if x[i] < 0:   ##behind the radar
            continue 

        if (((math.atan(z[i] / math.sqrt(x[i]*x[i] + y[i]*y[i]))) *  180.0 / math.pi) > angles_DOA_ele[1]) \
           or (((math.atan(z[i] / math.sqrt(x[i]*x[i] + y[i]*y[i]))) *  180.0 / math.pi) < angles_DOA_ele[0]):
            print(((math.atan(z[i] / math.sqrt(x[i]*x[i] + y[i]*y[i]))) *  180.0 / math.pi) )
            continue
        # print("lidar_pcd[i, :]", lidar_pcd[i, :])
        filtered_pcd_list.append(lidar_pcd[i, :])


    filtered_pcd= np.concatenate(filtered_pcd_list)
    fov_filtered_pcd = np.reshape(filtered_pcd, (-1, 3))  ## x, y, z, intensity

    zmax = np.max(z)
    zmin = np.min(z)


    params = pypatchworkpp.Parameters()
    params.verbose = True

    PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)

    # Load point cloud
    # pointcloud = read_bin(input_cloud_filepath)

    # Estimate Ground
    PatchworkPLUSPLUS.estimateGround(fov_filtered_pcd)

    # Get Ground and Nonground
    ground      = PatchworkPLUSPLUS.getGround()
    non_ground_cloud   = PatchworkPLUSPLUS.getNonground()

    if remove_ceiling_flag == True:
        print("remove ceiling!")
        non_ground_cloud = remove_ceiling(non_ground_cloud)


    #filtering cluttered points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(non_ground_cloud)  # Only use the first 3 columns (x, y, z coordinates)

    denoised_pcd = non_ground_cloud


    return lidar_pcd, fov_filtered_pcd, non_ground_cloud, denoised_pcd

def save_pcl_to_mesh(pointcloud, mesh_name):

    # 创建Open3D的PointCloud对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pointcloud)

    # 保存点云为一个.ply文件，这是一种常见的3D文件格式
    o3d.io.write_point_cloud(mesh_name, point_cloud)

def radar_filtering_by_lidar(radar_pcd_3d, lidar_pcd_3d, threshold = 0.5):  # for RPDNet_benchmark
    radar_pcd_3d_wo_index = radar_pcd_3d[:, :3]
    distances = np.linalg.norm(radar_pcd_3d_wo_index[:, np.newaxis] - lidar_pcd_3d, axis=-1)

    # 找到距离小于阈值的点的索引
    filtered_indices = np.any(distances < threshold, axis=1)

    # 使用索引筛选点云 A 中的点
    filtered_radar_pcl = radar_pcd_3d[filtered_indices]

    # 打印过滤后的点云 A 的形状
    # print(filtered_radar_pcl.shape)

    return filtered_radar_pcl

def antenna_array(file_path):
    rxl = []    # RX layout
    txl = []    # TX layout
    with open(os.path.join(file_path), "r") as fh:
        for line in fh:
            if line.startswith("# "):
                continue
            else:
                chunks = line.strip().split(" ")
                if chunks[0] == "rx":
                    rxl.append([int(x) for x in chunks[1:]])
                elif chunks[0] == "tx":
                    txl.append([int(x) for x in chunks[1:]])
                else:
                    continue

    txl = np.array(txl)
    rxl = np.array(rxl)
    return txl, rxl

def map_gray_to_rgb(ramap_numpy, cmap='jet'):

    cmap_array = plt.get_cmap(cmap)(np.arange(256))[:, :3] * 255
    rgb_numpy = cmap_array[ramap_numpy.astype(np.uint8)]
    rgb_image = Image.fromarray(rgb_numpy, mode='RGB')
    return rgb_image


def save_range_azimuth_image(ramap, radar_config, save_path):
    min_value = np.min(ramap)
    max_value = np.max(ramap)

    ramap = (ramap - min_value) * (255 / (max_value - min_value))
    ramap= np.clip(ramap, 0, 255)
    # print("dpcl_trans", dpcl_trans.shape)
    # ramap= np.clip(ramap, 0, 255)
    ramap_numpy = (ramap).astype(np.uint8)

    ramap_image = Image.fromarray(ramap_numpy, mode='L')

    ramap_image = ramap_image.resize((radar_config.angle_fftsize * 2, radar_config.range_fftsize))
    ramap_image = ramap_image.rotate(180)

    # img2 = cv2.cvtColor(ramap_image,cv2.COLOR_BGR2RGB)

    ramap_image.save(save_path)

    # ramap_image_colar = map_gray_to_rgb(ramap_numpy / 255.0)
    # ramap_image = ramap_image_colar.resize((radar_config.range_fftsize, 128))
    # ramap_image = ramap_image.rotate(180)
    # ramap_image.save(save_path)


def save_lidar_cartesian_image(lidar_pcl, radar_config, save_path):
    max_range = math.ceil(radar_config.max_range)
    X_pixel = int((max_range - 0)/ radar_config.lidar_resolution)
    Y_pixel = int((max_range - (-max_range)) / radar_config.lidar_resolution)
    # print("Y_pixel", Y_pixel)
    lidar_bev_image = np.zeros((X_pixel, Y_pixel))

    x_grid = np.linspace(0, max_range, X_pixel + 1)
    # print("x_grid", x_grid)
    y_grid = np.linspace(-max_range, max_range, Y_pixel + 1)

    x = lidar_pcl[:, 0]
    y = lidar_pcl[:, 1]

    for i in range(lidar_pcl.shape[0]):
        x_diff = np.abs(x_grid - x[i])
        y_diff = np.abs(y_grid - y[i])
        # print("y_diff", y_grid - y[i])
        x_i = np.argmin(x_diff)
        y_i = np.argmin(y_diff)

        if(x_i >= lidar_bev_image.shape[0]) or (y_i >= lidar_bev_image.shape[1]):
            continue 
        lidar_bev_image[x_i,y_i] = 1

    im = Image.fromarray(np.fliplr((np.flipud(lidar_bev_image)*255)).astype(np.uint8))

    im.save(save_path)


def save_lidar_polar_image(lidar_pcl, radar_config, save_path):
    max_range = math.ceil(radar_config.max_range)
    R_pixel = int((max_range - 0)/ radar_config.lidar_resolution)
    Azi_pixel = radar_config.lidar_polar_azimuth_bins
    polar_image = np.zeros((R_pixel, Azi_pixel))
    polar_data = np.zeros(lidar_pcl.shape)
    #pcl to polar
    for i in range(lidar_pcl.shape[0]):
        xi = lidar_pcl[i,0]
        yi = lidar_pcl[i,1]
        ri = np.sqrt(xi**2+yi**2)
        ai = np.rad2deg(np.arctan2(yi,xi))

        polar_data[i,0] = ri
        polar_data[i,1] = ai

    r_grid = np.linspace(0, max_range, R_pixel + 1)
    azi_grid = np.linspace(-90, 90, Azi_pixel)
    # print("r_grid", r_grid)
    r = polar_data[:,0]
    a = polar_data[:,1]
    # print("a", a)

    for i in range(polar_data.shape[0]):
        ri = np.argmax(r_grid>=r[i])
        ai = np.argmax(azi_grid>=a[i])
        if (ri < polar_image.shape[0]) and (ai < polar_image.shape[1]): 
            polar_image[ri,ai] = 1

    # print("polar_image", polar_image.shape)

    im = Image.fromarray((polar_image*255).astype(np.uint8))
    im = im.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    im.save(save_path)


def process_task (coloradar_path) -> None:

    
    config_file_path = "./config/1843_coloradar.yml"
    with open(config_file_path, 'r', encoding="utf-8") as fid:
        radar_config = edict(yaml.load(fid, Loader=yaml.FullLoader))
    radar_config.chirpRampTime = radar_config.SamplePerChripUp / radar_config.Fs
    radar_config.chirpBandwidth = radar_config.Kr * radar_config.chirpRampTime
    radar_config.max_range = (3e8 * radar_config.chirpRampTime * radar_config.Fs) / (2 * radar_config.chirpBandwidth )
    
    antenna_file_path =  "./config/antenna_array.txt"
    tx_array, rx_array = antenna_array(antenna_file_path)

    #for RPDNet benchmark
    # all_rdm_save_path = str(STORE_BASE_PATH + coloradar_path + "/range_doppler_heatmap_all/")
    # rdm_save_path = str(STORE_BASE_PATH + coloradar_path + "/range_doppler_mean/")
    # gt_save_path = str(STORE_BASE_PATH + coloradar_path + "/range_doppler_label/")
    # radar_ram_colar_save_pth = str(STORE_BASE_PATH + coloradar_path + "/range_azimuth_color_heatmap/")
    # radar_pcl_filtered_mesh_pth = str(STORE_BASE_PATH + coloradar_path + "/radar_pcl_filtered_mesh/")

    pcl_save_pth = str(STORE_BASE_PATH + coloradar_path + "/lidar_pcl/")
    pcl_mesh_pth = str(STORE_BASE_PATH + coloradar_path + "/lidar_mesh/")
    radar_cfar_pcl_pth = str(STORE_BASE_PATH + coloradar_path + "/pcl_npy/")
    pcl_bev_save_pth = str(STORE_BASE_PATH + coloradar_path + "/lidar_pcl_bev/")
    pcl_bev_img_save_pth = str(STORE_BASE_PATH + coloradar_path + "/lidar_pcl_bev_img/")
    pcl_bev_polar_img_save_pth = str(STORE_BASE_PATH + coloradar_path + "/lidar_pcl_bev_polar_img/")
    radar_ram_save_pth = str(STORE_BASE_PATH + coloradar_path + "/range_azimuth_heatmap/")
    radar_pcl_mesh_pth = str(STORE_BASE_PATH + coloradar_path + "/radar_pcl_mesh/")
    radar_pcl_cartesian_image_pth = str(STORE_BASE_PATH + coloradar_path + "/radar_pcl_cartesian_image/")
    

    if not os.path.exists(STORE_BASE_PATH):
        os.mkdir(STORE_BASE_PATH)
    if not os.path.exists(STORE_BASE_PATH + coloradar_path):
        os.mkdir(STORE_BASE_PATH + coloradar_path)
    if not os.path.exists(STORE_BASE_PATH + coloradar_path):
        os.mkdir(STORE_BASE_PATH + coloradar_path)
    if not os.path.exists(radar_cfar_pcl_pth):
        os.mkdir(radar_cfar_pcl_pth)
    if not os.path.exists(pcl_save_pth):
        os.mkdir(pcl_save_pth)
    if not os.path.exists(pcl_mesh_pth):
        os.mkdir(pcl_mesh_pth)
    if not os.path.exists(pcl_bev_save_pth):
        os.mkdir(pcl_bev_save_pth)
    if not os.path.exists(pcl_bev_img_save_pth):
        os.mkdir(pcl_bev_img_save_pth)
    if not os.path.exists(pcl_bev_polar_img_save_pth):
        os.mkdir(pcl_bev_polar_img_save_pth)
    if not os.path.exists(radar_ram_save_pth):
        os.mkdir(radar_ram_save_pth)
    if not os.path.exists(radar_pcl_mesh_pth):
        os.mkdir(radar_pcl_mesh_pth)
    if not os.path.exists(radar_pcl_cartesian_image_pth):
        os.mkdir(radar_pcl_cartesian_image_pth)


    radar_timestamp_index_path = COLO_BASE_PATH + coloradar_path + "/single_chip/adc_samples/radar_index_sequence.txt"
    lidar_timestamp_index_path = COLO_BASE_PATH + coloradar_path + "/lidar/lidar_index_sequence.txt"

    with open(radar_timestamp_index_path, 'r') as file:
        lines = file.readlines() 
        numbers = [int(line.strip()) for line in lines]  
        radar_sorted_index_list = sorted(numbers)  

    with open(lidar_timestamp_index_path, 'r') as file:
        lines = file.readlines()  
        numbers = [int(line.strip()) for line in lines]  
        lidar_sorted_index_list = sorted(numbers)  

    for idx in range(len(radar_sorted_index_list)):

        radar_index = radar_sorted_index_list[idx]
        lidar_index = lidar_sorted_index_list[idx]


        print("coloradar_path", coloradar_path)
        print("index_value", (radar_index, lidar_index))


        radar_adc = load_radar_data(radar_config, radar_index, coloradar_path)  #[128, 128, 4, 3]
        ########1. lidar preprocessing and data saving########
        lidar_pcd, lidar_pcd_3d, lidar_pcd_bev = load_lidar_data(lidar_index, coloradar_path)  #[65536, 3]

        lidar_pcd, fov_filtered_pcd, non_ground_cloud, denoised_pcd = lidar_preprocessing(lidar_pcd_3d, radar_config)
        save_pcl_to_mesh(denoised_pcd, pcl_mesh_pth + str(idx) + ".ply")
        lidar_pcd_filtered_bev = denoised_pcd[:, :2]
        np.save(pcl_save_pth + str(idx) + '.npy', denoised_pcd)
        np.save(pcl_bev_save_pth + str(idx) + '.npy', lidar_pcd_filtered_bev)

        save_lidar_cartesian_image(lidar_pcd_filtered_bev, radar_config, pcl_bev_img_save_pth + str(idx) + '.png')
        save_lidar_polar_image(lidar_pcd_filtered_bev, radar_config, pcl_bev_polar_img_save_pth + str(idx) + '.png')
        ##################################################

        ######## 2. radar range-azimuth heatmap generation and saving########
        rad_map = radar_preprocessing.RAmap(radar_adc, radar_config, tx_array, rx_array)
        save_range_azimuth_image(rad_map, radar_config, radar_ram_save_pth + str(idx) + '.png')
        ##################################################

        ######## 3. radar cfar_pcl generation and saving########
        xyz_ticode, dfft_all, dfft_sum = radar_preprocessing.adc2pcd_coloradar(radar_adc, radar_config, tx_array, rx_array) #TX, RX, V, R 
        radar_pcd_x = xyz_ticode[:, 0]
        radar_pcd_y = xyz_ticode[:, 1]
        radar_pcd_z = xyz_ticode[:, 2]
        radar_range_index = xyz_ticode[:, 5]
        radar_doppler_index = xyz_ticode[:, 6]

        radar_pcd_3d_coordinated_transformed = np.array([radar_pcd_y, -radar_pcd_x, radar_pcd_z, radar_range_index, radar_doppler_index])
        radar_pcd_3d_coordinated_transformed = np.transpose(radar_pcd_3d_coordinated_transformed, (1, 0))

        save_pcl_to_mesh(radar_pcd_3d_coordinated_transformed[:, :3], radar_pcl_mesh_pth + str(idx) + ".ply")
        np.save(radar_cfar_pcl_pth + str(idx) + '.npy', radar_pcd_3d_coordinated_transformed[:, :3])

        cartesian_save_path = radar_pcl_cartesian_image_pth + str(idx) +'.png'
        pcl_to_cartesian_image(radar_pcd_3d_coordinated_transformed[:, :3], cartesian_save_path)        
        ####################################################







    

if __name__ == '__main__':
    num_tasks = 16  


    tasks = [
            # "12_21_2020_arpg_lab_run0",
            # "12_21_2020_arpg_lab_run1",
            # "12_21_2020_arpg_lab_run2",
            # "12_21_2020_arpg_lab_run3",
            # "12_21_2020_arpg_lab_run4",
            "12_21_2020_ec_hallways_run0",
            #  "12_21_2020_ec_hallways_run1",
            #  "12_21_2020_ec_hallways_run2",
            #  "12_21_2020_ec_hallways_run3",
            #  "12_21_2020_ec_hallways_run4",
            #  "12_21_2020_ec_hallways_run5",
            #  "2_23_2021_edgar_army_run0",
            #  "2_23_2021_edgar_army_run1",
            #  "2_23_2021_edgar_army_run2",
            #  "2_23_2021_edgar_army_run3",
            #  "2_23_2021_edgar_army_run4",
            #  "2_23_2021_edgar_army_run5",
            #  "2_23_2021_edgar_classroom_run0",
            #  "2_23_2021_edgar_classroom_run1",
            #  "2_23_2021_edgar_classroom_run2",
            #  "2_23_2021_edgar_classroom_run3",
            #  "2_23_2021_edgar_classroom_run4",
            #  "2_23_2021_edgar_classroom_run5",

            #  "2_24_2021_aspen_run0",
            #  "2_24_2021_aspen_run1",
            #  "2_24_2021_aspen_run2",
            #  "2_24_2021_aspen_run3",
            #  "2_24_2021_aspen_run4",
            #  "2_24_2021_aspen_run5",
            #  "2_24_2021_aspen_run6",
            #  "2_24_2021_aspen_run7",
            #  "2_24_2021_aspen_run8",
            #  "2_24_2021_aspen_run9",
            #  "2_24_2021_aspen_run10",
            #  "2_24_2021_aspen_run11",

            # "2_28_2021_outdoors_run0",
            # "2_28_2021_outdoors_run1",
            # "2_28_2021_outdoors_run2",
            # "2_28_2021_outdoors_run3",
            # "2_28_2021_outdoors_run4",
            # "2_28_2021_outdoors_run5",
            # "2_28_2021_outdoors_run6",
            # "2_28_2021_outdoors_run7",
            # "2_28_2021_outdoors_run8",
            # "2_28_2021_outdoors_run9",
            #  ]
            # "2_22_2021_longboard_run0",
            # "2_22_2021_longboard_run1",
            # "2_22_2021_longboard_run2",
            # "2_22_2021_longboard_run3",
            # "2_22_2021_longboard_run4",
            # "2_22_2021_longboard_run5",
            # "2_22_2021_longboard_run6",
            # "2_22_2021_longboard_run7",

            # "12_21_2020_ec_hallways_run1",
            # "12_21_2020_ec_hallways_run2",
            # "12_21_2020_ec_hallways_run3",
            # "12_21_2020_ec_hallways_run4",
        
            # "12_21_2020_arpg_lab_run0",
            # "12_21_2020_arpg_lab_run1",
            # "12_21_2020_arpg_lab_run2",
            # "12_21_2020_arpg_lab_run3",
            # "12_21_2020_arpg_lab_run4",
            ]

    num_tasks = 2  
    num_processes = 16  #

    with Pool(processes=num_processes) as pool:
        results = pool.map(process_task, tasks)
