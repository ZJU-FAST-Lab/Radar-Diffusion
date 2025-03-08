"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from cm import dist_util, logger
from cm.script_util_cond import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from cm.random_util import get_generator
from cm.karras_diffusion import karras_sample
from PIL import Image
import torchvision.transforms as transforms
from cm.radarloader_coloradar_benchmark import *
import random

seed = 42
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
import yaml
from easydict import EasyDict as edict

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

import cv2
import open3d as o3d
import math

BASE_STORE_PATH = "./inference_results/"
MAX_RANGE = 16.0
IMAGE_SHAPE = 160
RANGE_RESOLUTION = 0.125

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
    
def polar_image_to_pcl (polar_image):
    width, height = polar_image.shape


    point_cloud = []


    for row in range(height): #angle
        for column in range(width): #range

            if polar_image[column, row] > 64:

                column_true = width - column
                distance = column_true / width * MAX_RANGE
                # print("distance", distance)
                row_true = height - row
                angle = (row_true - height / 2) / (height / 2) * (math.pi / 2)
                # print("angle", angle)
                x = distance * math.cos(angle)
                y = distance * math.sin(angle)

                z = 0

                point_cloud.append([x, y, z])

    pcl = np.array(point_cloud)

    pcl_o3d = o3d.geometry.PointCloud()
    pcl_o3d.points = o3d.utility.Vector3dVector(point_cloud)
    
    return pcl, pcl_o3d

def main():
    args = create_argparser().parse_args()

    # print("args.in_ch", args.in_ch)
    # print("args.out_ch", args.out_ch)
    args.in_ch = 2
    args.out_ch = 1
    dist_util.setup_dist()
    logger.configure()

    if "consistency" in args.training_mode:
        distillation = True
    else:
        distillation = False


    logger.log("creating data loader...")
    
    dataset_config_path = args.dataloading_config  
    with open(dataset_config_path, 'r') as fid:
        coloradar_config = edict(yaml.load(fid, Loader=yaml.FullLoader))
    
    tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
    transform_train = transforms.Compose(tran_list)
    test_data = init_dataset(coloradar_config, args.dataset_dir, transform_train, "test")
    
    datal= th.utils.data.DataLoader(
        test_data,
        num_workers=16,
        batch_size=args.batch_size,
        shuffle=True)

    data = iter(datal)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        distillation=distillation,
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    # model.to("cpu")
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    if args.sampler == "multistep":
        assert len(args.ts) > 0
        ts = tuple(int(x) for x in args.ts.split(","))
    else:
        ts = None

    all_images = []
    all_labels = []
    generator = get_generator(args.generator, args.num_samples, args.seed)

    if not os.path.exists(BASE_STORE_PATH):
        os.mkdir(BASE_STORE_PATH)

    if not os.path.exists(BASE_STORE_PATH + args.output_dir):
        os.mkdir(BASE_STORE_PATH + args.output_dir)
        

    # i = 0

    start = th.cuda.Event(enable_timing=True)
    end = th.cuda.Event(enable_timing=True)
    cnt = 0
    
    
    for test_i, (b, m, path) in enumerate(datal):
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes

        b, m, path = next(data)  #should return an image from the dataloader "data"

        c = th.randn_like(b)
        img = th.cat((b, c), dim=1)     #add a noise channel$

        b = b.to(dist_util.dev())

        radar_condition_dict = {"y": b}

        start.record()
        sample = karras_sample(
            diffusion = diffusion,
            model = model,
            shape = (args.batch_size, 1, args.image_size, args.image_size),
            steps=args.steps,
            model_kwargs=radar_condition_dict,
            device=dist_util.dev(),
            clip_denoised=args.clip_denoised,
            sampler=args.sampler,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            s_churn=args.s_churn,
            s_tmin=args.s_tmin,
            s_tmax=args.s_tmax,
            s_noise=args.s_noise,
            generator=generator,
            ts=ts,
        )

        end.record()
        th.cuda.synchronize()
        print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample

        cnt = cnt + args.batch_size
        print("cnt", cnt)
        for i in range(args.batch_size):
            split_str = path[i].split("_")


            image_id = split_str[-1]


            scene_name = "_".join(split_str[:-1])
            
            save_path = BASE_STORE_PATH + args.output_dir + scene_name
            polar_image_path = save_path + "/pre_polar_image/"
            cartesian_image_path = save_path + "/pre_cartesian_image/"
            pcl_np_path = save_path + "/pre_pcl_np/"
            pcl_mesh_path = save_path + "/pre_pcl_mesh/"
            gt_polar_path = save_path + "/gt_polar_image/"
            gt_bev_path = save_path + "/gt_bev_image/"
            gt_bev_pcl_path = save_path + "/gt_bev_pcl/"



            if not os.path.exists(save_path):
                os.mkdir(save_path)
            if not os.path.exists(cartesian_image_path):
                os.mkdir(cartesian_image_path)
            if not os.path.exists(polar_image_path):
                os.mkdir(polar_image_path)
            if not os.path.exists(pcl_np_path):
                os.mkdir(pcl_np_path)    
            if not os.path.exists(pcl_mesh_path):
                os.mkdir(pcl_mesh_path)   
            if not os.path.exists(gt_polar_path):
                os.mkdir(gt_polar_path)   
            if not os.path.exists(gt_bev_path):
                os.mkdir(gt_bev_path)   
            if not os.path.exists(gt_bev_pcl_path):
                os.mkdir(gt_bev_pcl_path)                   

            #save gt
            gt = (m[i] * 255).clamp(0, 255).to(th.uint8)

            gt_numpy = np.squeeze(gt.cpu().numpy())
            gt_numpy_img = Image.fromarray((gt_numpy).astype(np.uint8))
        

            gt_file_name = gt_polar_path + image_id +'.png'
            gt_numpy_img.save(gt_file_name)


            image_gt = cv2.imread(gt_file_name, cv2.IMREAD_GRAYSCALE)

            pcl_gt, pcl_o3d_gt = polar_image_to_pcl(image_gt)
            
            gt_cartesian_save_path = gt_bev_path + image_id +'.png'
            pcl_to_cartesian_image(pcl_gt, gt_cartesian_save_path)
            np.save(gt_bev_pcl_path + image_id +'.npy', pcl_gt)

            #radar prediction

            sample_i = (sample[i] * 255).clamp(0, 255).to(th.uint8)

            sample_i = sample_i.permute(1, 2, 0)
            sample_i = sample_i.contiguous()

            sample_numpy = np.squeeze(sample_i.cpu().numpy())
            img_numpy_img = Image.fromarray((sample_numpy).astype(np.uint8))

            im1_file_name = polar_image_path + image_id +'.png'
            img_numpy_img.save(im1_file_name)

            #save as pcl and mesh
            image_i = cv2.imread(im1_file_name, cv2.IMREAD_GRAYSCALE)

            pcl, pcl_o3d = polar_image_to_pcl(image_i)
            np.save(pcl_np_path + image_id +'.npy', pcl)


            o3d.io.write_point_cloud(pcl_mesh_path + image_id +'.ply', pcl_o3d)
            
            cartesian_save_path = cartesian_image_path + image_id +'.png'
            pcl_to_cartesian_image(pcl, cartesian_save_path)


    dist.barrier()
    logger.log("sampling complete")



def create_argparser():
    defaults = dict(
        # data_dir="/home/ubuntu/coloradar_ws/consistency_model_coloradar/cmu_dataset/",
        training_mode="edm",
        generator="determ",
        clip_denoised=True,
        num_samples=100,
        batch_size=1,
        sampler="heun",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=40,
        model_path="./checkpoint/consistency_distillation_140K_Step.pt",    #cd
        seed=42,
        ts="",
        in_ch = 2,
        out_ch = 1,
        dataset_dir = './example_batch/',
        dataloading_config = "./config/data_loading_config_example.yml",
        # dataset_dir = '/home/zrb/Mmwave_Dataset/Coloradar/coloradar_after_preprocessing',
        # dataloading_config = "./config/data_loading_config_example.yml",
        output_dir = "", ##edgar, outdoors, arpg_lab, ec_hallways, aspen, longboard
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
