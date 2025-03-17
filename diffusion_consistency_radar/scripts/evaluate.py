import os
import numpy as np
import math
from scipy.spatial import cKDTree

BASE_PATH = "./inference_results/Proposed_EDM/"
SCENE_NAME = "edgar" #edgar, outdoors, arpg_lab, ec_hallways, aspen, longboard


DISTANCE_Threshold = 0.1 #meters

def cumpute_chamfer_distance(GT, Pred):
    kdtree_GT = cKDTree(GT)
    kdtree_Pred = cKDTree(Pred)

    distance_Pred_to_GT, _ = kdtree_GT.query(Pred)
    distance_GT_to_Pred, _ = kdtree_Pred.query(GT)

    chamfer_dist = np.mean(distance_GT_to_Pred) + np.mean(distance_Pred_to_GT) 

    return  chamfer_dist, distance_GT_to_Pred, distance_Pred_to_GT

def evaluate_matches(distance_A_to_B, distance_B_to_A, threshold):
    TP = np.sum(distance_B_to_A <= threshold)
    FN = np.sum(distance_B_to_A > threshold)
    FP = np.sum(distance_A_to_B > threshold)
    TN = 0

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    return TP, FN, FP, TN, precision, recall

def read_inference_data(pcl_np_pred_path, pcl_np_gt_path):
    pred_pc_list = []
    gt_pc_list = []

    for dirpath, dirnames, filenames in os.walk(BASE_PATH):
        for dirname in dirnames:
            if SCENE_NAME in dirname:
                subdir = os.path.join(dirpath, dirname)
                file = os.listdir(subdir + pcl_np_pred_path)
                file.sort(key=lambda x:int(x.split('.')[0]))
                for i in file:
                    path = os.path.join(subdir + pcl_np_pred_path, i)
                    pred_pc = np.load(path)
                    pred_pc_list.append(pred_pc)

                file = os.listdir(subdir + pcl_np_gt_path)
                file.sort(key=lambda x:int(x.split('.')[0]))
                for i in file:
                    path = os.path.join(subdir + pcl_np_gt_path, i)
                    gt_pc = np.load(path)
                    gt_pc_list.append(gt_pc)    

    return pred_pc_list, gt_pc_list
    
def main():
    pcl_np_pred_path =  "/pcl_np/"
    pcl_np_gt_path = "/gt_bev_pcl/"

    pred_pc_list, gt_pc_list = read_inference_data(pcl_np_pred_path, pcl_np_gt_path)
    
    Chamfer_distance, Hausdorff_distance, prediction, recall, fscore = 0, 0, 0, 0, 0

    for i in range(len(pred_pc_list)):
        print("i", i)
        pred_pc_i = pred_pc_list[i]
        gt_pc_i = gt_pc_list[i]


        if pred_pc_i.shape[0] == 0 or gt_pc_i.shape[0] == 0:
            continue

        Chamfer_distance_i, distance_gt_to_pred, distance_pred_to_gt = cumpute_chamfer_distance(gt_pc_i, pred_pc_i)
        Chamfer_distance = Chamfer_distance + Chamfer_distance_i

        Hausdorff_distance = Hausdorff_distance + np.maximum(np.max(distance_gt_to_pred), np.max(distance_pred_to_gt))

        TP_i, FN_i, FP_i, TN_i, precision_i, recall_i = evaluate_matches(distance_gt_to_pred, distance_pred_to_gt, threshold = DISTANCE_Threshold)
        prediction = prediction + precision_i
        recall = recall + recall_i

        if (recall_i + precision_i) > 0:
            fscore = fscore + 2*precision_i*recall_i / (recall_i + precision_i)


    print("Chamfer_distance", Chamfer_distance/len(pred_pc_list))
    print("Hausdorff_distance", Hausdorff_distance/len(pred_pc_list))
    print("F-score", fscore/len(pred_pc_list))        

if __name__ == "__main__":
    main()
