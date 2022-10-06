# --------------------------------------------------------
# DenseFusion 6D Object Pose Estimation by Iterative Dense Fusion
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen
# --------------------------------------------------------

import _init_paths
import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets.ycb_nvidia_fat.dataset import PoseDataset as PoseDataset_ycb
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.utils import setup_logger


class LF:
    def __init__(self) -> None:

        self.dataset = "ycb_label_fusion"
        self.dataset_root = ""
        self.batch_size = 8
        self.workers = 10
        self.lr = 0.0001
        self.lr_decay = 0.3
        self.w = 0.015
        self.w_decay = 0.3
        self.decay_margin = 0.016
        self.refine_margin = 0.013
        self.noise_trans = 0.03
        self.iteration = 2
        self.nepoch = 500
        self.resume_posenet = ""  # for resume training
        self.resume_refinenet = ""  # for resume training
        self.start_epoch = 1
        self.seed = random.randint(1, 10000)
        self.num_objects = 21
        self.num_points = 1000
        self.out_dir = "trained_models/ycb_nvidia_fat"
        self.log_dir = "experiments/logs/ycb_nvidia_fat"
        self.repeat_epoch = 1


class FaT:
    def __init__(self) -> None:

        self.dataset = "ycb_nvidia_fat"
        self.dataset_root = "/media/qind/Data/QIND-DATA/fat"
        self.batch_size = 1
        self.workers = 1
        self.lr = 0.0001
        self.lr_decay = 0.3
        self.w = 0.015
        self.w_decay = 0.3
        self.decay_margin = 0.016
        self.refine_margin = 0.013
        self.noise_trans = 0.03
        self.iteration = 2
        self.nepoch = 500
        self.resume_posenet = ""  # for resume training
        self.resume_refinenet = ""  # for resume training
        self.start_epoch = 1
        self.seed = 1506
        self.num_objects = 21
        self.num_points = 1000
        self.out_dir = "trained_models/ycb_nvidia_fat"
        self.log_dir = "experiments/logs/ycb_nvidia_fat"
        self.repeat_epoch = 1


class CustomDataset:
    def __init__(self) -> None:

        self.dataset = "custom_dataset"
        self.dataset_root = ""
        self.batch_size = 8
        self.workers = 10
        self.lr = 0.0001
        self.lr_decay = 0.3
        self.w = 0.015
        self.w_decay = 0.3
        self.decay_margin = 0.016
        self.refine_margin = 0.013
        self.noise_trans = 0.03
        self.iteration = 2
        self.nepoch = 500
        self.resume_posenet = ""  # for resume training
        self.resume_refinenet = ""  # for resume training
        self.start_epoch = 1
        self.seed = random.randint(1, 10000)
        self.num_objects = 21
        self.num_points = 1000
        self.out_dir = "trained_models/ycb_nvidia_fat"
        self.log_dir = "experiments/logs/ycb_nvidia_fat"
        self.repeat_epoch = 1


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    setup_dataset = "ycb_nvidia_fat"
    if setup_dataset == "ycb_nvidia_fat":
        para = FaT()
    elif setup_dataset == "ycb_label_fusion":
        para = LF()
    elif setup_dataset == "custom_dataset":
        para = CustomDataset()

    manualSeed = para.seed
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    # pose estimator net
    estimator = PoseNet(num_points=para.num_points, num_obj=para.num_objects)
    estimator.to(device)

    # Pose-Refine net
    refiner = PoseRefineNet(num_points=para.num_points, num_obj=para.num_objects)
    refiner.to(device)

    if para.resume_posenet != "":
        estimator.load_state_dict(
            torch.load("{0}/{1}".format(para.out_dir, para.resume_posenet))
        )

    if para.resume_refinenet != "":
        refiner.load_state_dict(
            torch.load("{0}/{1}".format(para.out_dir, para.resume_refinenet))
        )
        para.refine_start = True
        para.decay_start = True
        para.lr *= para.lr_decay
        para.w *= para.w_decay
        para.batch_size = int(para.batch_size / para.iteration)
        optimizer = optim.Adam(refiner.parameters(), lr=para.lr)
    else:
        para.refine_start = False
        para.decay_start = False
        optimizer = optim.Adam(estimator.parameters(), lr=para.lr)

    if para.dataset == "ycb_nvidia_fat":
        dataset = PoseDataset_ycb(
            "train",
            para.num_points,
            True,
            para.dataset_root,
            para.noise_trans,
            para.refine_start,
        )
        test_dataset = PoseDataset_ycb(
            "test", para.num_points, False, para.dataset_root, 0.0, para.refine_start
        )
    elif para.dataset == "ycb_label_fusion":
        dataset = PoseDataset_ycb(
            "train",
            para.num_points,
            True,
            para.dataset_root,
            para.noise_trans,
            para.refine_start,
        )
        test_dataset = PoseDataset_ycb(
            "test", para.num_points, False, para.dataset_root, 0.0, para.refine_start
        )

    elif para.dataset == "custom_dataset":
        dataset = PoseDataset_ycb(
            "train",
            para.num_points,
            True,
            para.dataset_root,
            para.noise_trans,
            para.refine_start,
        )
        test_dataset = PoseDataset_ycb(
            "test", para.num_points, False, para.dataset_root, 0.0, para.refine_start
        )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=para.workers
    )
    testdataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=para.workers
    )

    para.sym_list = dataset.get_sym_list()
    para.num_points_mesh = dataset.get_num_points_mesh()

    print(
        ">>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}".format(
            len(dataset), len(test_dataset), para.num_points_mesh, para.sym_list
        )
    )

    criterion = Loss(para.num_points_mesh, para.sym_list)
    criterion_refine = Loss_refine(para.num_points_mesh, para.sym_list)

    best_test = np.Inf

    if para.start_epoch == 1:
        for log in os.listdir(para.log_dir):
            os.remove(os.path.join(para.log_dir, log))
    st_time = time.time()
    
    for epoch in range(para.start_epoch, para.nepoch):
        logger = setup_logger(
            "epoch%d" % epoch, os.path.join(para.log_dir, "epoch_%d_log.txt" % epoch)
        )
        logger.info(
            "Train time {0}".format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time))
                + ", "
                + "Training started"
            )
        )
        train_count = 0
        train_dis_avg = 0.0
        if para.refine_start:
            estimator.eval()
            refiner.train()
        else:
            estimator.train()
        # reset optimizer for a new training loop
        optimizer.zero_grad()

        for rep in range(para.repeat_epoch):
            for i, data in enumerate(dataloader, 0):
                points, choose, img, target, model_points, idx = data
                points, choose, img, target, model_points, idx = (
                    Variable(points).to(device),
                    Variable(choose).to(device),
                    Variable(img).to(device),
                    Variable(target).to(device),
                    Variable(model_points).to(device),
                    Variable(idx).to(device),
                )
                with open('var.txt','w') as f:
                    r1 = str(points.size())
                    r = str(points)
                    t1 = str(choose.size())
                    t = str(choose)
                    c1 = str(img.size())
                    c = str(img)
                    e1 = str(target.size())
                    e = str(target)
                    a1 = str(model_points.size())
                    a = str(model_points)
                    g1 = str(idx.size())
                    g = str(idx)
                    f.write(r1+'\n')
                    f.write(r+'\n')
                    f.write(t1+'\n')
                    f.write(t+'\n')
                    f.write(c1+'\n')
                    f.write(c+'\n')
                    f.write(e1+'\n')
                    f.write(e+'\n')
                    f.write(a1+'\n')
                    f.write(a+'\n')
                    f.write(g1+'\n')
                    f.write(g+'\n')
                    f.close
                
                pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
                with open('abc.txt','w') as f:
                    r = str(pred_r)
                    t = str(pred_t)
                    c = str(pred_c)
                    e = str(emb)
                    f.write(r+'\n')

                    f.write(t+'\n')
                    f.write(c+'\n')
                    f.write(e+'\n')
                    f.close
                loss, dis, new_points, new_target = criterion(
                    pred_r,
                    pred_t,
                    pred_c,
                    target,
                    model_points,
                    idx,
                    points,
                    para.w,
                    para.refine_start,
                )
                print('>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<')
                if para.refine_start:
                    for ite in range(0, para.iteration):
                        pred_r, pred_t = refiner(new_points, emb, idx)
                        dis, new_points, new_target = criterion_refine(
                            pred_r, pred_t, new_target, model_points, idx, new_points
                        )
                        dis.backward()
                else:
                    loss.backward()

                train_dis_avg += dis.item()
                train_count += 1

                if train_count % para.batch_size == 0:
                    logger.info(
                        "Train time {0} Epoch {1} Batch {2} Frame {3} Avg_dis:{4}".format(
                            time.strftime(
                                "%Hh %Mm %Ss", time.gmtime(time.time() - st_time)
                            ),
                            epoch,
                            int(train_count / para.batch_size),
                            train_count,
                            train_dis_avg / para.batch_size,
                        )
                    )
                    optimizer.step()
                    optimizer.zero_grad()
                    train_dis_avg = 0

                if train_count != 0 and train_count % 1000 == 0:
                    if para.refine_start:
                        torch.save(
                            refiner.state_dict(),
                            "{0}/pose_refine_model_current.pth".format(para.out_dir),
                        )
                    else:
                        torch.save(
                            estimator.state_dict(),
                            "{0}/pose_model_current.pth".format(para.out_dir),
                        )

        print(">>>>>>>>----------epoch {0} train finish---------<<<<<<<<".format(epoch))

        logger = setup_logger(
            "epoch%d_test" % epoch,
            os.path.join(para.log_dir, "epoch_%d_test_log.txt" % epoch),
        )
        logger.info(
            "Test time {0}".format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time))
                + ", "
                + "Testing started"
            )
        )
        test_dis = 0.0
        test_count = 0
        estimator.eval()
        refiner.eval()

        for j, data in enumerate(testdataloader, 0):
            points, choose, img, target, model_points, idx = data
            points, choose, img, target, model_points, idx = (
                Variable(points).to(device),
                Variable(choose).to(device),
                Variable(img).to(device),
                Variable(target).to(device),
                Variable(model_points).to(device),
                Variable(idx).to(device),
            )
            pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
            _, dis, new_points, new_target = criterion(
                pred_r,
                pred_t,
                pred_c,
                target,
                model_points,
                idx,
                points,
                para.w,
                para.refine_start,
            )

            if para.refine_start:
                for ite in range(0, para.iteration):
                    pred_r, pred_t = refiner(new_points, emb, idx)
                    dis, new_points, new_target = criterion_refine(
                        pred_r, pred_t, new_target, model_points, idx, new_points
                    )

            test_dis += dis.item()
            logger.info(
                "Test time {0} Test Frame No.{1} dis:{2}".format(
                    time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)),
                    test_count,
                    dis,
                )
            )

            test_count += 1

        test_dis = test_dis / test_count
        logger.info(
            "Test time {0} Epoch {1} TEST FINISH Avg dis: {2}".format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)),
                epoch,
                test_dis,
            )
        )
        if test_dis <= best_test:
            best_test = test_dis
            if para.refine_start:
                torch.save(
                    refiner.state_dict(),
                    "{0}/pose_refine_model_{1}_{2}.pth".format(
                        para.out_dir, epoch, test_dis
                    ),
                )
            else:
                torch.save(
                    estimator.state_dict(),
                    "{0}/pose_model_{1}_{2}.pth".format(para.out_dir, epoch, test_dis),
                )
            print(epoch, ">>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<")

        if best_test < para.decay_margin and not para.decay_start:
            para.decay_start = True
            para.lr *= para.lr_rate
            para.w *= para.w_rate
            optimizer = optim.Adam(estimator.parameters(), lr=para.lr)

        if best_test < para.refine_margin and not para.refine_start:
            para.refine_start = True
            para.batch_size = int(para.batch_size / para.iteration)
            optimizer = optim.Adam(refiner.parameters(), lr=para.lr)

            if para.dataset == "ycb_nvidia_fat":
                dataset = PoseDataset_ycb(
                    "train",
                    para.num_points,
                    True,
                    para.dataset_root,
                    para.noise_trans,
                    para.refine_start,
                )
                test_dataset = PoseDataset_ycb(
                    "test",
                    para.num_points,
                    False,
                    para.dataset_root,
                    0.0,
                    para.refine_start,
                )
            elif para.dataset == "ycb_label_fusion":
                dataset = PoseDataset_ycb(
                    "train",
                    para.num_points,
                    True,
                    para.dataset_root,
                    para.noise_trans,
                    para.refine_start,
                )
                test_dataset = PoseDataset_ycb(
                    "test",
                    para.num_points,
                    False,
                    para.dataset_root,
                    0.0,
                    para.refine_start,
                )
            elif para.dataset == "custom_dataset":
                dataset = PoseDataset_ycb(
                    "train",
                    para.num_points,
                    True,
                    para.dataset_root,
                    para.noise_trans,
                    para.refine_start,
                )
                test_dataset = PoseDataset_ycb(
                    "test",
                    para.num_points,
                    False,
                    para.dataset_root,
                    0.0,
                    para.refine_start,
                )

            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                shuffle=True,
                num_workers=para.workers,
            )
            testdataloader = torch.utils.data.DataLoader(
                test_dataset, batch_size=1, shuffle=False, num_workers=para.workers
            )

            para.sym_list = dataset.get_sym_list()
            para.num_points_mesh = dataset.get_num_points_mesh()

            print(
                ">>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}".format(
                    len(dataset), len(test_dataset), para.num_points_mesh, para.sym_list
                )
            )

            criterion = Loss(para.num_points_mesh, para.sym_list)
            criterion_refine = Loss_refine(para.num_points_mesh, para.sym_list)


if __name__ == "__main__":
    main()
