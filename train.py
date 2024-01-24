import os, argparse, time

import numpy as np
import time
import torch
import torch.nn as nn
from torchvision import transforms
import random
import torch.backends.cudnn as cudnn
import scipy
from scipy import stats
from scipy.optimize import curve_fit
from tqdm import tqdm

from models.multimodal_pcqa import MM_PCQAnet
from my_utils.MultimodalDataset import MMDataset
from my_utils.loss import L2RankLoss


def set_rand_seed(seed=1998):                 #定义了一个函数，接收一个参数seed，用作随机数生成的种子
    print("Random Seed: ", seed)
    random.seed(seed)                         #为python内置的随机数生成器，设置一个种子
    np.random.seed(seed)                      #为numpy的随机数生成器。设置一个种子
    torch.manual_seed(seed)                   #为pytorch的随机数生成器设置种子
    torch.cuda.manual_seed(seed)              #为当前运行的cpu设置随机种子
    torch.cuda.manual_seed_all(seed)          #为所有的cpu设置随机种子
    torch.backends.cudnn.benchmark = False    #禁用cuDNN的自动调优(benchmarking)功能。
    #cuDNN是NVIDIA提供的深度神经网络加速库。当benchmark设置为True时，cuDNN会在运行时自动寻找最优的算法来执行操作，
    #这可能会导致运行结果在不同的执行之间有所不同。关闭它可以增加实验的可重复性。
    torch.backends.cudnn.deterministic = True  # fix the random seed

    #确保CUDNN使用确定性的算法，增加实验的重复性


#def logistic_func(X, bayta1, bayta2, bayta3, bayta4):   #定义一个逻辑函数，通常用于回归分析或神经网络中的激活函数
    #logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    #yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    #return yhat     #逻辑函数值
def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat



#bayta1：上限渐近线（asymptote），即曲线的最大值。
#bayta2：下限渐近线，即曲线的最小值。
#bayta3：位置参数，决定了曲线中间部分的位置，通常与曲线的中点有关。
#bayta4：形状参数，决定了曲线从bayta2到bayta1的增长速率，影响曲线的斜率。


def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, \
                        y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)

    return y_output_logistic





def parse_args():
    """Parse input arguments. """
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument('--gpu', help="GPU device id to use [0]", default=0, type=int)
    parser.add_argument('--num_epochs', help='Maximum number of training epochs.', default=50, type=int)
    parser.add_argument('--batch_size', help='Batch size.', default=6, type=int)
    parser.add_argument('--learning_rate', default=0.00005, type=float, help='learning rate in training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--model', default='MM_PCQA', type=str)
    parser.add_argument('--data_dir_2d', default='G:\\database\\wpc2.0\\wpc2.0_projections\\', type=str,
                        help='path to the images')
    parser.add_argument('--data_dir_pc', default='G:\\database\\wpc2.0\\wpc2.0_patch_2048\\', type=str,
                        help='path to the patches')
    parser.add_argument('--patch_length_read', default=6, type=int, help='number of the using patches')
    parser.add_argument('--img_length_read', default=4, type=int, help='number of the using images')
    parser.add_argument('--loss', default='l2rank', type=str)
    parser.add_argument('--database', default='WPC2.0', type=str)
    parser.add_argument('--k_fold_num', default=4, type=int,
                        help='9 for the SJTU-PCQA, 5 for the WPC, 4 for the WPC2.0')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    print(
        '*************************************************************************************************************************')
    args = parse_args()
    set_rand_seed()

    gpu = args.gpu
    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    database = args.database
    patch_length_read = args.patch_length_read
    img_length_read = args.img_length_read
    data_dir_2d = args.data_dir_2d
    data_dir_pc = args.data_dir_pc
    best_all = np.zeros([args.k_fold_num, 4])

    for k_fold_id in range(1, args.k_fold_num + 1):               #进行k-fold交叉验证

        print('The current k_fold_id is ' + str(k_fold_id))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        if database == 'SJTU':
            train_filename_list = 'csvfiles/sjtu_data_info/train_' + str(k_fold_id) + '.csv'
            test_filename_list = 'csvfiles/sjtu_data_info/test_' + str(k_fold_id) + '.csv'
        elif database == 'WPC':
            train_filename_list = 'csvfiles/wpc_data_info/train_' + str(k_fold_id) + '.csv'
            test_filename_list = 'csvfiles/wpc_data_info/test_' + str(k_fold_id) + '.csv'
        elif database == 'WPC2.0':
            train_filename_list = 'csvfiles/wpc2.0_data_info/train_' + str(k_fold_id) + '.csv'
            test_filename_list = 'csvfiles/wpc2.0_data_info/test_' + str(k_fold_id) + '.csv'

        transformations_train = transforms.Compose([transforms.RandomCrop(224), transforms.ToTensor(), \
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225])])
        #对于训练集做了数据转换操作，采用了随机裁剪和数据标准化

        transformations_test = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), \
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                        std=[0.229, 0.224, 0.225])])

    #对于测试集，采用了中心裁剪和数据标准化

        print('Trainging set: ' + train_filename_list)

        # load the network
        if args.model == 'MM_PCQA':
            model = MM_PCQAnet()
            model = model.to(device)
            print('Using model: MM-PCQA')

        if args.loss == 'l2rank':
            criterion = L2RankLoss().to(device)
            print('Using l2rank loss')

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)
        print('Using Adam optimizer, initial learning rate: ' + str(args.learning_rate))

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.9)

        print("Ready to train network")
        print(
            '*************************************************************************************************************************')
        best_test_criterion = -1  # SROCC min
        best = np.zeros(4)

        train_dataset = MMDataset(data_dir_2d=data_dir_2d, data_dir_pc=data_dir_pc, datainfo_path=train_filename_list,
                                  transform=transformations_train)
        test_dataset = MMDataset(data_dir_2d=data_dir_2d, data_dir_pc=data_dir_pc, datainfo_path=test_filename_list,
                                 transform=transformations_test, is_train=False)

        for epoch in range(num_epochs):
            # begin training, during each epoch, the crops and patches are randomly selected for the training set and fixed for the testing set
            # if you want to change the number of images or projections, load the parameters here 'img_length_read = img_length_read, patch_length_read = patch_length_read'
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                                       num_workers=8)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8)

            n_train = len(train_dataset)
            n_test = len(test_dataset)

            model.train()
            start = time.time()
            batch_losses = []
            batch_losses_each_disp = []
            x_output = np.zeros(n_train)
            x_test = np.zeros(n_train)
            for i, (imgs, pc, mos) in tqdm(enumerate(train_loader)):
                imgs = imgs.to(device)
                pc = torch.Tensor(pc.float())
                pc = pc.to(device)
                mos = mos[:, np.newaxis]
                mos = mos.to(device)
                mos_output = model(imgs, pc)

                # compute loss
                loss = criterion(mos_output, mos)
                batch_losses.append(loss.item())
                batch_losses_each_disp.append(loss.item())
                optimizer.zero_grad()  # clear gradients for next train
                torch.autograd.backward(loss)
                optimizer.step()

            avg_loss = sum(batch_losses) / (len(train_dataset) // batch_size)
            print('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))

            scheduler.step()
            lr_current = scheduler.get_last_lr()
            print('The current learning rate is {:.06f}'.format(lr_current[0]))

            end = time.time()
            print('Epoch %d training time cost: %.4f seconds' % (epoch + 1, end - start))

            # Test 
            model.eval()
            y_output = np.zeros(n_test)
            y_test = np.zeros(n_test)

            with torch.no_grad():
                for i, (imgs, pc, mos) in enumerate(test_loader):
                    imgs = imgs.to(device)
                    pc = torch.Tensor(pc.float())
                    pc = pc.to(device)
                    y_test[i] = mos.item()
                    outputs = model(imgs, pc)
                    y_output[i] = outputs.item()

                y_output_logistic = fit_function(y_test, y_output)
                test_PLCC = stats.pearsonr(y_output_logistic, y_test)[0]
                test_SROCC = stats.spearmanr(y_output, y_test)[0]
                test_RMSE = np.sqrt(((y_output_logistic - y_test) ** 2).mean())
                test_KROCC = scipy.stats.kendalltau(y_output, y_test)[0]
                print(
                    "Test results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(test_SROCC, test_KROCC,
                                                                                                test_PLCC, test_RMSE))

                if test_SROCC > best_test_criterion:
                    print("Update best model using best_val_criterion ")
                    # torch.save(model.state_dict(), 'ckpts/' + database + '_' + str(k_fold_id) + '_best_model.pth')
                    # scio.savemat(trained_model_file+'.mat',{'y_pred':y_pred,'y_test':y_test})
                    best[0:4] = [test_SROCC, test_KROCC, test_PLCC, test_RMSE]
                    best_test_criterion = test_SROCC  # update best val SROCC

                    print("Update the best Test results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(
                        test_SROCC, test_KROCC, test_PLCC, test_RMSE))

        print(database)
        best_all[k_fold_id - 1, :] = best
        print("The best Val results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(best[0], best[1],
                                                                                                  best[2], best[3]))
        print(
            '*************************************************************************************************************************')

    # average score
    best_mean = np.mean(best_all, 0)
    print(
        '*************************************************************************************************************************')
    print(
        "The mean val results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(best_mean[0], best_mean[1],
                                                                                            best_mean[2], best_mean[3]))
    print(
        '*************************************************************************************************************************')
