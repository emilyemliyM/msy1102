import gc
import torch
import os
import random
import argparse
from torch.utils.data import random_split
import numpy as np
import torch.optim as optim
import logging
from tqdm import tqdm
from network.polarsegformer import PolarSegFormer
from torch.utils.tensorboard import  SummaryWriter
from dataloader_nuscenes.nusenes_dataset import collate_fn_BEV, My_nuscenes, get_scene_name, spherical_dataset
from network.lovasz_losses import lovasz_softmax
import warnings
from confusion_matrix import per_class_iu, fast_hist_crop,get_gpu_memory

warnings.filterwarnings("ignore")


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count = np.bincount(
        n * label[k].astype(int) + pred[k], minlength = n ** 2)
    return bin_count[:n ** 2].reshape(n, n)


def SemKITTI2train(label):  # label torch.Size([1, 480, 360, 32])
    if isinstance(label, list):
        return [SemKITTI2train_single(a) for a in label]
    else:
        return SemKITTI2train_single(label)


def SemKITTI2train_single(label):
    return label - 1  # uint8 trick


def parse_args():
    parser = argparse.ArgumentParser(description = 'PolarNuscenes')
    parser.add_argument('-d', '--data_dir', default = 'data/nuscenes_moving_obstacle_detection')
    parser.add_argument('-p', '--model_save_path', default = './SemKITTI_PolarSeg.pt')
    parser.add_argument('-s', '--grid_size', nargs = '+', type = int, default = [480, 480, 32])
    parser.add_argument('--train_batch_size', type = int, default = 1, help = 'batch size for training (default: 2)')
    parser.add_argument('--val_batch_size', type = int, default = 1, help = 'batch size for validation (default: 2)')
    parser.add_argument('--check_iter', type = int, default = 1)
    parser.add_argument('--max_epoch', type = int, default = 10)
    parser.add_argument('--log_dir', type = str, default = "runs/", help = 'Log path [default: None]')
    return parser.parse_args()


SemKITTI_label_name = {0: 'background', 1: 'moving_vehicles', 2: 'moving_pedestrains'}

# BASE_dir = Path(__file__).resolve().parent  # E://xiaomeng
# SCENE_dir = os.path.join(BASE_dir, 'data', 'nuscenes_moving_obstacle_detection')
# scene_list = get_scene_name(SCENE_dir)  #重要的是 SCENE_dir，用参数喂了，--data_dir

def set_seed(seed = 1024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed()
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)

    def log_string(*str):
        logger.info(*str)
        print(*str, end = " ")

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('logs.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    data_path = args.data_dir
    max_epoch = args.max_epoch
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    check_iter = args.check_iter
    model_save_path = args.model_save_path
    compression_model = args.grid_size[2]  # 32 GRID SIZE 的最后一维
    grid_size = args.grid_size  # [480, 360, 32]

    # pytorch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pytorch_device = torch.device("cuda:0" )

    """
    gpu_memory = get_gpu_memory()
    if not gpu_memory:
        print("gpu free memory: {}".format(gpu_memory))
        gpu_list = np.argsort(gpu_memory)[::-1]

        gpu_list_str = ','.join(map(str, gpu_list))
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)
        pytorch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """
    tb_writer= SummaryWriter('runs/')

    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[
                   1:] - 1  # 2 list [ 0  1   ]
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]  # 取出除了背景以外的 类别名字

    # max_pt_per_encode 每个体素最大包含点云数
    # my_BEV_model = BEV_Unet(n_class = len(unique_label), n_height = compression_model, input_batch_norm = True,
    #                         dropout = 0.5, circular_padding = True)
    my_model = PolarSegFormer(n_class = len(unique_label), grid_size = grid_size, max_pt_per_encode = 256,
                              kernal_size = 1, fea_compre = compression_model)

    if os.path.exists(model_save_path):
        my_model.load_state_dict(torch.load(model_save_path))
    my_model.to(pytorch_device)

    optimizer = optim.Adam(my_model.parameters())
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.1)
    scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10, eta_min = 0.)

    loss_fun = torch.nn.CrossEntropyLoss(ignore_index = 255)
    # loss_fun = torch.nn.CrossEntropyLoss()

    # prepare dataset  1当前帧
    scene_list = get_scene_name(data_path)
    pt_dataset = My_nuscenes(data_path, scene_list)
    # print( pt_dataset) 试一下清显存
    del scene_list
    gc.collect()


    all_dataset = spherical_dataset(pt_dataset, grid_size = grid_size, flip_aug = True, ignore_label = 0,
                                    rotate_aug = True, fixed_volume_space = True)

    train_size = int(len(all_dataset) * 0.6)
    test_size = int(len(all_dataset) * 0.2)
    val_size = len(all_dataset) - train_size - test_size
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(all_dataset,
                                                                             [train_size, test_size, val_size])
    del train_size,val_size,test_size

    train_dataset_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = train_batch_size,
                                                       collate_fn = collate_fn_BEV,
                                                       shuffle = True, num_workers = 0)

    """#  test和 val的 collate_fn_BEV 需要换，还没写，哈哈哈哈。。。 不要忘了！！"""
    test_dataset_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = train_batch_size,
                                                      collate_fn = collate_fn_BEV,
                                                      shuffle = False, num_workers = 0)

    val_dataset_loader = torch.utils.data.DataLoader(dataset = val_dataset, batch_size = train_batch_size,
                                                     collate_fn = collate_fn_BEV,
                                                     shuffle = False, num_workers = 0)

    log_string("The number of training data is: %d" % len(train_dataset))
    log_string("The number of test data is: %d" % len(val_dataset))

    # training
    epoch = 0
    best_val_miou = 0
    start_training = False
    my_model.train()
    global_iter = 0
    exce_counter = 0

    for epoch in range(max_epoch):
        log_string('**** Epoch (%d/%s) ****' % (epoch + 1, max_epoch))
        train_loss = []
        loss_list = []
        hist_list = []
        train_pa_acc = 0
        pbar = tqdm(total = len(train_dataset_loader))
        for i_iter, data in enumerate(train_dataset_loader):
            # 是data2stack_1, label2stack_1, grid_ind_stack_1, point_label_1, xyz_1(8wei),后面3都是list
            data_0 = data[0]  # 前一帧
            data_1 = data[1]  # 当前帧
            _, train_vox_label_0, train_grid_0, _, train_pt_fea_0 = data_0
            _, train_vox_label_1, train_grid_1, train_pt_labs_1, train_pt_fea_1 = data_1
            # print( train_pt_fea_0[0].shape)

            train_vox_label_0 = SemKITTI2train(train_vox_label_0)  # 全255
            # train_pt_labs_0 = SemKITTI2train(train_pt_labs_0)  # list label[ npoint 类别]  0 1 2-> -1 0 1
            # 转换数据类型
            train_pt_fea_ten_0 = [torch.from_numpy(i).type(torch.FloatTensor).to( pytorch_device) for i in
                                  train_pt_fea_0]
            # train_grid: [nponint,3]  ,trian_grid_ten: 不要z了
            # 放到cuda gpu里
            train_grid_ten_0 = [torch.from_numpy(i[:, :2]).to( pytorch_device) for i in train_grid_0]
            train_vox_ten_0 = [torch.from_numpy(i) for i in train_grid_0]
            point_label_tensor_0 = train_vox_label_0.type(torch.IntTensor)
            del point_label_tensor_0 ,train_vox_ten_0
            gc.collect()

            train_vox_label_1 = SemKITTI2train(train_vox_label_1)  # 全255 # 转换数据类型
            train_pt_labs_1 = SemKITTI2train(train_pt_labs_1)  # 只用1就可以了预测现在的嘛，省个变量
            train_pt_fea_ten_1 = [torch.from_numpy(i).type(torch.FloatTensor).to( pytorch_device) for i in train_pt_fea_1]
            # train_grid: [nponint,3]  ,trian_grid_ten: 不要z了
            train_grid_ten_1 = [torch.from_numpy(i[:, :2]).to( pytorch_device) for i in train_grid_1]
            train_vox_ten_1 = [torch.from_numpy(i).to( pytorch_device) for i in train_grid_1]
            point_label_tensor_1 = train_vox_label_1.type(torch.LongTensor).to( pytorch_device)  # 计算loss

            # print("point_label_tensor_1的设备：", point_label_tensor_1.get_device())
            # forward + backward + optimize
            # train_pt_fea_ten (npoints，9),  train_grid_ten ( torch.Size([113316, 2])), 这俩均被[]list包了一层
            outputs = my_model(train_pt_fea_ten_0, train_grid_ten_0, train_pt_fea_ten_1, train_grid_ten_1)
            # print( outputs.shape)  #[1, 2, 480, 360, 32])

            # loss_fun:是交叉熵
            loss = 0.001 * lovasz_softmax(torch.nn.functional.softmax(outputs), point_label_tensor_1,
                                          ignore = 0) + loss_fun(outputs, point_label_tensor_1)
            # loss_0 = lovasz_softmax(torch.nn.functional.softmax(outputs_0), point_label_tensor_0, ignore = 0)
            # print('--l0ss---', loss.item())
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            pbar.update(1)
            start_training = True
            global_iter += 1
            tb_writer.add_scalars("Loss", {"Train": loss}, global_iter)
            # tb_writer.add_scalars("Loss_Train",loss, global_iter)  # 放到gpu就需要 loss.item()，暂时去掉
            # 评估一哈
            train_loss.append( loss.detach().cpu().numpy() )

            outputs = torch.argmax(outputs, dim = 1)
            outputs = outputs.cpu().detach().numpy()
            for count, i_val_grid in enumerate(train_grid_1):
                hist_list.append(fast_hist_crop(outputs[
                                                    count, train_grid_1[count][:, 0], train_grid_1[count][:, 1],
                                                    train_grid_1[count][:, 2]], train_pt_labs_1[count],
                                                unique_label))
            confusion_matrix = sum(hist_list)
            """train 的  iou和 miou"""
            # iou = per_class_iu(confusion_matrix)
            # print('per class iou: -------')
            # for class_name, class_iou in zip(unique_label_str, iou):
            #     print('%s : %.2f%%' % (class_name, class_iou * 100))
            # train_miou = np.nanmean(iou) * 100
            # acc
            train_pa_acc += np.diag(sum(hist_list)) / (confusion_matrix.sum(1) + 1e-10)
            del train_vox_label_1, train_grid_1, train_pt_fea_ten_0, train_grid_ten_0,point_label_tensor_1
            gc.collect()

        # print("train----------", train_pa_acc / len(train_dataset_loader))# [acc1,acc2]
        train_acc= train_pa_acc / len(train_dataset_loader)
        log_string('Training PA accuracy' , (train_acc) )
        tb_writer.add_scalars("PA_accuracy_car", {"training": train_acc[0]}, global_iter)
        tb_writer.add_scalars("PA_accuracy_men", {"training": train_acc[1]}, global_iter)
        del train_acc
        gc.collect()
        log_string('Training mean loss: %f' % (np.mean(train_loss)))

        scheduler_lr.step()
        pbar.close()
        epoch += 1
        #
        if epoch % check_iter == 0:
            my_model.eval()
            val_hist_list = []
            val_loss = []
            val_pa_acc=0
            with torch.no_grad():
                for i_iter, data in enumerate(val_dataset_loader):
                    # 是data2stack_1, label2stack_1, grid_ind_stack_1, point_label_1, xyz_1(8wei),后面3都是list
                    data_0 = data[0]  # 前一帧
                    data_1 = data[1]  # 当前帧
                    _, val_vox_label_0, val_grid_0, _, val_pt_fea_0 = data_0
                    _, val_vox_label_1, val_grid_1, val_pt_labs_1, val_pt_fea_1 = data_1
                    # print( train_pt_fea_0[0].shape)

                    val_vox_label_0 = SemKITTI2train(val_vox_label_0)  # 全255
                    # train_pt_labs_0 = SemKITTI2train(train_pt_labs_0)  # list label[ npoint 类别]  0 1 2-> -1 0 1
                    # 转换数据类型  原来是 FolatTensor
                    val_pt_fea_ten_0 = [torch.from_numpy(i).type(torch.FloatTensor).to( pytorch_device) for i in
                                          val_pt_fea_0]
                    # 放到cuda gpu里
                    val_grid_ten_0 = [torch.from_numpy(i[:, :2]).to(pytorch_device ) for i in val_grid_0]
                    val_vox_ten_0 = [torch.from_numpy(i) for i in val_grid_0]
                    val_point_label_tensor_0 = val_vox_label_0.type(torch.LongTensor)
                    del val_vox_ten_0
                    gc.collect()

                    val_vox_label_1 = SemKITTI2train(val_vox_label_1)  # 全255 # 转换数据类型
                    val_pt_labs_1 = SemKITTI2train(val_pt_labs_1)  # 只用1就可以了预测现在的嘛，省个变量
                    val_pt_fea_ten_1 = [torch.from_numpy(i).type(torch.FloatTensor).to( pytorch_device) for i in val_pt_fea_1]
                    # train_grid: [nponint,3]  ,trian_grid_ten: 不要z了
                    val_grid_ten_1 = [torch.from_numpy(i[:, :2]).to( pytorch_device) for i in val_grid_1]
                    val_vox_ten_1 = [torch.from_numpy(i).to( pytorch_device) for i in val_grid_1]
                    val_point_label_tensor_1 = val_vox_label_1.type(torch.LongTensor).to( pytorch_device) # 原来是longTensor

                    # forward + backward + optimize
                    # train_pt_fea_ten (npoints，9),  train_grid_ten ( torch.Size([113316, 2])), 这俩均被[]list包了一层
                    predict_labels = my_model(val_pt_fea_ten_0, val_grid_ten_0, val_pt_fea_ten_1, val_grid_ten_1)
                    # print( outputs.shape)  #[1, 2, 480, 360, 32])

                    # loss_fun:是交叉熵
                    loss = 0.001 * lovasz_softmax(torch.nn.functional.softmax(predict_labels).detach(), val_point_label_tensor_1,
                                                  ignore = 0) + loss_fun(predict_labels.detach(), val_point_label_tensor_1)

                    val_loss.append(loss.detach().cpu().numpy() )

                    predict_labels = torch.argmax(predict_labels, dim = 1)
                    predict_labels = predict_labels.cpu().detach().numpy()
                    for count, i_val_grid in enumerate(val_grid_1):
                        val_hist_list.append(fast_hist_crop(predict_labels[
                                                            count, val_grid_1[count][:, 0], val_grid_1[count][:, 1],
                                                            val_grid_1[count][:, 2]], val_pt_labs_1[count],
                                                        unique_label))
                confusion_matrix = sum(hist_list)
                iou = per_class_iu(confusion_matrix)
                print('per class iou: -------',end = " ")
                for class_name, class_iou in zip(unique_label_str, iou):
                    print('%s : %.2f%%' % (class_name, class_iou * 100))
                val_miou = np.nanmean(iou) * 100
                log_string('epoch: %d,val_miou: %f' % (epoch+1 , val_miou))

                tb_writer.add_scalar("Loss_Valid", np.mean(val_loss ), global_iter)

                val_pa_acc += np.diag(sum(hist_list)) / (confusion_matrix.sum(1) + 1e-10)
                del val_vox_label_1, val_grid_1, val_pt_fea_ten_0, val_grid_ten_0, val_point_label_tensor_1
                del confusion_matrix
                gc.collect()

                val_acc = val_pa_acc / len(val_dataset_loader)
                log_string('Validate PA car_accuracy',( val_acc[0]))
                log_string('Validate PA men_accuracy', (val_acc[1]))

                tb_writer.add_scalars("PA_accuracy_car", {"validate": val_acc[0]}, global_iter)
                tb_writer.add_scalars("PA_accuracy_men", {"validate": val_acc[1]}, global_iter)

                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()

                if best_val_miou < val_miou:
                    best_val_miou = val_miou
                    state = { "model_net":my_model.state_dict(),
                              "optimizer":optimizer.state_dict(),
                              "epoch":epoch}
                    torch.save( state, './best_model.pth')
                    logger.info('Save model...')
                log_string('Best mIoU: %f' % best_val_miou)
                print('Current val miou is %.3f while the best val miou is %.3f' %  (val_miou, best_val_miou))
                # print('Current val loss is %.3f' %  (np.mean(val_loss)))
                log_string(' (np.mean(val_loss)): %f' %  (np.mean(val_loss)) )


if __name__ == '__main__':
    main()
