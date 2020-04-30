import torch
import torch.utils.data
import torchvision
from loader import *
import os
from fcrn import FCRN
from torch.autograd import Variable
from weights import load_weights
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
import criteria
from torch.optim.lr_scheduler import StepLR # import for lr scheduler
import argparse
from data import getTrainingTestingData

dtype = torch.cuda.FloatTensor
weights_file = "NYU_ResNet-UpProj.npy"



def load_split():
    # 读取当前文件根目录
    current_directoty = os.getcwd()

    # 加载根目录/地址文件路径
    train_lists_path = current_directoty + '/trainIdxs.txt'
    test_lists_path = current_directoty + '/testIdxs.txt'

    # 打开地址txt文件
    train_f = open(train_lists_path)
    test_f = open(test_lists_path)

    # 创建训练列表和测试列表
    train_lists = []
    test_lists = []

    # 读取训练集地址txt文件并写入List
    train_lists_line = train_f.readline()
    while train_lists_line:
        train_lists.append(int(train_lists_line) - 1)
        train_lists_line = train_f.readline()
    train_f.close()

    # 读取测试集地址txt文件并写入List
    test_lists_line = test_f.readline()
    while test_lists_line:
        test_lists.append(int(test_lists_line) - 1)
        test_lists_line = test_f.readline()
    test_f.close()

    # 按8：2分割训练集和验证集
    val_start_idx = int(len(train_lists) * 0.8)

    # 前80%为训练集training dataset
    # 后80%为测试集validation dataset
    val_lists = train_lists[val_start_idx:-1]
    train_lists = train_lists[0:val_start_idx]

    return train_lists, val_lists, test_lists


def main():
    batch_size = args.batch_size
    data_path = 'nyu_depth_v2_labeled.mat'
    learning_rate = args.lr#1.0e-4 #1.0e-5
    monentum = 0.9
    weight_decay = 0.0005
    num_epochs = args.epochs
    step_size = args.step_size
    step_gamma = args.step_gamma
    resume_from_file = False
    isDataAug = args.data_aug
    max_depth = 1000

    # 1.Load data
    train_lists, val_lists, test_lists = load_split()
    print("Loading data......")
    train_loader = torch.utils.data.DataLoader(NyuDepthLoader(data_path, train_lists),
                                               batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(NyuDepthLoader(data_path, val_lists),
                                               batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(NyuDepthLoader(data_path, test_lists),
                                             batch_size=batch_size, shuffle=False, drop_last=True)
    print(train_loader)

    # 2.set the model
    print("Set the model......")
    model = FCRN(batch_size)
    resnet = torchvision.models.resnet50()

    # 加载训练到一半的模型
    # resnet.load_state_dict(torch.load('/home/xpfly/nets/ResNet/resnet50-19c8e357.pth'))
    # print("resnet50 params loaded.")

    # model.load_state_dict(load_weights(model, weights_file, dtype))

    model = model.cuda()

    # 3.Loss
    # loss_fn = torch.nn.MSELoss().cuda()
    if args.loss_type == "berhu":
        loss_fn = criteria.berHuLoss().cuda()
        print("berhu loss_fn set.")
    elif args.loss_type == "L1":
        loss_fn = criteria.MaskedL1Loss().cuda()
        print("L1 loss_fn set.")
    elif args.loss_type == "mse":
        loss_fn = criteria.MaskedMSELoss().cuda()
        print("MSE loss_fn set.")
    elif args.loss_type == "ssim":
        loss_fn = criteria.SsimLoss().cuda()
        print("Ssim loss_fn set.")
    elif args.loss_type == "three":
        loss_fn = criteria.Ssim_grad_L1().cuda()
        print("SSIM+L1+Grad loss_fn set.")

    # 5.Train
    best_val_err = 1.0e3

    # validate
    model.eval()
    num_correct, num_samples = 0, 0
    loss_local = 0
    with torch.no_grad():
        for input, depth in val_loader:
            input_var = Variable(input.type(dtype))
            depth_var = Variable(depth.type(dtype))

            output = model(input_var)

            input_rgb_image = input_var[0].data.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            input_gt_depth_image = depth_var[0][0].data.cpu().numpy().astype(np.float32)
            pred_depth_image = output[0].data.squeeze().cpu().numpy().astype(np.float32)

            input_gt_depth_image /= np.max(input_gt_depth_image)
            pred_depth_image /= np.max(pred_depth_image)

            plot.imsave('./result/input_rgb_epoch_0.png', input_rgb_image)
            plot.imsave('./result/gt_depth_epoch_0.png', input_gt_depth_image, cmap="viridis")

            plot.imsave('pred_depth_epoch_0.png', pred_depth_image, cmap="viridis")

            # depth_var = depth_var[:, 0, :, :]
            # loss_fn_local = torch.nn.MSELoss()

            loss_local += loss_fn(output, depth_var)

            num_samples += 1

    err = float(loss_local) / num_samples
    print('val_error before train:', err)

    start_epoch = 0

    resume_file = 'checkpoint.pth.tar'
    if resume_from_file:
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume_file))
            checkpoint = torch.load(resume_file)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_file, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_file))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=step_gamma)  # may change to other value

    for epoch in range(num_epochs):

        # 4.Optim

        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=monentum)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=monentum, weight_decay=weight_decay)
        print("optimizer set.")

        print('Starting train epoch %d / %d' % (start_epoch + epoch + 1, num_epochs))
        model.train()
        running_loss = 0
        count = 0
        epoch_loss = 0

        #for i, (input, depth) in enumerate(train_loader):
        for input, depth in train_loader:
            print("depth", depth)
            if isDataAug:
                depth = depth * 1000
                depth = torch.clamp(depth, 10, 1000)
                depth = max_depth / depth

            input_var = Variable(input.type(dtype)) # variable is for derivative
            depth_var = Variable(depth.type(dtype)) # variable is for derivative
            # print("depth_var",depth_var)

            output = model(input_var)

            loss = loss_fn(output, depth_var)
            print('loss:', loss.data.cpu())
            count += 1
            running_loss += loss.data.cpu().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = running_loss / count
        print('epoch loss:', epoch_loss)

        # validate
        model.eval()
        num_correct, num_samples = 0, 0
        loss_local = 0
        with torch.no_grad():
            for input, depth in val_loader:

                if isDataAug:
                    depth = depth * 1000
                    depth = torch.clamp(depth, 10, 1000)
                    depth = max_depth / depth

                input_var = Variable(input.type(dtype))
                depth_var = Variable(depth.type(dtype))

                output = model(input_var)

                input_rgb_image = input_var[0].data.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                input_gt_depth_image = depth_var[0][0].data.cpu().numpy().astype(np.float32)
                pred_depth_image = output[0].data.squeeze().cpu().numpy().astype(np.float32)

                # normalization
                input_gt_depth_image /= np.max(input_gt_depth_image)
                pred_depth_image /= np.max(pred_depth_image)

                plot.imsave('./result/input_rgb_epoch_{}.png'.format(start_epoch + epoch + 1), input_rgb_image)
                plot.imsave('./result/gt_depth_epoch_{}.png'.format(start_epoch + epoch + 1), input_gt_depth_image, cmap="viridis")
                plot.imsave('./result/pred_depth_epoch_{}.png'.format(start_epoch + epoch + 1), pred_depth_image, cmap="viridis")

                # depth_var = depth_var[:, 0, :, :]
                # loss_fn_local = torch.nn.MSELoss()

                loss_local += loss_fn(output, depth_var)

                num_samples += 1

                if epoch % 10 == 9:
                    PATH = args.loss_type+'.pth'
                    torch.save(model.state_dict(), PATH)

        err = float(loss_local) / num_samples
        print('val_error:', err)

        if err < best_val_err:
            best_val_err = err
            torch.save({
                'epoch': start_epoch + epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, 'checkpoint.pth.tar')

        scheduler.step()

        # if epoch % 10 == 0:
        #     learning_rate = learning_rate * 0.6


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set lr, stepper, and Loss Function')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='N',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--step-size', type=int, default=6, metavar='N',
                        help='step size of scheduler (default: 5)')
    parser.add_argument('--step-gamma', type=float, default=0.5, metavar='N',
                        help='deprecate rate of scheduler(default: 0.5)')
    parser.add_argument('--loss-type', type=str, default="three", metavar='N',
                        help='loss type "berhu" "L1" "mse" "ssim" "three"(default: berhu)')
    parser.add_argument('--data-aug', type=bool, default=True, metavar='N',
                        help='Do data augmentation or not (default: Ture)')
    args = parser.parse_args()
    main()