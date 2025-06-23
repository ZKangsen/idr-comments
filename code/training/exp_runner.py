import sys
sys.path.append('../code')
import argparse
import GPUtil

from training.idr_train import IDRTrainRunner

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size') # batch size大小
    parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for') # 训练epoch数
    parser.add_argument('--conf', type=str, default='./confs/dtu_fixed_cameras.conf') # 配置文件
    parser.add_argument('--expname', type=str, default='') # 实验名称
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]') # gpu ID
    parser.add_argument('--is_continue', default=False, action="store_true", help='If set, indicates continuing from a previous run.') # 是否从上次中断继续训练
    parser.add_argument('--timestamp', default='latest', type=str, help='The timestamp of the run to be used in case of continuing from a previous run.') # 上次训练的时间戳
    parser.add_argument('--checkpoint', default='latest',type=str,help='The checkpoint epoch number of the run to be used in case of continuing from a previous run.') # 上次训练的checkpoint
    parser.add_argument('--train_cameras', default=False, action="store_true", help='If set, optimizing also camera location.') # 是否优化相机位置
    parser.add_argument('--scan_id', type=int, default=-1, help='If set, taken to be the scan id.') # 数据ID

    opt = parser.parse_args() # 解析参数

    if opt.gpu == "auto": # 自动选择gpu
        # 使用GPUtil库获取可用的GPU设备ID列表，按照内存占用排序，限制返回的设备数量为1，最大负载为0.5，最大内存占用为0.5，不包括NaN值，排除ID为空，排除UUID为空
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0] # 选择第一个gpu
    else:
        gpu = opt.gpu # 手动选择gpu

    print('gpu:', gpu)

    # IDR训练器
    trainrunner = IDRTrainRunner(conf=opt.conf,
                                 batch_size=opt.batch_size,
                                 nepochs=opt.nepoch,
                                 expname=opt.expname,
                                 gpu_index=gpu,
                                 exps_folder_name='exps',
                                 is_continue=opt.is_continue,
                                 timestamp=opt.timestamp,
                                 checkpoint=opt.checkpoint,
                                 scan_id=opt.scan_id,
                                 train_cameras=opt.train_cameras
                                 )
    # 开始训练
    trainrunner.run()
