import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import torch

import utils.general as utils
import utils.plots as plt

class IDRTrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32) # 设置默认的浮点类型为torch.float32
        torch.set_num_threads(1) # 设置使用的线程数为1

        self.conf = ConfigFactory.parse_file(kwargs['conf']) # 解析配置文件
        self.batch_size = kwargs['batch_size'] # batch size大小
        self.nepochs = kwargs['nepochs'] # 训练epoch数
        self.exps_folder_name = kwargs['exps_folder_name'] # 实验文件夹名称
        self.GPU_INDEX = kwargs['gpu_index'] # gpu ID
        self.train_cameras = kwargs['train_cameras'] # 是否优化相机位置

        self.expname = self.conf.get_string('train.expname') + kwargs['expname'] # 实验名称
        scan_id = kwargs['scan_id'] if kwargs['scan_id'] != -1 else self.conf.get_int('dataset.scan_id', default=-1) # 数据ID
        if scan_id != -1:
            self.expname = self.expname + '_{0}'.format(scan_id) # 实验名称+=加上数据ID

        # 设置是否从上次中断继续训练，以及上次训练的时间戳和checkpoint
        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join('../',kwargs['exps_folder_name'],self.expname)):
                timestamps = os.listdir(os.path.join('../',kwargs['exps_folder_name'],self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        utils.mkdir_ifnotexists(os.path.join('../',self.exps_folder_name)) # 创建 ../exps
        self.expdir = os.path.join('../', self.exps_folder_name, self.expname) 
        utils.mkdir_ifnotexists(self.expdir)                               # 创建 ../exps/expname
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp)) # 创建../exps/expname/timestamp

        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir) # 创建../exps/expname/timestamp/plots

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path) # 创建../exps/expname/timestamp/checkpoints
        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"
        self.scheduler_params_subdir = "SchedulerParameters"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir)) # 创建../exps/expname/timestamp/checkpoints/ModelParameters
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir)) # 创建../exps/expname/timestamp/checkpoints/OptimizerParameters
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir)) # 创建../exps/expname/timestamp/checkpoints/SchedulerParameters

        # 如果优化相机位置，创建相应文件夹
        if self.train_cameras:
            self.optimizer_cam_params_subdir = "OptimizerCamParameters"
            self.cam_params_subdir = "CamParameters"

            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_cam_params_subdir)) # 创建../exps/expname/timestamp/checkpoints/OptimizerCamParameters
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.cam_params_subdir)) # 创建../exps/expname/timestamp/checkpoints/CamParameters

        # copy配置文件到../exps/expname/timestamp/runconf.conf
        os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))

        # 设置gpu环境变量
        if (not self.GPU_INDEX == 'ignore'):
            os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')
        
        dataset_conf = self.conf.get_config('dataset') # 获取dataset配置
        if kwargs['scan_id'] != -1:
            dataset_conf['scan_id'] = kwargs['scan_id']

        # 加载数据
        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(self.train_cameras,
                                                                                          **dataset_conf)

        print('Finish loading data ...')
        # 创建数据加载器，用于训练
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn
                                                            )
        # 创建数据加载器，用于绘制图像
        self.plot_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=True,
                                                           collate_fn=self.train_dataset.collate_fn
                                                           )

        # 创建模型
        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=self.conf.get_config('model'))
        # 将模型导入cuda
        if torch.cuda.is_available():
            self.model.cuda()

        # 创建损失函数
        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))

        # 学习率，优化器，学习率调度器
        self.lr = self.conf.get_float('train.learning_rate')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.sched_milestones = self.conf.get_list('train.sched_milestones', default=[])
        self.sched_factor = self.conf.get_float('train.sched_factor', default=0.0)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.sched_milestones, gamma=self.sched_factor)

        # settings for camera optimization
        # 设置优化相机, 能够优化相机pose这点很有价值
        if self.train_cameras:
            num_images = len(self.train_dataset) # 图像数量，也是相机pose数量
            self.pose_vecs = torch.nn.Embedding(num_images, 7, sparse=True).cuda() # 创建待优化的pose嵌入向量(dim=7, quat+t),并使用稀疏梯度更新(节省内存)
            self.pose_vecs.weight.data.copy_(self.train_dataset.get_pose_init())   # 将初始pose赋值给权重
            
            # 创建相机pose优化器，稀疏Adam
            self.optimizer_cam = torch.optim.SparseAdam(self.pose_vecs.parameters(), self.conf.get_float('train.learning_rate_cam'))

        self.start_epoch = 0

        # 如果从上次中断继续训练，则加载checkpoint
        if is_continue:
            # checkpoint路径
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')
            
            # 加载模型参数和start_epoch
            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"])
            self.start_epoch = saved_model_state['epoch']

            # 加载优化器参数
            data = torch.load(
                os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])

            # 加载学习率调度器参数
            data = torch.load(
                os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.scheduler.load_state_dict(data["scheduler_state_dict"])

            if self.train_cameras:
                # 加载相机优化器参数
                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.optimizer_cam_params_subdir, str(kwargs['checkpoint']) + ".pth"))
                self.optimizer_cam.load_state_dict(data["optimizer_cam_state_dict"])

                # 加载相机pose参数
                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.cam_params_subdir, str(kwargs['checkpoint']) + ".pth"))
                self.pose_vecs.load_state_dict(data["pose_vecs_state_dict"])

        self.num_pixels = self.conf.get_int('train.num_pixels') # 采样像素数量
        self.total_pixels = self.train_dataset.total_pixels # 图像总像素数量
        self.img_res = self.train_dataset.img_res # 图像分辨率
        self.n_batches = len(self.train_dataloader) # batch数量
        self.plot_freq = self.conf.get_int('train.plot_freq') # 绘制频率
        self.plot_conf = self.conf.get_config('plot') # 绘制配置

        self.alpha_milestones = self.conf.get_list('train.alpha_milestones', default=[]) # alpha衰减 milestones
        self.alpha_factor = self.conf.get_float('train.alpha_factor', default=0.0) # alpha衰减因子
        # 根据epoch调整alpha
        for acc in self.alpha_milestones:
            if self.start_epoch > acc:
                self.loss.alpha = self.loss.alpha * self.alpha_factor

    # 保存checkpoint：模型，优化器，学习率调度器，相机优化器，相机pose
    def save_checkpoints(self, epoch):
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))

        if self.train_cameras:
            torch.save(
                {"epoch": epoch, "optimizer_cam_state_dict": self.optimizer_cam.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_cam_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "optimizer_cam_state_dict": self.optimizer_cam.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_cam_params_subdir, "latest.pth"))

            torch.save(
                {"epoch": epoch, "pose_vecs_state_dict": self.pose_vecs.state_dict()},
                os.path.join(self.checkpoints_path, self.cam_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "pose_vecs_state_dict": self.pose_vecs.state_dict()},
                os.path.join(self.checkpoints_path, self.cam_params_subdir, "latest.pth"))

    # 运行主函数：IDR的整个pipeline
    def run(self):
        # 立正！开始训练！！！
        print("training...")

        # 训练大循环，epoch: [start_epoch, nepochs + 1]
        # start_epoch不一定是0，有可能是从某个checkpoint加载的
        for epoch in range(self.start_epoch, self.nepochs + 1):
            # 根据epoch调整alpha，每经过一定epochs, 会乘以factor来增大
            # alpha是用于计算mask loss的
            if epoch in self.alpha_milestones:
                self.loss.alpha = self.loss.alpha * self.alpha_factor
            
            # 每经过100个epoch，保存一次checkpoint
            if epoch % 100 == 0:
                self.save_checkpoints(epoch)

            # 每经过plot_freq个epoch，绘制一次图像
            if epoch % self.plot_freq == 0:
                # 模型设置为评估模式，即去掉训练时的行为(如dropout，batchnorm等)，保证模型输出的一致性
                self.model.eval()
                if self.train_cameras:
                    self.pose_vecs.eval() # 相机pose设置为评估模式
                # 采样像素数量设置为-1，即所有像素都参与采样
                self.train_dataset.change_sampling_idx(-1)
                # 选择一个batch进行绘制，包括ids，模型输入，真值
                indices, model_input, ground_truth = next(iter(self.plot_dataloader))

                # 将模型输入导入cuda
                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input["object_mask"] = model_input["object_mask"].cuda()

                if self.train_cameras:
                    # 如果训练相机，将嵌入后的相机pose向量导入cuda
                    pose_input = self.pose_vecs(indices.cuda())
                    model_input['pose'] = pose_input
                else:
                    # 如果不训练相机，将相机pose导入cuda
                    model_input['pose'] = model_input['pose'].cuda()

                # 将模型输入分为多个chunk进行处理，因为显存可能不够
                split = utils.split_input(model_input, self.total_pixels) # 将模型输入分为多个chunk
                res = [] # 存储模型输出
                # 对每个chunk进行处理，得到模型输出，将输出存储到res中
                for s in split:
                    out = self.model(s) # 推理得到模型输出
                    res.append({
                        'points': out['points'].detach(), # 射线与表面交点3D坐标
                        'rgb_values': out['rgb_values'].detach(), # 模型输出的rgb预测值
                        'network_object_mask': out['network_object_mask'].detach(), # ray_tracing预测的mask
                        'object_mask': out['object_mask'].detach() # 真值的mask
                    })

                # 合并split结果
                batch_size = ground_truth['rgb'].shape[0]
                model_outputs = utils.merge_output(res, self.total_pixels, batch_size)

                # 绘制模型预测结果
                plt.plot(self.model,          # 模型
                         indices,             # 图像索引
                         model_outputs,       # 合并后的模型输出
                         model_input['pose'], # pose输入
                         ground_truth['rgb'], # rgb真值
                         self.plots_dir,      # 绘制结果保存路径
                         epoch,               # 训练epoch
                         self.img_res,        # 图像分辨率
                         **self.plot_conf     # 绘制参数
                         )

                # 模型设置为训练模式
                self.model.train()
                if self.train_cameras:
                    self.pose_vecs.train()

            # 采样像素数量设置为num_pixels，即随机采样num_pixels个像素点进行训练
            self.train_dataset.change_sampling_idx(self.num_pixels)

            # 单个epoch的训练
            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                # 将模型输入导入cuda
                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input["object_mask"] = model_input["object_mask"].cuda()

                if self.train_cameras:
                    # 如果训练相机，将嵌入后的相机pose向量导入cuda
                    pose_input = self.pose_vecs(indices.cuda())
                    model_input['pose'] = pose_input
                else:
                    # 如果不训练相机，将相机pose导入cuda
                    model_input['pose'] = model_input['pose'].cuda()

                model_outputs = self.model(model_input) # 推理得到模型输出
                loss_output = self.loss(model_outputs, ground_truth) # 计算loss

                loss = loss_output['loss'] # 总loss(rgb_loss+eikonal_loss+mask_loss)

                # 将梯度置为0
                self.optimizer.zero_grad()
                if self.train_cameras:
                    self.optimizer_cam.zero_grad()

                # 反向传播，计算梯度
                loss.backward()

                # 更新权重和pose参数
                self.optimizer.step()
                if self.train_cameras:
                    self.optimizer_cam.step()
                
                # 打印loss信息
                print(
                    '{0} [{1}] ({2}/{3}): loss = {4}, rgb_loss = {5}, eikonal_loss = {6}, mask_loss = {7}, alpha = {8}, lr = {9}'
                        .format(self.expname, epoch, data_index, self.n_batches, loss.item(),
                                loss_output['rgb_loss'].item(),
                                loss_output['eikonal_loss'].item(),
                                loss_output['mask_loss'].item(),
                                self.loss.alpha,
                                self.scheduler.get_lr()[0]))
            # 更新学习率
            self.scheduler.step()
