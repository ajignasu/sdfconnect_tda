# -*- coding: utf-8 -*-

import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from models.dataset import DatasetNP, DatasetNP_TDA
from models.fields import NPullNetwork
import argparse
from pyhocon import ConfigFactory
import os
from shutil import copyfile
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from models.utils import get_root_logger, print_log, compute_cubical_cmplx, new_top_loss1, new_top_loss2
import math
import mcubes
from pyhocon import ConfigFactory
import warnings
warnings.filterwarnings("ignore")

import gudhi as gd
import wandb
wandb.login()


class Runner:
    def __init__(self, args, conf_path, mode='train'):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        logging_config = self.conf.as_plain_ordered_dict()
        wandb.init(project=self.conf['general.project_name'], config=logging_config, name=args.dataname)
        self.conf['dataset.np_data_name'] = self.conf['dataset.np_data_name']
        self.dataset_name = self.conf['dataset.dataset_type']

        if self.dataset_name == 'srb':
            self.gt_path_metrics = os.path.join(self.conf['dataset.data_dir'] + '/ground_truth/', args.dataname + '.ply')
        elif self.dataset_name == 'famous':
            temp_name = args.dataname.split(".xyz")[0]
            self.gt_path_metrics = os.path.join(self.conf['dataset.data_dir'] + '/03_meshes/', temp_name + '.ply')
        elif self.dataset_name == 'abc':
            temp_name = args.dataname.split(".xyz")[0]
            self.gt_path_metrics = os.path.join(self.conf['dataset.data_dir'] + '/03_meshes/', temp_name + '.ply')
        

        self.base_exp_dir = self.conf['general.base_exp_dir'] + args.dir
        os.makedirs(self.base_exp_dir, exist_ok=True)
        
        self.mode = mode
        self.save_sdf = self.conf.get_bool('train.save_sdf')
        self.sdf_grid_res = self.conf.get_int('train.sdf_grid_res')
        print('save_sdf: ', self.save_sdf)
        print('sdf_grid_res: ', self.sdf_grid_res)
        # exit()
        self.sigma_val = self.conf.get_float('train.sigma_val')
        if self.mode == 'train':
            self.dataset_np = DatasetNP(self.conf['dataset'], args.dataname, self.dataset_name, self.sigma_val)
        else:
            # get TDA radius parameter
            self.persistence_radius = self.conf.get_float('train.persistence_radius')
            self.persistence_dim = self.conf.get_int('train.persistence_dim')
            self.persistence_lambda = self.conf.get_float('train.persistence_lambda')
            self.persistence_lambda_1 = self.conf.get_float('train.persistence_lambda_1')
            self.persistence_lambda_2 = self.conf.get_float('train.persistence_lambda_2')
            print('persistence_radius: ', self.persistence_radius)
            print('persistence_dim: ', self.persistence_dim)
            print('persistence_lambda: ', self.persistence_lambda)
            print('persistence_lambda_1: ', self.persistence_lambda_1)
            print('persistence_lambda_2: ', self.persistence_lambda_2)
            self.dataset_np = DatasetNP_TDA(self.conf['dataset'], args.dataname, self.dataset_name, self.persistence_radius, self.persistence_dim, self.sigma_val, self.base_exp_dir)

        self.dataname = args.dataname
        self.iter_step = 0

        # Training parameters
        self.maxiter = self.conf.get_int('train.maxiter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.eval_num_points = self.conf.get_int('train.eval_num_points')
        
        self.metric_eval_pts = self.conf.get_int('train.metric_sample_pts')

        # Networks
        self.sdf_network = NPullNetwork(**self.conf['model.sdf_network']).to(self.device)

        print('network: ', self.sdf_network)
        self.optimizer = torch.optim.Adam(self.sdf_network.parameters(), lr=self.learning_rate)

        # Backup codes and configs for debug
        if self.mode == 'train' or self.mode == 'train_tda':
            self.file_backup()
    


    def train(self):
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(os.path.join(self.base_exp_dir), f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, name='outs')
        self.logger = logger
        batch_size = self.batch_size

        res_step = self.maxiter - self.iter_step

        # pc_barcodes, pers_barcodes = compute_pers_diagram(radius_threshold, point_gt, max_dim_pc)
        for iter_i in tqdm(range(res_step)):
            self.update_learning_rate_np(iter_i)

            if self.mode == 'train':
                points, samples, point_gt = self.dataset_np.np_train_data(batch_size)
            else:
                points, samples, point_gt, all_dim_barcodes = self.dataset_np.np_train_data(batch_size)
                
            samples.requires_grad = True
    
            gradients_sample = self.sdf_network.gradient(samples).squeeze() # 5000x3; now 4096x3
            sdf_sample = self.sdf_network.sdf(samples)                      # 5000x1; now 4096x1
            grad_norm = F.normalize(gradients_sample, dim=1)                # 5000x3; now 4096x3
            sample_moved = samples - grad_norm * sdf_sample                 # 5000x3; now 4096x3

            loss_sdf = torch.linalg.norm((points - sample_moved), ord=2, dim=-1).mean()
            
           
            ########################## topology loss ##########################
            if self.mode == 'train_tda':
                domain_all_dim_barcodes, cc_barcodes, zero_dim_barcodes = compute_cubical_cmplx(torch.abs(sdf_sample), self.persistence_dim)
                
                topology_loss1 = new_top_loss1(domain_all_dim_barcodes)
                topology_loss2 = new_top_loss2(domain_all_dim_barcodes)
                top_loss = (
                        + self.persistence_lambda_1 * topology_loss1
                        + self.persistence_lambda_2 * topology_loss2
                        )
                
                loss = loss_sdf + top_loss
                
            else:
                loss = loss_sdf
            ###################################################################
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.iter_step += 1
            if self.iter_step % self.report_freq == 0:
                print_log('iter:{:8>d} cd_l1 = {} lr={}'.format(self.iter_step, loss_sdf, self.optimizer.param_groups[0]['lr']), logger=logger)
                if self.mode == 'train_tda':
                    wandb.log({'iter': self.iter_step, 'total_loss': loss, 'loss_sdf': loss_sdf, 'total_tda_loss': topology_loss1+topology_loss2, 'top_loss_1': topology_loss1, 'top_loss_2': topology_loss2})
                    wandb.log({'iter': self.iter_step, 'total_loss': loss, 'loss_sdf': loss_sdf, 'total_tda_loss_scaled': top_loss, 'top_loss_1_scaled': self.persistence_lambda_1 * topology_loss1, 'top_loss_2_scaled': self.persistence_lambda_2 * topology_loss2})
                    wandb.log({'iter': self.iter_step, 'learning_rate': self.optimizer.param_groups[0]['lr']})
                    self.save_tda_plot(zero_dim_barcodes[0])
                    img = plt.imread(os.path.join(self.base_exp_dir, f"cc_{self.iter_step}.jpg"))
                    wandb.log({"persistence diagram": wandb.Image(img)})
                else:
                    wandb.log({'iter': self.iter_step, 'total_loss': loss, 'loss_sdf': loss_sdf})
                    wandb.log({'iter': self.iter_step, 'learning_rate': self.optimizer.param_groups[0]['lr']})


            if self.iter_step % self.val_freq == 0 and self.iter_step!=0:
                self.validate_mesh(resolution=self.sdf_grid_res, threshold=args.mcubes_threshold, point_gt=point_gt, iter_step=self.iter_step, logger=logger, save_sdf=self.save_sdf)


            if self.iter_step % self.save_freq == 0 and self.iter_step!=0: 
                self.save_checkpoint()
        
        self.generate_metrics(self.gt_path_metrics, self.metric_eval_pts)
        wandb.finish()

    def normalize_to_unit_cube(self, points):
        """
        Normalize the points of the point cloud to fit within a unit cube [-1, 1]^3.
        """
        
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)

        max_range = np.max(max_bound - min_bound)
        
        scale = 2.0 / max_range
        translation = (min_bound + max_bound) / 2.0

        normalized_points = (points - translation) * scale

        return normalized_points
    
    def torch_distance_matrix(self, A, B, batch_size=1024):
        """
        Compute a distance matrix in a memory-efficient manner using PyTorch.
        """
        A = torch.tensor(A, device=self.device)
        B = torch.tensor(B, device=self.device)

        dist_matrix = torch.zeros(A.size(0), B.size(0), device=self.device)

        for i in range(0, A.size(0), batch_size):
            end_i = min(i + batch_size, A.size(0))
            for j in range(0, B.size(0), batch_size):
                end_j = min(j + batch_size, B.size(0))
                diff = A[i:end_i].unsqueeze(1) - B[j:end_j].unsqueeze(0)
                dist_matrix[i:end_i, j:end_j] = torch.sqrt((diff ** 2).sum(2))

        return dist_matrix
    
    def torch_chamfer_distance(self, A, B):
        """
        Compute the Chamfer distance between two sets of points, A and B, using PyTorch for GPU acceleration.
        """
        distances_A_to_B = self.torch_distance_matrix(A, B)
        distances_B_to_A = self.torch_distance_matrix(B, A)

        min_A_to_B = distances_A_to_B.min(dim=1)[0]
        min_B_to_A = distances_B_to_A.min(dim=1)[0]

        cd_one_sided = min_A_to_B.mean()
        cd_two_sided = cd_one_sided + min_B_to_A.mean()

        return cd_one_sided.item(), cd_two_sided.item()

    def torch_hausdorff_distance(self, A, B):
        """
        Compute the Hausdorff distance between two sets of points, A and B, using PyTorch.
        """
        A = torch.tensor(A, device='cuda')
        B = torch.tensor(B, device='cuda')

        distances_A_to_B = self.torch_distance_matrix(A, B)
        distances_B_to_A = self.torch_distance_matrix(B, A)

        min_A_to_B = distances_A_to_B.min(dim=1)[0]
        min_B_to_A = distances_B_to_A.min(dim=0)[0]

        h_distance = max(min_A_to_B.max(), min_B_to_A.max())

        return h_distance.item()

    def generate_metrics(self, gt_path, metric_sample_pts):
        print('Generating metrics...')

        if self.dataset_name == 'srb':
            print(f'GT path: {gt_path}')
            print(type(gt_path))
            gt = trimesh.load(gt_path) # incase of SRB, gt is a point cloud
            pred_mesh = trimesh.load(os.path.join(self.base_exp_dir, "outputs/00040000_0.0.ply"))
        elif self.dataset_name == 'abc':
            print(f'GT path: {gt_path}')
            print(type(gt_path))
            gt = trimesh.load(gt_path) # in case of ABC, gt is a mesh
            gt = trimesh.sample.sample_surface(gt, metric_sample_pts)[0]
            pred_mesh = trimesh.load(os.path.join(self.base_exp_dir, "outputs/00040000_0.0.ply"))
        elif self.dataset_name == 'famous':
            print(f'GT path: {gt_path}')
            print(type(gt_path))
            gt = trimesh.load(gt_path) # in case of FAMOUS, gt is a mesh
            gt = trimesh.sample.sample_surface(gt, metric_sample_pts)[0]
            pred_mesh = trimesh.load(os.path.join(self.base_exp_dir, "outputs/00040000_0.0.ply"))
        
        indices = np.random.choice(gt.shape[0], metric_sample_pts, replace=False)
        gt_samples = gt[indices, :]
        pred_pts = trimesh.sample.sample_surface(pred_mesh, metric_sample_pts)[0]
        normalized_gt_pts = self.normalize_to_unit_cube(gt_samples)
        normalized_pred_pts = self.normalize_to_unit_cube(pred_pts)
        chamfer_dist_one_sided, chamfer_distance_two_sided = self.torch_chamfer_distance(normalized_pred_pts, normalized_gt_pts)
        hausdorff_dist = self.torch_hausdorff_distance(normalized_pred_pts, normalized_gt_pts)
        save_name = os.path.join(self.base_exp_dir, "metrics.txt")
        with open(save_name, 'w') as f:
                f.write(f'cd_one_sided: {chamfer_dist_one_sided}\n')
                f.write(f'cd_two_sided: {chamfer_distance_two_sided}\n')
                f.write(f'hd: {hausdorff_dist}\n')
        print('Metrics saved!')

    def save_tda_plot(self, cc_barcodes):
        figure_1 = plt.figure()
        cc_diag = gd.plot_persistence_diagram(cc_barcodes)
        file_path = os.path.join(self.base_exp_dir, f"cc_{self.iter_step}.jpg")
        plt.savefig(file_path)
        plt.close(figure_1)


    def validate_mesh(self, resolution=64, threshold=0.0, point_gt=None, iter_step=0, logger=None, save_sdf=False):

        bound_min = torch.tensor(self.dataset_np.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset_np.object_bbox_max, dtype=torch.float32)
        os.makedirs(os.path.join(self.base_exp_dir, 'outputs'), exist_ok=True)
        mesh, vertices, sdf = self.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold, query_func=lambda pts: -self.sdf_network.sdf(pts))

        mesh.export(os.path.join(self.base_exp_dir, 'outputs', '{:0>8d}_{}.ply'.format(self.iter_step,str(threshold))))
        np.save(os.path.join(self.base_exp_dir, 'outputs', '{:0>8d}_{}.npy'.format(self.iter_step,str(threshold))), np.asarray(vertices))
        if save_sdf == True:
            np.save(os.path.join(self.base_exp_dir, 'outputs', '{:0>8d}_{}_sdf.npy'.format(self.iter_step,str(threshold))), np.asarray(sdf))


    def update_learning_rate_np(self, iter_step):
        warn_up = self.warm_up_end
        max_iter = self.maxiter
        init_lr = self.learning_rate
        lr =  (iter_step / warn_up) if iter_step < warn_up else 0.5 * (math.cos((iter_step - warn_up)/(max_iter - warn_up) * math.pi) + 1) 
        lr = lr * init_lr
        for g in self.optimizer.param_groups:
            g['lr'] = lr
    
    def extract_fields(self, bound_min, bound_max, resolution, query_func):
        N = 32
        X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
        Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
        Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

        u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
        with torch.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = torch.meshgrid(xs, ys, zs)
                        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                        val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                        u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
        return u

    def extract_geometry(self, bound_min, bound_max, resolution, threshold, query_func):
        print('Creating mesh with threshold: {}'.format(threshold))
        u = self.extract_fields(bound_min, bound_max, resolution, query_func)
        vertices, triangles = mcubes.marching_cubes(u, threshold)
        b_max_np = bound_max.detach().cpu().numpy()
        b_min_np = bound_min.detach().cpu().numpy()

        vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
        mesh = trimesh.Trimesh(vertices, triangles)

        return mesh, vertices, u

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        print(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name))
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        
        self.iter_step = checkpoint['iter_step']
            
    def save_checkpoint(self):
        checkpoint = {
            'sdf_network_fine': self.sdf_network.state_dict(),
            'iter_step': self.iter_step,
        }
        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))
    
        
if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/np_srb.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcubes_threshold', type=float, default=0.0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dir', type=str, default='gargoyle')
    parser.add_argument('--dataname', type=str, default='gargoyle')
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args, args.conf, args.mode)

    if args.mode == 'train' or 'train_tda':
        runner.train()
    elif args.mode == 'validate_mesh':
        threshs = [-0.001,-0.0025,-0.005,-0.01,-0.02,0.0,0.001,0.0025,0.005,0.01,0.02]
        for thresh in threshs:
            runner.validate_mesh(resolution=256, threshold=thresh)