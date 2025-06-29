# This implementation is adapted from Dongliang Cao, et al. (2024): https://github.com/dongliangcao/unsupervised-learning-of-robust-spectral-shape-matching. New modules & refinements are added

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.distributed as dist

from src.metric import BaseMetric
from src.infra.registry import METRIC_REGISTRY
from src.utils.fmap import fmap2pointmap
from src.utils.tensor import to_numpy


@METRIC_REGISTRY.register()
class GeodesicDist(BaseMetric):
    def __init__(self, pck_threshold=0.10, pck_steps=40):
        super(GeodesicDist, self).__init__()
        self.pck_threshold = pck_threshold
        self.pck_steps = pck_steps

    def calculate_geodesic_error(self, dist_x, corr_x, corr_y, p2p, return_mean=True):
        """
        Calculate the geodesic error between predicted correspondence and gt correspondence

        Args:
            dist_x (np.ndarray): Geodesic distance matrix of shape x. shape [Vx, Vx]
            corr_x (np.ndarray): Ground truth correspondences of shape x. shape [V]
            corr_y (np.ndarray): Ground truth correspondences of shape y. shape [V]
            p2p (np.ndarray): Point-to-point map (shape y -> shape x). shape [Vy]
            return_mean (bool, optional): Average the geodesic error. Default True.
        Returns:
            avg_geodesic_error (np.ndarray): Average geodesic error.
        """
        ind21 = np.stack([corr_x, p2p[corr_y]], axis=-1)
        ind21 = np.ravel_multi_index(ind21.T, dims=[dist_x.shape[0], dist_x.shape[0]])
        geo_err = np.take(dist_x, ind21)
        if return_mean:
            return geo_err.mean()
        else:
            return geo_err

    def pck_and_auc(self, dists):
        # compute pck
        dists = np.ravel(dists)
        thresholds = np.linspace(0., self.pck_threshold, self.pck_steps)
        pck_ls = []

        for i in range(thresholds.shape[0]):
            thres = thresholds[i]
            pck = np.mean((dists <= thres).astype(float))
            pck_ls.append(pck)

        pck_arr = np.array(pck_ls)

        # compute auc
        auc = np.trapz(pck_arr, np.linspace(0., 1., self.pck_steps))
        return pck_arr, auc

    def forward(self, Cxy, evecs_x, evecs_y, dist_x, corr_x, corr_y):
        num = Cxy.shape[0]
        dist_sum = 0.0
        auc_sum = 0.0
        pck_arr_sum = np.zeros(self.pck_steps)

        for it in range(num):
            p2p = fmap2pointmap(Cxy[it], evecs_x[it], evecs_y[it])
            dists = self.calculate_geodesic_error(
                to_numpy(dist_x[it]),
                to_numpy(corr_x[it]),
                to_numpy(corr_y[it]),
                to_numpy(p2p),
                return_mean=False,
            )
            pck, auc = self.pck_and_auc(dists)

            dist_sum += dists.mean()
            auc_sum += auc
            pck_arr_sum += pck

        return dist_sum, auc_sum, pck_arr_sum

    def start_feed(self, script, name, rank=None):
        """Method to start feeding the metric method. Typically called at the beginning of the testing/benchmark epoch
        
        Args:
            script: The traning/benchmark script object
            name: The name of the metric method
            rank: The rank of the process (if using distributed training)
        """
        self.script = script
        self.name = name
        self.rank = rank

        # if self.eval() is called, prep for total metric gathering
        if not self.training:
            self.metric_total = 0.0
            self.metric_avg = 0.0
            self.auc_total = 0.0
            self.pck_arr_total = np.zeros(self.pck_steps)
            self.sample_total = 0

    def feed(self, infer, data):
        Cxy = infer['Cxy']
        evecs_x = data['first']['evecs']
        evecs_y = data['second']['evecs']
        dist_x = data['first']['dist']
        corr_x = data['first']['corr']
        corr_y = data['second']['corr']
        dist_sum, auc_sum, pck_arr_sum = self(Cxy, evecs_x, evecs_y, dist_x, corr_x, corr_y)

        # if self.eval() is called, gather total metric
        if not self.training:
            self.metric_total += dist_sum
            self.auc_total += auc_sum
            self.pck_arr_total += pck_arr_sum
            self.sample_total += len(Cxy)

    def end_feed(self):
        # if self.eval() is called, log avg metric
        if not self.training:
            if self.rank is None:
                # if not using distributed training, simply log the average loss
                self.metric_avg = self.metric_total / self.sample_total
                self.script.writer.add_scalar(f'metric/test/{self.name}', self.metric_avg, self.script.global_step)

                self.script.writer.add_scalar(f'metric/test/{self.name}/auc', self.auc_total / self.sample_total, self.script.global_step)

                fig = plt.figure()
                plt.plot(
                    np.linspace(0., self.pck_threshold, self.pck_steps),
                    self.pck_arr_total / self.sample_total,
                )
                plt.xlabel('geodist')
                plt.ylabel('ratio')
                plt.grid(linestyle='--')
                self.script.writer.add_figure(f'metric/test/{self.name}/pck', fig, self.script.global_step)
        
            else:
                # if using distributed training, gather the total loss and sample count across all ranks
                self.metric_total = torch.tensor(self.metric_total).to(device=self.rank)
                self.auc_total = torch.tensor(self.auc_total).to(device=self.rank)
                self.pck_arr_total = torch.tensor(self.pck_arr_total).to(device=self.rank)
                self.sample_total = torch.tensor(self.sample_total).to(device=self.rank)
                dist.all_reduce(self.metric_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(self.auc_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(self.pck_arr_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(self.sample_total, op=dist.ReduceOp.SUM)

                if self.rank == 0:
                    # only log in rank 0
                    self.metric_avg = self.metric_total / self.sample_total
                    self.script.writer.add_scalar(f'metric/test/{self.name}', self.metric_avg, self.script.global_step)

                    self.script.writer.add_scalar(f'metric/test/{self.name}/auc', self.auc_total / self.sample_total, self.script.global_step)

                    fig = plt.figure()
                    plt.plot(
                        np.linspace(0., self.pck_threshold, self.pck_steps),
                        (self.pck_arr_total / self.sample_total).cpu().numpy(),
                    )
                    plt.xlabel('geodist')
                    plt.ylabel('ratio')
                    plt.grid(linestyle='--')
                    self.script.writer.add_figure(f'metric/test/{self.name}/pck', fig, self.script.global_step)
