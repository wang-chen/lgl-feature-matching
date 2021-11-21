import torch
import numpy as np
import pickle, bz2
import torch.nn as nn
from collections import deque
from .geometry import Projector
import kornia.geometry.conversions as C
from prettytable import PrettyTable, MARKDOWN
from models import ConsecutiveMatch, GridSample


class MatchEvaluator():
    def __init__(self, back=[1], viz=None, viz_dist_min=0, viz_dist_max=100, top=None, writer=None, counter=None):
        self.back, self.top = back, top
        self.viz, self.viz_min, self.viz_max = viz, viz_dist_min, viz_dist_max
        self.hist = []
        self.grid_sample = GridSample()
        self.match = ConsecutiveMatch()
        self.error = {b: [] for b in back}
        self.writer = writer
        self.counter = counter if counter is not None else GlobalStepCounter()
        self.cur_env = None
        self.env_err_seg = {}

    @torch.no_grad()
    def observe(self, descriptors, points, scores, score_map, depth_map, poses, Ks, imgs, env_seq):
        B, N, _ = points.shape
        _, _, H, W = imgs.shape
        top = N if self.top is None else self.top

        env_seq = "_".join(zip(*env_seq).__next__())
        if self.cur_env != env_seq:
            last_env, self.cur_env = self.cur_env, env_seq
            self.hist = []
            n_batches = len(self.error[self.back[0]])
            self.env_err_seg[self.cur_env] = [n_batches, -1]
            if last_env is not None:
                self.env_err_seg[last_env][1] = n_batches

        # populate hist until sufficient
        depths = self.grid_sample((depth_map, points)).squeeze(-1)
        for img, desc, pt, pose, K, depth in zip(imgs, descriptors, points, poses, Ks, depths):
            self.hist.insert(0, (img, desc, pt, pose, K, depth))
        if len(self.hist) - B < max(self.back):
            return

        viz_img1, viz_img2, viz_pts1, viz_pts2, viz_val = [], [], [], [], []
        imgs_new, descs_new, pts_new, poses_new, Ks_new, depths_new = self._unpack_hist(reversed(self.hist[:B]))
        for b in self.back:
            imgs_old, descs_old, pts_old, poses_old, Ks_old, depths_old = self._unpack_hist(reversed(self.hist[b:b+B]))
            matched_new_pt, conf = self.match(descs_old, descs_new, pts_new)
            match_conf, top_idx = conf.topk(top)
            match_src, match_dst = self._point_select(pts_old, top_idx), self._point_select(matched_new_pt, top_idx)
            depths_old = self._point_select(depths_old, top_idx)
            error = reproj_error(match_src, Ks_old, poses_old, depths_old, match_dst, Ks_new, poses_new, H, W)
            self.error[b].append(error)
            if self.viz is not None:
                viz_img1.append(imgs_old), viz_img2.append(imgs_new)
                viz_pts1.append(match_src), viz_pts2.append(match_dst)
                viz_val.append(error)

        if self.viz is not None:
            # B, back, *
            viz_img1, viz_img2 = torch.stack(viz_img1, dim=1), torch.stack(viz_img2, dim=1)
            viz_pts1, viz_pts2 = torch.stack(viz_pts1, dim=1), torch.stack(viz_pts2, dim=1)
            viz_val = torch.stack(viz_val, dim=1).detach().cpu().numpy()
            for img1, img2, pts1, pts2, val in zip(viz_img1, viz_img2, viz_pts1, viz_pts2, viz_val):
                self.viz.showmatch(img1, pts1, img2, pts2, 'hot', val, self.viz_min, self.viz_max, name='backs')
                break

        self.hist = self.hist[:-B]

        return self.error

    def ave_reproj_error(self, quantile=None, env=None):
        mean, quantiles = {}, {}
        for b in self.back:
            seg = self.env_err_seg[env] if env is not None else [0, -1]
            error = torch.cat(self.error[b][seg[0]:seg[1]])
            mean[b] = error.mean().item()
            if quantile is not None:
                quantiles[b] = torch.quantile(error, torch.tensor(quantile).to(error)).tolist()
        return (mean, quantiles) if quantile is not None else mean

    def ave_prec(self, thresh=1, env=None):
        perc = {}
        for b in self.back:
            seg = self.env_err_seg[env] if env is not None else [0, -1]
            perc[b] = (torch.cat(self.error[b][seg[0]:seg[1]]) < 1).to(torch.float).mean().item()
        return perc

    def report(self, err_quant=[0.5, 0.9]):
        result = PrettyTable(['Env Name', 'n-back',
                              'Mean Err (%s)' % self._fmt_list(np.array(err_quant) * 100, '%d%%'), 'Ave Prec'])
        result.float_format['Ave Prec'] = '.2'
        result.align['Env Name'] = 'r'

        all_mean, all_quantiles = self.ave_reproj_error(quantile=err_quant)
        prec = self.ave_prec()
        result.add_rows([['All', b, '%.2f (%s)' % (all_mean[b], self._fmt_list(all_quantiles[b], '%.2f')),
                          prec[b]] for b in self.back])

        for e in self.env_err_seg:
            env_mean, env_quantiles = self.ave_reproj_error(quantile=err_quant, env=e)
            env_prec = self.ave_prec(env=e)
            result.add_rows([[e, b, '%.2f (%s)' % (env_mean[b], self._fmt_list(env_quantiles[b], '%.2f')),
                              env_prec[b]] for b in self.back])

        n_iter = self.counter.steps
        if self.writer is not None:
            self.writer.add_scalars('Eval/Match/MeanErr', {'%d-back' % b: v for b, v in all_mean.items()}, n_iter)
            self.writer.add_scalars('Eval/Match/AvePrec', {'%d-back' % b: v for b, v in prec.items()}, n_iter)
            for b in self.back:
                self.writer.add_histogram('Eval/Match/LogErr/%d-back' % b,
                                          torch.log10(torch.cat(self.error[b]).clamp(min=1e-10)), n_iter)
            # TensorBoard supports markdown table although column alignment is broken
            result.set_style(MARKDOWN)
            self.writer.add_text('Eval/Match/PerSeq', result.get_string(sortby='n-back'), n_iter)
        else:
            print('Evaluation: step %d' % n_iter)
            print(result.get_string(sortby='n-back'))

        # clear history
        self.error = {b: [] for b in self.back}
        self.hist = []
        self.cur_env = None
        self.env_err_seg = {}

    @staticmethod
    def _fmt_list(elems, fmt, delim=', '):
        return delim.join([fmt % e for e in elems])

    @staticmethod
    def _unpack_hist(hist):
        return [torch.stack(attr, dim=0) for attr in zip(*hist)]

    @staticmethod
    def _point_select(attr, idx):
        B, N = idx.shape
        return attr.gather(1, idx.view(B, N, *([1] * (len(attr.shape) - 2))).expand(B, N, *attr.shape[2:]))

    def save_error(self, file_path):
        error = {k: [v.detach().cpu().numpy() for v in vs] for k, vs in self.error.items()}
        with bz2.BZ2File(file_path, 'wb') as f:
            pickle.dump([error, self.env_err_seg], f)


def reproj_error(pts_src, K_src, pose_src, depth_src, pts_dst, K_dst, pose_dst, H, W):
    projector = Projector()
    cam_src = Projector._make_camera(H, W, K_src, pose_src)
    cam_dst = Projector._make_camera(H, W, K_dst, pose_dst)
    pts_prj, _ = Projector._project_points(pts_src, depth_src, cam_src, cam_dst)
    diff = C.denormalize_pixel_coordinates(pts_prj, H, W) - C.denormalize_pixel_coordinates(pts_dst, H, W)
    return torch.hypot(diff[..., 0], diff[..., 1])
