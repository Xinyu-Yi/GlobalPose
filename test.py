import os
import torch
import tqdm
import numpy as np
import articulate as art
import matplotlib.pyplot as plt
from net import GPNet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MotionEvaluator:
    names = ['L SIP Err (deg)', 'L Angle Err (deg)', 'L Joint Err (cm)', 'L Vertex Err (cm)',
             'G SIP Err (deg)', 'G Angle Err (deg)', 'G Joint Err (cm)', 'G Vertex Err (cm)',
             'Root Jitter (km/s^3)', 'Joint Jitter (km/s^3)']

    def __init__(self):
        self._base_motion_loss_fn = art.FullMotionEvaluator('models/SMPL_male.pkl', joint_mask=torch.tensor([1, 2, 16, 17]), device=device)
        self.ignored_joint_mask = [7, 8, 10, 11, 20, 21, 22, 23]

    def __call__(self, pose_p, pose_t, tran_p, tran_t):
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)

        pose_p[:, self.ignored_joint_mask] = torch.eye(3, device=pose_p.device)
        pose_t[:, self.ignored_joint_mask] = torch.eye(3, device=pose_t.device)
        global_errs = self._base_motion_loss_fn(pose_p=pose_p, pose_t=pose_t, tran_p=tran_p, tran_t=tran_t)

        pose_p[:, 0] = torch.eye(3, device=pose_p.device)
        pose_t[:, 0] = torch.eye(3, device=pose_t.device)
        local_errs = self._base_motion_loss_fn(pose_p=pose_p, pose_t=pose_t)

        root_jitter = ((tran_p[3:] - 3 * tran_p[2:-1] + 3 * tran_p[1:-2] - tran_p[:-3]) * (60 ** 3)).norm(dim=1)
        root_jitter = torch.tensor([root_jitter.mean(), root_jitter.std()])

        return torch.stack([local_errs[9], local_errs[3], local_errs[0] * 100, local_errs[1] * 100,
                            global_errs[9], global_errs[3], global_errs[0] * 100, global_errs[1] * 100,
                            root_jitter / 1000, global_errs[4] / 1000])


class ResultLoader:
    def __init__(self, file, begin_frame=0):
        self.begin_frame = begin_frame
        self.seq_idx = -1
        self.frame_idx = -begin_frame - 1
        self.data = torch.load(file)
        # if 'tran' not in self.data.keys():
        #     pose = self.data['pose']
        #     tran = [torch.zeros(_.shape[0], 3) for _ in self.data['pose']]
        #     self.data = {'pose': pose, 'tran': tran}
        #     torch.save(self.data, file)

    def rnn_initialize(self, *args):
        self.seq_idx += 1
        self.frame_idx = -self.begin_frame - 1

    def forward_frame(self, *args):
        self.frame_idx += 1
        frame_idx = np.clip(self.frame_idx, 0, self.data['pose'][self.seq_idx].shape[0] - 1)
        return self.data['pose'][self.seq_idx][frame_idx], self.data['tran'][self.seq_idx][frame_idx]


def compare_realimu(data, dataset_name='', save_results_dir='data/temp/results', evaluate_pose=True, evaluate_tran=True):
    print('======================= Testing on %s Real Dataset =======================' % dataset_name)
    motion_evaluator = MotionEvaluator()
    g = torch.tensor([0, -9.8, 0])
    nets = {
        # 'DIP       ': ResultLoader('data/results/%s/dip.pt' % dataset_name),
        # 'TransPose ': ResultLoader('data/results/%s/transpose.pt' % dataset_name),
        # 'TIP       ': ResultLoader('data/results/%s/tip.pt' % dataset_name, begin_frame=30),
        # 'PIP       ': ResultLoader('data/results/%s/pip.pt' % dataset_name),
        # 'PNP       ': ResultLoader('data/results/%s/pnp.pt' % dataset_name),
        # 'DynaIP-X  ': ResultLoader('data/results/%s/dynaip_x.pt' % dataset_name, begin_frame=6),
        # 'DynaIP-XD ': ResultLoader('data/results/%s/dynaip_xd.pt' % dataset_name, begin_frame=6),
        # 'DynaIP-AD ': ResultLoader('data/results/%s/dynaip_ad.pt' % dataset_name, begin_frame=6),
        # 'AIP       ': ResultLoader('data/results/%s/aip_GR_OV.pt' % dataset_name),

        # 'AIP-G-OV  ': ResultLoader('data/results/%s/aip_G_OV.pt' % dataset_name),
        # 'AIP-B-OV  ': ResultLoader('data/results/%s/aip_B_OV.pt' % dataset_name),
        # 'AIP-GR-V  ': ResultLoader('data/results/%s/aip_GR_V.pt' % dataset_name),
        # 'AIP-GR-B  ': ResultLoader('data/results/%s/aip_GR_B.pt' % dataset_name),
        # 'AIP-nofuse': ResultLoader('data/results/%s/aip_nofuse.pt' % dataset_name),
        # 'AIP-nophys': ResultLoader('data/results/%s/aip_nophys.pt' % dataset_name),
        # 'AIP-noretrack': ResultLoader('data/results/%s/aip_noretrack.pt' % dataset_name),

        # 'PNP       ': PNP().eval().to(device),
        'GlobalPose': GPNet().eval().to(device),
        # 'AIP-G-OV  ': Full_G_OV().eval().to(device),
        # 'AIP-B-OV  ': Full_B_OV().eval().to(device),
        # 'AIP-GR-V  ': Full_GR_V().eval().to(device),
        # 'AIP-GR-B  ': Full_GR_B().eval().to(device),
        # 'AIP-nofuse': Full_GR_OV_NoFuse().eval().to(device),
        # 'AIP-nophys': Full_GR_OV_NoPhys().eval().to(device),
        # 'AIP-noretrack': Full_GR_OV_NoRetrack().eval().to(device),
    }
    pose_errors = {k: [] for k in nets.keys()}
    tran_errors = {k: {window_size: [] for window_size in list(range(1, 8))} for k in nets.keys()}
    pose_results = {k: [] for k in nets.keys()}
    tran_results = {k: [] for k in nets.keys()}

    for seq_idx in range(len(data['pose'])):
        aS = data['aS'][seq_idx]
        wS = data['wS'][seq_idx]
        mS = data['mS'][seq_idx]
        RIS = data['RIS'][seq_idx]
        RIM = data['RIM'][seq_idx]
        RSB = data['RSB'][seq_idx]
        tran = data['tran'][seq_idx]
        pose = data['pose'][seq_idx]

        RMB = RIM.transpose(1, 2).matmul(RIS).matmul(RSB).to(device)
        aM = (RIM.transpose(1, 2).matmul(RIS).matmul(aS.unsqueeze(-1)).squeeze(-1) + g).to(device)
        wM = RIM.transpose(1, 2).matmul(RIS).matmul(wS.unsqueeze(-1)).squeeze(-1).to(device)
        pose = art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)

        for net in nets.values():
            net.rnn_initialize(pose[0])
            net.pose_prediction = torch.zeros_like(pose)
            net.tran_prediction = torch.zeros_like(tran)

        for i in tqdm.trange(pose.shape[0]):
            for net in nets.values():
                net.pose_prediction[i], net.tran_prediction[i] = net.forward_frame(aM[i], wM[i], RMB[i])

        for k in nets.keys():
            pose_results[k].append(nets[k].pose_prediction)
            tran_results[k].append(nets[k].tran_prediction)

        if evaluate_pose:
            print('[%3d/%3d  pose]' % (seq_idx, len(data['pose'])), end='')
            for k in nets.keys():
                e = motion_evaluator(nets[k].pose_prediction, pose, nets[k].tran_prediction, tran)
                pose_errors[k].append(e)
                print('\t%s: %5.2fcm' % (k, e[2, 0]), end=' ')  # joint position error
            print('')

        if evaluate_tran:
            print('[%3d/%3d  tran]' % (seq_idx, len(data['pose'])), end='')

            # compute gt move distance at every frame
            move_distance_t = torch.zeros(tran.shape[0])
            v = (tran[1:] - tran[:-1]).norm(dim=1)
            for j in range(len(v)):
                move_distance_t[j + 1] = move_distance_t[j] + v[j]

            for k in nets.keys():
                for window_size in tran_errors[k].keys():
                    # find all pairs of start/end frames where gt moves `window_size` meters
                    frame_pairs = []
                    start, end = 0, 1
                    while end < len(move_distance_t):
                        if move_distance_t[end] - move_distance_t[start] < window_size:
                            end += 1
                        else:
                            if len(frame_pairs) == 0 or frame_pairs[-1][1] != end:
                                frame_pairs.append((start, end))
                            start += 1

                    # calculate mean distance error
                    errs = []
                    for start, end in frame_pairs:
                        vel_p = nets[k].tran_prediction[end] - nets[k].tran_prediction[start]
                        vel_t = tran[end] - tran[start]
                        errs.append((vel_t - vel_p).norm() / (move_distance_t[end] - move_distance_t[start]) * window_size)
                    if len(errs) > 0:
                        tran_errors[k][window_size].append(torch.stack(errs))
                print('\t%s: %5.2fm ' % (k, tran_errors[k][7][-1].mean()), end=' ')
            print('')

    print('======================= Results on %s Real Dataset =======================' % dataset_name)
    os.makedirs(save_results_dir, exist_ok=True)
    for k in nets.keys():
        torch.save({'pose': pose_results[k], 'tran': tran_results[k]}, os.path.join(save_results_dir, dataset_name + '_' + k.strip() + '.pt'))
    if evaluate_pose:
        print('Metrics: ', motion_evaluator.names)
        for net_name, error in pose_errors.items():
            error = torch.stack(error).mean(dim=0)
            print(net_name, end='\t')
            for error_item in error:
                print('%.2f±%.2f' % (error_item[0], error_item[1]), end='\t')  # mean & std
            print('')
    if evaluate_tran:
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Times New Roman'
        plt.figure(dpi=200)
        plt.grid(linestyle='-.')
        plt.ylim(0, 0.65)
        plt.xlim(0, 7.2)
        plt.xlabel('Real travelled distance (m)', fontsize=16)
        plt.ylabel('Mean translation error (m)', fontsize=16)
        plt.title('Cumulative Translation Error\n' + dataset_name, fontsize=18)
        for net_name in tran_errors.keys():
            x = np.array([0] + [_ for _ in tran_errors[net_name].keys()])
            y = np.array([0] + [torch.stack([_.mean() for _ in e]).mean().item() for e in tran_errors[net_name].values()])
            y_std = np.array([0] + [torch.stack([_.std() for _ in e]).mean().item() for e in tran_errors[net_name].values()])
            plt.plot(x, y, label=net_name)
            # plt.fill_between(x, y - y_std, y + y_std, alpha=0.1)
        plt.legend(fontsize=15)
        plt.show()
        print('Metrics: Translation Drift')
        print('\n'.join([k + '  %.2f%%' % (torch.stack([_.mean() for _ in v[7]]).mean().item() / 7 * 100) for k, v in tran_errors.items()]))


if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False)

    data = torch.load('data/test_datasets/totalcapture_officalib.pt')
    compare_realimu(data, dataset_name='TotalCapture (Official Calibration)')

    data = torch.load('data/test_datasets/totalcapture_dipcalib.pt')
    compare_realimu(data, dataset_name='TotalCapture (DIP Calibration)')

    data = torch.load('data/test_datasets/dipimu.pt')
    compare_realimu(data, dataset_name='DIP-IMU', evaluate_tran=False)

