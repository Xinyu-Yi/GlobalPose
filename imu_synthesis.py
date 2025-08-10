import torch
import numpy as np
import articulate as art
import carticulate as cart


class Constants:
    gW = (0, -9.8, 0)
    mW = (1., 0, 0)
    v_imu = (1961, 5424, 1176, 4662, 411, 3021)
    j_imu = (18, 19, 4, 5, 15, 0)
    j_reduce = (1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19)
    j_ignore = (0, 7, 8, 10, 11, 20, 21, 22, 23)
    j_contact = (0, 10, 11, 22, 23)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
body_model = art.ParametricModel('models/SMPL_male.pkl', vert_mask=Constants.v_imu, device=device)


def _walking_noise(shape, std):
    return torch.cumsum(torch.normal(torch.zeros(shape), std), dim=0)


def _forward_smpl(pose, tran):
    pose = art.math.axis_angle_to_rotation_matrix(pose.view(-1, 24, 3)).view(-1, 24, 3, 3).to(device)
    tran = tran.view(-1, 3).to(device)
    grot = torch.empty(0, 24, 3, 3, device=device)
    joint = torch.empty(0, 24, 3, device=device)
    vert = torch.empty(0, 6, 3, device=device)
    for p, t in zip(pose.split(800), tran.split(800)):
        grot_, joint_, vert_ = body_model.forward_kinematics(p, None, t, calc_mesh=True)
        grot = torch.cat((grot, grot_), dim=0)
        joint = torch.cat((joint, joint_), dim=0)
        vert = torch.cat((vert, vert_), dim=0)
    return grot, joint, vert


def _syn_imu(p, R, skip_ESKF=True):
    from articulate.utils.imu import IMUSimulator

    # simulate IMU trajectory
    N = len(p)
    k = np.sqrt(np.pi / 8)
    RBS = art.math.generate_random_rotation_matrix(6).to(device)
    dp = _walking_noise(shape=(N, 6, 3), std=1e-3 * k * np.sqrt(1 / 60)) + torch.randn(6, 3) * 1e-2 * k
    dw = _walking_noise(shape=(N, 6, 3), std=1e-2 * k * np.sqrt(1 / 60)) + torch.randn(6, 3) * 1e-1 * k
    dR = art.math.axis_angle_to_rotation_matrix(dw).view(-1, 6, 3, 3)   # model calibration error as init error
    p_imu = p + R.matmul(dp.unsqueeze(-1).to(device)).squeeze(-1)
    R_imu = R.matmul(RBS).matmul(dR.to(device))

    # simulate IMU signals
    imu_simulator = IMUSimulator()
    imu_simulator.set_trajectory(p_imu, R_imu, fps=60)
    aS = imu_simulator.get_acceleration(gW=Constants.gW)
    wS = imu_simulator.get_angular_velocity()
    mS = imu_simulator.get_magnetic_field(mW=Constants.mW)

    # simulate IMU noise
    aS = torch.normal(aS, std=5e-2) + _walking_noise(shape=(N, 6, 3), std=1e-4 * np.sqrt(1 / 60)).view_as(aS).to(device)
    wS = torch.normal(wS, std=5e-3) + _walking_noise(shape=(N, 6, 3), std=1e-5 * np.sqrt(1 / 60)).view_as(wS).to(device)
    mS = torch.normal(mS, std=5e-3) + _walking_noise(shape=(N, 6, 3), std=1e-5 * np.sqrt(1 / 60)).view_as(mS).to(device)

    # simulate IMU ESKF
    R_sim = torch.empty(N, 6, 3, 3)
    if not skip_ESKF:
        for i in range(6):
            eskf = cart.ESKF(an=5e-2, wn=5e-3, aw=1e-4, ww=1e-5, mn=5e-3)
            eskf.initialize_9dof(RIS=R_imu[0, i].cpu().numpy(), gI=np.array(Constants.gW), nI=np.array(Constants.mW))
            for j in range(N):
                eskf.predict(am=aS[j, i].cpu().numpy(), wm=wS[j, i].cpu().numpy(), dt=1 / 60)
                eskf.correct(am=aS[j, i].cpu().numpy(), wm=wS[j, i].cpu().numpy(), mm=mS[j, i].cpu().numpy())
                R_sim[j, i] = torch.from_numpy(eskf.get_orientation_R())
    else:
        # angular velocity integration, much faster for approximate training
        dR = art.math.axis_angle_to_rotation_matrix(wS / 60).view(-1, 6, 3, 3).cpu()
        R_sim[0] = R_imu[0].cpu()
        for i in range(1, N):
            R_sim[i] = R_sim[i - 1].matmul(dR[i])

    # add Gaussian noise
    nR = art.math.axis_angle_to_rotation_matrix(torch.randn(N, 6, 3) * 0.1 * k).view(-1, 6, 3, 3)
    R_sim = R_sim.matmul(nR)

    # simulate T-pose calibration
    R_sim = R_sim.to(device)
    a_sim = R_sim.matmul(aS.unsqueeze(-1)).squeeze(-1) + torch.tensor(Constants.gW, device=device)
    w_sim = R_sim.matmul(wS.unsqueeze(-1)).squeeze(-1)
    R_sim = R_sim.matmul(RBS.transpose(1, 2))
    return a_sim, w_sim, R_sim


def syn_imu_from_smpl(pose, tran):
    grot, joint, vert = _forward_smpl(pose, tran)
    p, R = vert, grot[:, Constants.j_imu]
    a_sim, w_sim, R_sim = _syn_imu(p, R)

    # # global frame to root frame
    # a_sim = a_sim.bmm(R_sim[:, 5])
    # w_sim = w_sim.bmm(R_sim[:, 5])
    # R_sim = R_sim[:, 5:].transpose(-1, -2).matmul(R_sim[:, :5])

    return a_sim, w_sim, R_sim


if __name__ == '__main__':
    example_data = torch.load('data/test_datasets/totalcapture_dipcalib.pt')
    smpl_pose = example_data['pose'][0]
    smpl_tran = example_data['tran'][0]
    a, w, R = syn_imu_from_smpl(smpl_pose, smpl_tran)
    print('a:', a.shape)
    print('w:', w.shape)
    print('R:', R.shape)


