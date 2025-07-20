import os
import keyboard
import torch
import articulate as art
from articulate.utils.noitom import CalibratedIMUSet
from pygame.time import Clock
from net import GPNet
from datetime import datetime


GPNet.Visualization.enable = True
GPNet.Visualization.show_stationary = False
GPNet.Visualization.show_contact = False
GPNet.Visualization.show_contact_force = False
GPNet.Visualization.show_residual_force = False
GPNet.Visualization.show_block = False
GPNet.Visualization.show_torque = False

data = []
clock = Clock()
net = GPNet().cuda()
net.rnn_initialize(torch.load('data/Ipose.pt'))
imus = CalibratedIMUSet()
imus.calibrate('walking_6dof')

while True:
    clock.tick(60)
    t, RIS, aS, wS, mS, aI, wI, mI, RMB, aM, wM, mM = imus.get()
    RMB = art.math.normalize_rotation_matrix(RMB).view_as(RMB)
    pose, tran = net.forward_frame(aM.cuda(), wM.cuda(), RMB.cuda())
    data.append([t, RIS, aS, wS, mS, aI, wI, mI, RMB, aM, wM, mM, pose, tran])
    print('\r', clock.get_fps(), end='')
    if keyboard.is_pressed('r'):
        net.tran_offset[0] = tran[0]
        net.tran_offset[2] = tran[2]
    if keyboard.is_pressed('esc'):
        break

file = 'data/records/' + datetime.now().strftime('%Y%m%d/%H%M%S') + '.pt'
os.makedirs(os.path.dirname(file), exist_ok=True)
torch.save(data, file)
print('\rSaved at', file)
