from utils.vis_utils import load_pickle_data as read
from scipy.spatial.transform import Rotation as Rt
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
import numpy as np


def interp(func, gt_idx, pred_len, data):
    return func(gt_idx, data)(range(pred_len))

# load annotation from ho3d
seqlen = 500
path = '/mnt/8T/kangbo/HO3d/evaluation/MPM10/meta/'
all_rot = []
all_trans = []
for i in range(seqlen):
    anno = read(path + '{}.pkl'.format(str(i).zfill(4)))
    rot, trans = anno['objRot'], anno['objTrans']
    all_rot.append(rot.squeeze())
    all_trans.append(trans.squeeze())
    if i in [53, 54, 82, 83, 202, 203]:
        print(rot)
        print(np.linalg.norm(rot))
        rot_hpo = Rt.from_rotvec(rot.squeeze()).as_euler('xyz')
        rot_vec = Rt.from_euler('xyz', rot.squeeze()).as_rotvec()
        print(rot_vec)
        print(np.linalg.norm(rot_vec))
        print(rot_hpo)
        print(np.linalg.norm(rot_hpo))
        # input()

# create fake data
indices = list(range(0, seqlen, 10)) + [seqlen - 1]
trans = np.array(all_trans)[indices]
rot = np.array(all_rot)[indices]

# slerp, interp1d in scipy to interpolate rotation, transition
rots = interp(Slerp, indices, seqlen, Rt.from_rotvec(rot))
transis = np.c_[[interp(interp1d, indices, seqlen, trans[:, i]) for i in range(3)]].T

# compute rot error
rot_err = Rt.from_rotvec(all_rot) * rots.inv()
rot_err = np.linalg.norm(rot_err.as_rotvec(), axis=1) * 180 / np.pi

# compute trans error
trans_err = transis - all_trans
trans_err = np.linalg.norm(trans_err, axis=1)

# print the error and return the result
print(np.mean(rot_err), np.max(rot_err), np.argmax(rot_err))
print(np.mean(trans_err), np.max(trans_err), np.argmax(trans_err))
result = (rots.as_rotvec(), transis)

'''
print(rots.as_rotvec()[205])

def cpt_err(a, b):
    c = Rt.from_rotvec(a) * Rt.from_rotvec(b).inv()
    return np.linalg.norm(c.as_rotvec()) * 180 / np.pi

err = Rt.from_rotvec(all_rot[205]) * Rt.from_rotvec([1.45, 2.46, -0.283]).inv()
print(np.linalg.norm(err.as_rotvec()) * 180 / np.pi)
# 0.0024948468391458958 0.028688037248910586
# 1.45 2.46 -0.283

# -0.966, 0.466, 1.13
# -0.926, 0.526, 0.93
# -0.996, 0.474, 0.979
print(cpt_err([-0.966, 0.466, 1.13], [-0.99386, 0.42782, 1.0804]))
# 11.7 -> 8.25 -> 3.74 -> 3.61
print(cpt_err([-0.966, 0.466, 1.13], [-0.926, 0.526, 1.03]))
print(cpt_err([-0.966, 0.466, 1.13], [-0.987, 0.453, 1.073]))
# 6.67 -> 3.3
print(cpt_err([1.382, 2.255, -0.612], [1.435, 2.204, -0.6675]))
############################################################
'''