from utils.vis_utils import *
from optimization.objectTrackingSingleFrame import my_objectTracker
from optimization.ext.mesh_loaders import load_mesh
import cv2
import os


class camProps(object):
    def __init__(self, ID, f, c, near, far, frameSize, pose):
        self.ID = ID
        self.f = f
        self.c = c
        self.near = near
        self.far = far
        self.frameSize = frameSize
        self.pose = pose

    def getCamMat(self):
        camMat = np.array([[self.f[0], 0, self.c[0]],[0., self.f[1], self.c[1]],[0., 0., 1.]]).astype(np.float32)
        return camMat


import argparse
parser = argparse.ArgumentParser()
# FLAGS = flags.FLAGS
#
parser.add_argument('--ycb', default='/mnt/8T/kangbo/ycb/models', type=str)
parser.add_argument('--ho3d', default='/mnt/8T/kangbo/', type=str)
parser.add_argument('--pkl', default='/mnt/8T/kangbo/HO3d/evaluation/SM1/meta/0000.pkl', type=str)
parser.add_argument('--frameID', default=0, type=int)
# parser.add_argument('--objDepth', default='seg.png', type=str)
parser.add_argument('--handMask', default='seg.png', type=str)
parser.add_argument('--depth-filename', default='/mnt/8T/kangbo/HO3d/evaluation/SM1/depth/0000.png', type=str)
parser.add_argument('--img-filename', default='/mnt/8T/kangbo/HO3d/evaluation/SM1/rgb/0000.png', type=str)
parser.add_argument('--objMask', default='seg.png', type=str)

args = parser.parse_args()
YCB_MODELS_DIR = args.ycb
HO3D_MULTI_CAMERA_DIR = args.ho3d
meta_filename = args.pkl
#
# USE_PYTHON_RENDERER = FLAGS.doPyRender # for visualization, but slows down



def pred_func(img, anno):
    h, w, _ = img.shape # assert img.shape == (640, 480, 3)

    rot, trans = anno['objRot'], anno['objTrans']
    objMesh = read_obj(os.path.join(YCB_MODELS_DIR, 'models', anno['objName'], 'textured_simple.obj'))



def track():
    anno = load_pickle_data(meta_filename)

    # configFile = os.path.join(HO3D_MULTI_CAMERA_DIR, 'test', 'configs/configObjPose.json')
    # with open(configFile) as config_file:
    #     configData = yaml.safe_load(config_file)
    base_dir = os.path.join(HO3D_MULTI_CAMERA_DIR, 'test')
    out_dir = os.path.join(base_dir, 'dirt_obj_pose')

    # create out dir
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    rot = anno['objRot']
    trans = anno['objTrans']

    modelPath = os.path.join(YCB_MODELS_DIR, anno['objName'], 'textured_simple.obj')
    mesh = load_mesh(modelPath)

    # ready the arguments
    w, h = 480, 640
    camMat = anno['camMat']
    camProp = camProps(ID='cam1', f=np.array([camMat[0,0], camMat[1,1]], dtype=np.float32),
                           c=np.array([camMat[0,2], camMat[1,2]], dtype=np.float32),
                           near=0.001, far=2.0, frameSize=[w, h],
                           pose=np.eye(4, dtype=np.float32))

    objImg = cv2.imread(args.img_filename)

    depth_scale = 0.00012498664727900177
    depth_img = cv2.imread(args.depth_filename)

    dpt = depth_img[:, :, 2] + depth_img[:, :, 1] * 256
    objDepth = dpt * depth_scale
    objMask = cv2.imread(args.objMask)
    handMask = cv2.imread(args.handMask)

    print(rot, trans)
    my_objectTracker(w, h, rot, trans, camProp, mesh, out_dir,
                     args.frameID, objMask, objDepth, objImg, handMask)


if __name__ == '__main__':
    track()