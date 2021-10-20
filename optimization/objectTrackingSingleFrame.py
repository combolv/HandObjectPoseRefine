import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0,os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))

from ghope.common import *
from ghope.scene import Scene
from ghope.rendering import DirtRenderer
from ghope.loss import LossObservs
from ghope.optimization import Optimizer
from ghope.constraints import Constraints
from ghope.icp import Icp

from ghope.utils import *
from ghope.vis import renderScene

from HOdatasets.ho3d_multicamera.dataset import datasetHo3dMultiCamera
from HOdatasets.commonDS import datasetType, splitType
from HOdatasets.mypaths import *

import argparse, yaml
from ext.mesh_loaders import *
import matplotlib.pyplot as plt

depthScale = 0.00012498664727900177 # constant depth scale used throughout the project
bgDepth = 2.0
handLabel = 2
DEPTH_THRESH = 0.75

datasetName = datasetType.HO3D_MULTICAMERA

from absl import flags
from absl import app
# import argparse
# parser = argparse.ArgumentParser()
# FLAGS = flags.FLAGS
#
# parser.add_argument('--seq', default='0010', help='Sequence Name')
# parser.add_argument('--camID', default='0', help='Sequence Name')
# parser.add_argument('--doPyRender', action='store_true', help='Show object rendering, very slow!')
# FLAGS = parser.parse_args()
#
# USE_PYTHON_RENDERER = FLAGS.doPyRender # for visualization, but slows down


def my_objectTracker(w, h, rot, trans, camProp, objMesh, out_dir,
                     frameID, objMask, objDepth, objImg, handMask):
    ds = tf.data.Dataset.from_generator(lambda: dataGen(frameID, objMask, objDepth, objImg, handMask),
                                        (tf.string, tf.float32, tf.float32, tf.float32, tf.float32),
                                        ((None,), (None, h, w, 3), (None, h, w, 3), (None, h, w, 3), (None, h, w, 3)))
    numFrames = 1

    # read real observations
    frameCntInt, loadData, realObservs = LossObservs.getRealObservables(ds, numFrames, w, h)
    icp = Icp(realObservs, camProp)

    # set up the scene
    scene = Scene(optModeEnum.MULTIFRAME_JOINT, frameCnt=1)
    objID = scene.my_addObject(objMesh, rot, trans, segColor=np.array([1.,1.,1.]))
    scene.addCamera(f=camProp.f, c=camProp.c, near=camProp.near, far=camProp.far, frameSize=camProp.frameSize)
    finalMesh = scene.getFinalMesh()

    # render the scene
    renderer = DirtRenderer(finalMesh, renderModeEnum.SEG_DEPTH)
    virtObservs = renderer.render()

    # get loss over observables
    observLoss = LossObservs(virtObservs, realObservs, renderModeEnum.SEG_DEPTH)
    segLoss, depthLoss, _ = observLoss.getL2Loss(isClipDepthLoss=True, pyrLevel=2)

    # get constraints
    handConstrs = Constraints()
    paramList = scene.getParamsByItemID([parTypeEnum.OBJ_ROT, parTypeEnum.OBJ_TRANS, parTypeEnum.OBJ_POSE_MAT], objID)
    rot = paramList[0]
    trans = paramList[1]
    poseMat = paramList[2]

    # get icp loss
    icpLoss = icp.getLoss(finalMesh.vUnClipped)

    # get final loss
    objImg = (realObservs.col)
    # totalLoss1 = 1.0*segLoss + 1e1*depthLoss + 1e4*icpLoss + 0.0*tf.reduce_sum(objImg-virtObservs.seg)
    totalLoss1 = 1.0e0 * segLoss + 1e1 * depthLoss  + 1e2 * icpLoss + 0.0 * tf.reduce_sum(objImg - virtObservs.seg)
    totalLoss2 = 1.15 * segLoss + 5.0 * depthLoss + 500.0*icpLoss

    # get the variables for opt
    optVarsList = scene.getVarsByItemID(objID, [varTypeEnum.OBJ_ROT, varTypeEnum.OBJ_TRANS])

    # setup optimizer
    opti1 = Optimizer(totalLoss1, optVarsList, 'Adam', learning_rate=0.02/2.0)
    opti2 = Optimizer(totalLoss2, optVarsList, 'Adam', learning_rate=0.005)
    optiICP = Optimizer(1e1*icpLoss, optVarsList, 'Adam', learning_rate=0.01)

    # get the optimization reset ops
    resetOpt1 = tf.variables_initializer(opti1.optimizer.variables())
    resetOpt2 = tf.variables_initializer(opti2.optimizer.variables())


    # tf stuff
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    session = tf.Session(config=config)
    session.__enter__()
    tf.global_variables_initializer().run()

    # setup the plot window
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    lGT = ax1.imshow(np.zeros((240,320,3), dtype=np.float32))
    ax2 = fig.add_subplot(2, 2, 2)
    lRen = ax2.imshow(np.zeros((240, 320, 3), dtype=np.float32))
    ax3 = fig.add_subplot(2, 2, 3)
    lDep = ax3.imshow(np.random.uniform(0,2,(240,320)))
    ax4 = fig.add_subplot(2, 2, 4)
    lMask = ax4.imshow(np.random.uniform(0,2,(240,320,3)))

    session.run(resetOpt1)
    session.run(resetOpt2)

    # load new frame
    opti1.runOptimization(session, 1, {loadData:True})
    print(icpLoss.eval(feed_dict={loadData: False}))
    print(segLoss.eval(feed_dict={loadData: False}))
    print(depthLoss.eval(feed_dict={loadData: False}))

    # run the optimization for new frame
    frameID = (realObservs.frameID.eval(feed_dict={loadData: False}))[0].decode('UTF-8')
    # opti1.runOptimization(session, 200, {loadData: False})#, logLossFunc=True, lossPlotName=out_dir+'/LossFunc/'+frameID+'_1.png')
    opti2.runOptimization(session, 25, {loadData: False})#, logLossFunc=True, lossPlotName='handLoss/'+frameID+'_2.png')

    plt.title(frameID)
    depRen = virtObservs.depth.eval(feed_dict={loadData: False})[0]
    depGT = realObservs.depth.eval(feed_dict={loadData: False})[0]
    segRen = virtObservs.seg.eval(feed_dict={loadData: False})[0]
    segGT = realObservs.seg.eval(feed_dict={loadData: False})[0]

    lGT.set_data(objImg.eval(feed_dict={loadData: False})[0]) # input image
    # if USE_PYTHON_RENDERER:
    #     lRen.set_data(cRend) # object rendered in the optimized pose
    lDep.set_data(np.abs(depRen-depGT)[:,:,0]) # depth map error
    lMask.set_data(np.abs(segRen-segGT)[:,:,:]) # mask error
    plt.savefig(out_dir+'/'+frameID+'.png')
    plt.waitforbuttonpress(0.01)

    transNp = trans.eval(feed_dict={loadData: False})
    rotNp = rot.eval(feed_dict={loadData: False})
    savePickleData(out_dir+'/'+frameID+'.pkl', {'rot': rotNp, 'trans': transNp})
    print(rotNp, transNp)


def dataGen(frameID, objMask, objDepth, objImg, handMask):
    return (tf.float32(frameID),
            tf.float32(objMask),
            tf.float32(objDepth),
            tf.float32(objImg),
            tf.float32(handMask))