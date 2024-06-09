import cv2
import time
import numpy as np
from util import kinect
import sys

from pyqtgraph.Qt import QtCore
import pyqtgraph as pg
from PyQt5 import uic, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel

N=5                 #脊椎の分割数
GRAPH_WIDTH=150     #グラフの横軸の数
FPS = 30            #フレームレート
DEPTH_WIDTH = 180     #procImageBox.frameGeometry().width()
COLOR_WIDTH = 280
COLOR_HEIGHT = 240
dt = 1/FPS

frame_times = []

app = pg.mkQApp("Measure")
win = uic.loadUi("./ui/recorder_v1.0.ui")
pg.setConfigOptions(antialias=True)

colorImageBox = win.ColorImage
procImageBox = win.ProcImage

procImageBox.setScene(QtWidgets.QGraphicsScene(0, 0, DEPTH_WIDTH, DEPTH_WIDTH, procImageBox) )
procImageBox.setRenderHint(QtGui.QPainter.Antialiasing, False)
colorImageBox.setScene(QtWidgets.QGraphicsScene(0, 20, COLOR_WIDTH, COLOR_HEIGHT, colorImageBox) )
colorImageBox.setRenderHint(QtGui.QPainter.Antialiasing, False)

figure1 = win.graph1.addPlot(title="graph 1")
figure2 = win.graph2.addPlot(title="graph 2")
plot1 = figure1.plot(pen="y")

figure2.showAxis('right')
figure2.getAxis('left').setLabel('max', color='#f00')
figure2.getAxis('right').setLabel('average', color='#ff0')
fig2_ave = pg.ViewBox()
figure2.scene().addItem(fig2_ave)
figure2.getAxis('right').linkToView(fig2_ave)
fig2_ave.setXLink(figure2)
def updateViews():
  fig2_ave.setGeometry(figure2.vb.sceneBoundingRect())
  fig2_ave.linkedViewChanged(figure2.vb, fig2_ave.XAxis)
  
updateViews()
figure2.vb.sigResized.connect(updateViews)


T=[]
Y_MAX = []
Y_AVE = []
ptr=1

kinect = kinect.Kinect()

def update():
  global ptr, kinect, start_time, T, Y_MAX, Y_AVE, plot1, plot2
  
  frame_start = time.time()
  #depth_img, (neck, pelvis, left_shoulder, right_shoulder), color_image = kinect.update()
  kinect.update()
  (neck, pelvis, left_shoulder, right_shoulder) = kinect.joints
  depth_img = kinect.depth_img
  color_image = kinect.color_img
  
  
  #region 深度マップと胴体位置のクロッピング(頭部や複数検出時の誤作動除去)
  h,w = depth_img.shape
  vertical_spine = int( max( abs(neck[0] - pelvis[0]), abs(neck[1] - pelvis[1]) ) /2 )
  center_spine = (neck+pelvis)//2
  left_top = (center_spine - vertical_spine).astype(np.uint16)
  right_bottom = (center_spine + vertical_spine).astype(np.uint16)
  cropped_depth = depth_img[
    max(0,left_top[1]) : min(w,right_bottom[1]),
    max(0,left_top[0]) : min(h,right_bottom[0])
  ]
  cropped_neck, cropped_pelvis = neck-left_top, pelvis-left_top
  cropped_left_shoulder, cropped_right_shoulder = left_shoulder-left_top, right_shoulder-left_top
  #endregion
  
  #胴体の傾きの取得
  spine_vec = cropped_neck - cropped_pelvis
  slope = spine_vec[1]/spine_vec[0]

  #計算用の座標グリッド
  x_line = np.arange(cropped_depth.shape[0])
  y_line = np.arange(cropped_depth.shape[1])
  x_grid,y_grid = np.meshgrid(x_line, y_line)
  
  #領域分割用の１次関数
  gradation_split = x_grid + slope*y_grid
  split_pos = np.stack((np.linspace(cropped_neck[0], cropped_pelvis[0], N+1), np.linspace(cropped_neck[1], cropped_pelvis[1], N+1)), 1)

  #重み付け用の１次関数（軸からの距離）  
  gradation_distance = (y_grid-cropped_neck[1]) - (x_grid-cropped_neck[0])*slope
  
  #1時間数に沿った肩幅位置
  distance_right_shoulder = (cropped_right_shoulder[1]-cropped_neck[1]) - slope * (cropped_right_shoulder[0]-cropped_neck[0])
  distance_left_shoulder = (cropped_left_shoulder[1]-cropped_neck[1]) - slope * (cropped_left_shoulder[0]-cropped_neck[0])
  
  depth_torso = np.where(
    ((gradation_distance > distance_right_shoulder)&(gradation_distance < distance_left_shoulder) | 
     (gradation_distance < distance_right_shoulder)&(gradation_distance > distance_left_shoulder)),
    cropped_depth, 0.)
  
  #縦領域の選定
  border1, border2 = split_pos[3], split_pos[4]
  threshold1 = border1[0] + border1[1]*slope
  threshold2 = border2[0] + border2[1]*slope
  mask = np.where( (gradation_split > threshold1) ^ (gradation_split > threshold2), True, False)
  region = (depth_torso > 0.0) & mask
  target_depths = depth_torso[region]
  
  T.append(time.time()-start_time)
  Y_MAX.append(depth_torso.mean())
  Y_AVE.append(target_depths.mean()-depth_torso.mean())
  
  if(len(T)>GRAPH_WIDTH):
    T=T[-GRAPH_WIDTH:]
    Y_MAX = Y_MAX[-GRAPH_WIDTH:]
    Y_AVE = Y_AVE[-GRAPH_WIDTH:]
    
  histgram_values, histgram_bins = np.histogram(np.array(target_depths).flatten(), bins=50)
  plot1.setData(histgram_bins[:-1], histgram_values)
  
  figure2.clear()
  fig2_ave.clear()
  figure2.plot(Y_MAX, pen="r")
  fig2_ave.addItem(pg.PlotCurveItem(Y_AVE, pen="y"))
  
  #表示用画像の処理
  view_image = depth_torso.copy().astype(np.float32)
  valid_min = np.where(view_image > 0, view_image, view_image.max()).min()
  view_image = np.where(view_image<valid_min, 0, ((view_image-valid_min)/(view_image.max()-valid_min)))
  view_image *= 255
  view_image = (np.stack([view_image, view_image, view_image], axis=-1)).astype(np.uint8)

  cv2.circle(view_image, cropped_right_shoulder.astype(np.uint16), 10, (255,0,0), 2)
  cv2.circle(view_image, cropped_left_shoulder.astype(np.uint16), 10, (255,0,0), 2)  
  
  
  
  #region 深度マップ、その他情報の表示
  color_img_view = cv2.cvtColor(cv2.resize(color_image, (COLOR_WIDTH,COLOR_HEIGHT)), cv2.COLOR_RGB2BGR)
  color_image = QtGui.QImage(color_img_view, COLOR_WIDTH, COLOR_HEIGHT, COLOR_WIDTH*3, QtGui.QImage.Format_RGB888)
  color_pixmap = QtGui.QPixmap.fromImage(color_image)
  colorImageBox.scene().clear()
  colorImageBox.scene().addPixmap(color_pixmap)
  
  depth_img_view = cv2.resize(view_image, (DEPTH_WIDTH, DEPTH_WIDTH))
  depth_image = QtGui.QImage(depth_img_view, DEPTH_WIDTH, DEPTH_WIDTH, DEPTH_WIDTH*3, QtGui.QImage.Format_RGB888)
  depth_pixmap = QtGui.QPixmap.fromImage(depth_image)
  procImageBox.scene().clear()
  procImageBox.scene().addPixmap(depth_pixmap)
  win.Info.setText(f"""aaa""")
  #endregion
  
if __name__=="__main__":
  qtTimer = QtCore.QTimer()
  qtTimer.timeout.connect(update)
  qtTimer.start(1000//FPS)
  win.show()
  start_time=time.time()
  pg.exec()