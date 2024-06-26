import pykinect_azure as pykinect
import numpy as np
import time

FPS_CONFS = {5:pykinect.K4A_FRAMES_PER_SECOND_5, 15:pykinect.K4A_FRAMES_PER_SECOND_15, 30:pykinect.K4A_FRAMES_PER_SECOND_30}

def joints2dist(joints):
    data = {}
    for j in joints:
      data[j.get_name()] = j.numpy()
    return data
   
def mapCvt(org, level, width):
    min_value = level-width/2
    img = org - min_value
    img = img/width*255
    img = np.where(img < 0, 0, img)
    img = np.where(img > 255, 255, img)   
    return img.astype(np.uint8)

class Kinect:
  """
      joints: [neck, pelvis, left_shoulder, right_shoulder] の座標配列
      masked_depth: 身体以外深度を0とした深度マップ
      orientation: 胴体の向き
  """
  def __init__(self, device_config=None, fps=None):
    pykinect.initialize_libraries(track_body=True)

    if device_config is None:
      device_config = pykinect.default_configuration
      device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
      device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
      device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
    
    device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
    
    if fps is not None and fps in FPS_CONFS:
      device_config.camera_fps = FPS_CONFS[fps]

    self.device = pykinect.start_device(config=device_config)
    self.bodyTracker = pykinect.start_body_tracker()#model_type=pykinect.K4ABT_LITE_MODEL)
    self.capture = None
    self.body_num = 0
    
    self.neck = None
    self.pelvis = None
    self.masked_depth = None
    self.orientation = None
    self.color_skeleton = None
  
  @property
  def depth_img(self):
    return self.masked_depth
    
  @property
  def color_img(self):
    return self.colored_skeleton
  
  def joints2d(self):
    body2d = self.body_frame.get_body2d()
    joints2d = joints2dist(body2d.joints)
    return joints2d
      
  def update(self):
    while True:
      self.capture = self.device.update()
      self.body_frame = self.bodyTracker.update()
      
      self.update_body_closest()
      if self.body_num != 0:
        break
      time.sleep(0.1)

    _, segment_image = self.body_frame.get_body_index_map_image()
    _, depth_image = self.capture.get_depth_image()
    _, color_image = self.capture.get_color_image()
    self.colored_skeleton = self.body_frame.draw_bodies(color_image, pykinect.K4A_CALIBRATION_TYPE_COLOR)
    
    self.masked_depth = np.where(segment_image==0, depth_image, 0)
      
    self.neck, self.pelvis = self.joints["neck"], self.joints["pelvis"]
    self.left_shoulder, self.right_shoulder = self.joints["right shoulder"], self.joints["left shoulder"]
   
    #return masked_depth, (neck, pelvis, left_shoulder, right_shoulder), color_skeleton   
  
  def update_body_closest(self):
    self.body_num = self.body_frame.get_num_bodies()
    self.target_idx = None
    norm_max = 0
    for idx in range(self.body_num):
      body = self.body_frame.get_body(idx)
      joints = joints2dist(body.joints)
      neck = joints["neck"]
      pelvis = joints["pelvis"]
      vec_norm = np.linalg.norm(neck - pelvis)
      if vec_norm > norm_max:
        self.target_idx = idx
        self.joints = joints.copy()
      
  def get_depth_image(self):
    return 
  
if __name__=="__main__":
  kinect = Kinect()
  kinect.update()
  (neck, pelvis, left_shoulder, right_shoulder) = kinect.joints
  depth_img = kinect.depth_img
  color_image = kinect.color_img
  print(neck)