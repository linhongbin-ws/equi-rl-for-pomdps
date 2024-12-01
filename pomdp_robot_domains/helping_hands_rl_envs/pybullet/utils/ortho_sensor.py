import pybullet as pb
import numpy as np
from helping_hands_rl_envs.pybullet.utils.sensor import Sensor

class OrthographicSensor(Sensor):
  def __init__(self, cam_pos, cam_up_vector, target_pos, target_size, near, far):
    super().__init__(cam_pos, cam_up_vector, target_pos, target_size, near, far)
    self.target_size = target_size
    self.near = near
    self.far = far

  def setCamMatrix(self, cam_pos, cam_up_vector, target_pos):
    self.cam_z_min = 0.5
    cam_height = cam_pos[2] +self.cam_z_min
    
    self.view_matrix = pb.computeViewMatrix(
      cameraEyePosition=[cam_pos[0], cam_pos[1], cam_height],
      cameraUpVector=cam_up_vector,
      cameraTargetPosition=target_pos,
    )
    self.near = self.cam_z_min    
    self.proj_matrix = pb.computeProjectionMatrixFOV(70, 1, self.near, self.far)

  # def getDepth(self, size):
  #   image_arr = pb.getCameraImage(width=size, height=size,
  #                                 viewMatrix=self.view_matrix,
  #                                 projectionMatrix=self.proj_matrix)
  #   depth = np.array(image_arr[3])
  #   depth = self.far * self.near / (self.far - (self.far - self.near) * depth)

  #   return depth
