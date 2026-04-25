#open the spacemit k1 camera
#huangxiaoyuan

import cv2

gst_str = 'spacemitsrc location=/home/cam/csi3_camera_ov5647_auto.json close-dmabuf=1 ! video/x-raw,format=NV12,width=1920,height=1080 ! appsink '

cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)  # 打开默认的摄像头

# 设置目标宽度和高度
target_width = 600  # 你可以根据需要调整这个值
target_height = 360 # 你可以根据需要调整这个值

while True:
  ret, frame = cap.read()  # 读取视频帧
  if not ret:
      print("无法读取帧，退出。")
      break

  frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12)

  # 调整图像大小
  resized_frame = cv2.resize(frame, (target_width, target_height))
  # 向左旋转90度
  rotated_frame = cv2.rotate(resized_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

  cv2.imshow('Video', resized_frame)  # 显示调整大小后的视频帧

  if cv2.waitKey(1) & 0xFF == ord('q'):  # 按下 'q' 键退出循环
      break

cap.release()  # 释放摄像头
cv2.destroyAllWindows()  # 关闭所有窗口