from ultralytics import YOLO

from ultralytics import settings
## https://docs.ultralytics.com/ja/quickstart/#modifying-settings
## You may change the path
settings.update({"datasets_dir": "/cellpose/scripts"})

model = YOLO("./yolo11x-pose.pt")
#model = YOLO("./runs/pose/train20241129/weights/best.pt")

results = model.train(data='./yolo_pose_config_1204.yaml', 
                      epochs=100, batch=20,
                      device=[2, 3]) # 先ほど作成したデータセット内のyamlファイルまでのパスを指定
# Export the model
model.export(format="onnx")