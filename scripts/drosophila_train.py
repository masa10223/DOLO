from ultralytics import YOLO

#from ultralytics import settings
## https://docs.ultralytics.com/ja/quickstart/#modifying-settings
## You may change the path
#settings.update({"datasets_dir": "/cellpose/scripts"})

model = YOLO("./yolo11x-pose.pt")

results = model.train(data='./yolo_pose_config_1125.yaml', 
                      epochs=10, batch=20,
                      device=[1, 2]) # 先ほど作成したデータセット内のyamlファイルまでのパスを指定
# Export the model
model.export(format="onnx")