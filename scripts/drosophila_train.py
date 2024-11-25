from ultralytics import YOLO

# YOLOv8(torch)モデルロード
model = YOLO("yolov8x-pose.pt")

results = model.train(data='./yolo_pose_config.yaml', 
                      epochs=1000, batch=20,
                      device=[1, 2]) # 先ほど作成したデータセット内のyamlファイルまでのパスを指定
# Export the model
model.export(format="onnx")