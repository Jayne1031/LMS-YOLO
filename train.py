from ultralytics import YOLO

# model = YOLO('ultralytics/cfg/models/11/yolo11_org.yaml')
# model = YOLO('/home1/zzh/ultralytics-zz/ultralytics/cfg/models/v8/yolov8_modify.yaml')#.load('yolov8n.pt')
# model.info()
# model = YOLO('/home1/zzh/ultralytics-zz/runs/detect/train51/weights/best.pt')
# result = model.val(data='/home1/zzh/ultralytics-zz/ultralytics/cfg/datasets/cityscapes.yaml')
# model.predict('/home1/zzh/dataset/8classes/images/test', visualize=True)
# print(result)
# model.train(device=[3,4], batch=32, data='/home1/zzh/ultralytics-zz/ultralytics/cfg/datasets/cityscapes.yaml', epochs=100)

# validation
model = YOLO('/home1/zzh/ultralytics-zz/runs/detect/train51/weights/best.pt')
result = model.val(data='/home1/zzh/ultralytics-zz/ultralytics/cfg/datasets/cityscapes.yaml',batch=1,device=0)