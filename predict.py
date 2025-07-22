from ultralytics import YOLO

model = YOLO("yolov11_custom.pt")

model.predict(source = "123.mp4",show=True, save=True, line_width = 2, save_crop = True, save_txt = True , show_labels = True, show_conf=True, classes =0)