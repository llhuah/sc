if __name__ == '__main__':

    from ultralytics import YOLO

    # Load a model
    model = YOLO('ultralytics/cfg/models/v8/yolov8-CGFPN.yaml') # build a new model from scratch
    model.load("yolov8n.pt")  # load a pretrained model (recommended for training)
    # Use the model
    model.train(data='data/PV.yaml', epochs=200, device=1)  # train the model
