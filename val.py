import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/bestresult\PV\pv-CGF+C2f+Wiou\weights/best.pt')
    model.val(data='data\ELDDS.yaml',
              split='val',
              project='runs/val',
              name='exp',
              batch=1,
              )