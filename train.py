import comet_ml
from ultralytics import YOLO
import argparse

comet_ml.init()

# Parameters List:
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="yolov8n.pt")
parser.add_argument('--data', type=str, default="./settings/shooter_M_S.yaml")
parser.add_argument('--project', type=str, default="ASTERS_UE5")
parser.add_argument('--name', type=str, default="yolov8n_Masked_S")
parser.add_argument('--device', type=int, default=0)
args = parser.parse_args()

# Load a model
model = YOLO(args.model)  # load a pretrained model (recommended for training)

# train the model
results = model.train(
	data=args.data,
	epochs=100,
	project=args.project,
	name=args.name,
	pretrained=True,
	device=args.device
)