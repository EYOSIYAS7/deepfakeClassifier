import os
import re
import time
import torch
import cv2
from kernel_utils import VideoReader, FaceExtractor, confident_strategy, predict_on_video

from classifier import DeepFakeClassifier

def model_fn():
	
	model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns") # default: CPU
	checkpoint = torch.load("weight/b7_ns_best (1).pth", map_location="cpu")
	state_dict = checkpoint.get("state_dict", checkpoint)
	model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=True)
	model.eval()
	del checkpoint
	#models.append(model.half())

	return model

def model_fn_load():
 
    model = torch.load("model/loaded_model.pth", map_location="cpu")
    model.eval()

    return model

model = model_fn_load()

def convert_result(pred, class_names=["Real", "Fake"]):
	preds = [pred, 1 - pred]
	assert len(class_names) == len(preds), "Class / Prediction should have the same length"
	return {n: float(p) for n, p in zip(class_names, preds)}

def predict_fn(video):
	start = time.time()
	prediction = predict_on_video(face_extractor=meta["face_extractor"],
							   video_path=video,
							   batch_size=meta["fps"],
							   input_size=meta["input_size"],
							   models=model,
							   strategy=meta["strategy"],
							   apply_compression=False,
							   device='cpu')

	elapsed_time = round(time.time() - start, 2)

	prediction = convert_result(prediction)

	return prediction, elapsed_time


model_dir = 'weights'
frames_per_video = 32
video_reader = VideoReader()
video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
face_extractor = FaceExtractor(video_read_fn)
input_size = 380
strategy = confident_strategy
class_names = ["Real", "Fake"]

meta = {"fps": 32,
		"face_extractor": face_extractor,
		"input_size": input_size,
		"strategy": strategy}



prediction , time = predict_fn("fake-1.mp4")

print( "prediction -> ", prediction)
print( "time  -> ", time )















# import json
# from gradio_client import Client
# import os
# client = Client("https://thecho7-deepfake.hf.space/")
# result = client.predict(
# 				"yt1s.com - You Wont Believe What Obama Says In This Video .mp4",	# str representing input in 'video' Video component
# 				api_name="/predict"
# )


# with open(result[0], 'r') as file:
#     data = json.load(file)
# print(client)
# print("prediction -> ",data['label'])
# print("confidence -> ",data['confidences'][0]["confidence"])

# print("confidence -> ",data['confidences'][1]["confidence"] ,"is ", data['confidences'][1]["label"])