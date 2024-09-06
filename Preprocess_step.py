import sys, time, os, tqdm, argparse, subprocess, warnings, cv2,  numpy
from audio_separator.separator import Separator
from shutil import rmtree

# from audio_separator.separator import Separator

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description = "TalkNet Demo or Columnbia ASD Evaluation")

parser.add_argument('--videoName',             type=str, default="001",   help='Demo video name')
parser.add_argument('--videoFolder',           type=str, default="data_to_process",  help='Path for inputs')
parser.add_argument('--targetFolder',           type=str, default="data_processed",  help='Path for outputs')
parser.add_argument('--pretrainModel',         type=str, default="pretrain_TalkSet.model",   help='Path for the pretrained TalkNet model')
parser.add_argument('--worker',             type=int, default=8,   help='number of worker for multiprocessing')

parser.add_argument('--nDataLoaderThread',     type=int,   default=16,   help='Number of workers')
parser.add_argument('--facedetScale',          type=float, default=0.25, help='Scale factor for face detection, the frames will be scale to 0.25 orig')
parser.add_argument('--minTrack',              type=int,   default=20,   help='Number of min frames for each shot')
parser.add_argument('--numFailedDet',          type=int,   default=10,   help='Number of missed detections allowed before tracking is stopped')
parser.add_argument('--minFaceSize',           type=int,   default=1,    help='Minimum face size in pixels')
parser.add_argument('--cropScale',             type=float, default=0.50, help='Scale bounding box')

parser.add_argument('--start',                 type=int, default=0,   help='The start time of the video')
parser.add_argument('--duration',              type=int, default=0,  help='The duration of the video, when set as 0, will extract the whole video')

parser.add_argument('--evalCol',               dest='evalCol', action='store_true', help='Evaluate on Columnbia dataset')
parser.add_argument('--colSavePath',           type=str, default="/data08/col",  help='Path for inputs, tmps and outputs')

args = parser.parse_args()
from multiprocessing import Pool




# Main function
def main():
	# This preprocesstion is modified based on this [repository](https://github.com/joonson/syncnet_python).
	# ```
	# .
	# ├── pyavi
	# │   ├── audio.wav (Audio from input video)
	# │   ├── video.avi (Copy of the input video)
	# │   ├── video_only.avi (Output video without audio)
	# │   └── video_out.avi  (Output video with audio)
	# ├── pycrop (The detected face videos and audios)
	# │   ├── 000000.avi
	# │   ├── 000000.wav
	# │   ├── 000001.avi
	# │   ├── 000001.wav
	# │   └── ...
	# ├── pyframes (All the video frames in this video)
	# │   ├── 000001.jpg
	# │   ├── 000002.jpg
	# │   └── ...	
	# └── pywork
	#     ├── faces.pckl (face detection result)
	#     ├── scene.pckl (scene detection result)
	#     ├── scores.pckl (ASD result)
	#     └── tracks.pckl (face tracking result)
	# ```
	print("# of worker is set to ", args.worker)
	print("# of dataloader thread is set to ," , args.nDataLoaderThread)
	video_folder = args.videoFolder

	audio_separator_model_file = "audio_separator_v1/Kim_Vocal_2.onnx"
	audio_separator_model_path = os.path.dirname(audio_separator_model_file)
	audio_separator_model_name = os.path.basename(audio_separator_model_file)

	for video_name in os.listdir(video_folder):
		video_path = os.path.join(video_folder, video_name)

		args.videoPath = video_path
		args.savePath = os.path.join(args.targetFolder, video_name)

		# Initialization
		args.pyaviPath = os.path.join(args.savePath, 'pyavi')
		args.pyframesPath = os.path.join(args.savePath, 'pyframes')
		args.pyworkPath = os.path.join(args.savePath, 'pywork')
		args.pycropPath = os.path.join(args.savePath, 'pycrop')
		args.pywholePath = os.path.join(args.savePath, 'pywhole')
		if os.path.exists(args.savePath):
			rmtree(args.savePath)
		os.makedirs(args.pyaviPath, exist_ok = True) # The path for the input video, input audio, output video
		os.makedirs(args.pyframesPath, exist_ok = True) # Save all the video frames
		os.makedirs(args.pyworkPath, exist_ok = True) # Save the results in this process by the pckl method
		os.makedirs(args.pycropPath, exist_ok = True) # Save the detected face clips (audio+video) in this process
		os.makedirs(args.pywholePath, exist_ok=True)  # Save the detected face clips (audio+video) in this process
		# Extract video
		args.videoFilePath = os.path.join(args.pyaviPath, 'video.avi')
		# If duration did not set, extract the whole video, otherwise extract the video from 'args.start' to 'args.start + args.duration'
		time_1 = time.time()
		print(f"Extracting video {video_name}")
		if args.duration == 0:
			command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -async 1 -r 25 %s -loglevel panic" % \
				(args.videoPath, args.nDataLoaderThread, args.videoFilePath))
		else:
			command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -ss %.3f -to %.3f -async 1 -r 25 %s -loglevel panic" % \
				(args.videoPath, args.nDataLoaderThread, args.start, args.start + args.duration, args.videoFilePath))
		subprocess.call(command, shell=True, stdout=None)
		sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the video and save in %s \r\n" %(args.videoFilePath))

		print("Extract video done, time cost %.3f s"%(time.time() - time_1))
		# cost ~= 0.1 * video length , 60 min video will take about 6 min.

		# Extract audio

		time_2 = time.time()

		args.audioFilePath = os.path.join(args.pyaviPath, 'audio.wav')
		command = ("ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" % \
			(args.videoFilePath, args.nDataLoaderThread, args.audioFilePath))
		subprocess.call(command, shell=True, stdout=None)
		sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the audio and save in %s \r\n" %(args.audioFilePath))
		print("Extract audio done, time cost %.3f s"%(time.time() - time_2))
		# Tiny time cost, can be ignored

		#seperate BGM from audio
		time_3 = time.time()
		seperated_audio_dir = args.pyaviPath
		sample_rate = 16000
		if not os.path.exists(seperated_audio_dir):
			os.makedirs(seperated_audio_dir)

		audio_separator = Separator(
			output_dir=seperated_audio_dir,
			output_single_stem="vocals",
			model_file_dir=audio_separator_model_path,
		)
		# print("model loading")
		audio_separator.load_model(audio_separator_model_name)
		assert audio_separator.model_instance is not None, "Fail to load audio separate model."

		outputs = audio_separator.separate(args.audioFilePath)
		vocal_audio_file = outputs[0]
		print("vocal file is : ", vocal_audio_file)

		print("Audio seperator , time cost %.3f s" % (time.time() - time_3))

		time_4 = time.time()
		# Extract the video frames
		command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -f image2 %s -loglevel panic" % \
			(args.videoFilePath, args.nDataLoaderThread, os.path.join(args.pyframesPath, '%06d.jpg')))
		subprocess.call(command, shell=True, stdout=None)
		sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the frames and save in %s \r\n" %(args.pyframesPath))
		print("Extract video frame done, time cost %.3f s"%(time.time() - time_4))



if __name__ == '__main__':
    main()
