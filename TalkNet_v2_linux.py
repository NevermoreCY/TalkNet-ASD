import sys, time, os, tqdm, torch, argparse, glob, subprocess, warnings, cv2, pickle, numpy, pdb, math, python_speech_features

from scipy import signal
from shutil import rmtree
from scipy.io import wavfile
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, f1_score

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from model.faceDetector.s3fd import S3FD
from talkNet import talkNet
import numpy as np
import json

# from audio_separator.separator import Separator
import multiprocessing
from multiprocessing import Pool
multiprocessing.set_start_method('spawn', force=True)

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

parser.add_argument('--data_parallel_num',           type=int, default=0,  help='Path for inputs, tmps and outputs')
parser.add_argument('--job_num',           type=int, default=0,  help='Path for inputs, tmps and outputs')

args = parser.parse_args()


if os.path.isfile(args.pretrainModel) == False: # Download the pretrained model
    Link = "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea"
    cmd = "gdown --id %s -O %s"%(Link, args.pretrainModel)
    subprocess.call(cmd, shell=True, stdout=None)

if args.evalCol == True:
	# The process is: 1. download video and labels(I have modified the format of labels to make it easiler for using)
	# 	              2. extract audio, extract video frames
	#                 3. scend detection, face detection and face tracking
	#                 4. active speaker detection for the detected face clips
	#                 5. use iou to find the identity of each face clips, compute the F1 results
	# The step 1 to 3 will take some time (That is one-time process). It depends on your cpu and gpu speed. For reference, I used 1.5 hour
	# The step 4 and 5 need less than 10 minutes
	# Need about 20G space finally
	# ```
	args.videoName = 'col'
	args.videoFolder = args.colSavePath
	args.savePath = os.path.join(args.videoFolder, args.videoName)
	args.videoPath = os.path.join(args.videoFolder, args.videoName + '.mp4')
	args.duration = 0
	if os.path.isfile(args.videoPath) == False:  # Download video
		link = 'https://www.youtube.com/watch?v=6GzxbrO0DHM&t=2s'
		cmd = "youtube-dl -f best -o %s '%s'"%(args.videoPath, link)
		output = subprocess.call(cmd, shell=True, stdout=None)
	if os.path.isdir(args.videoFolder + '/col_labels') == False: # Download label
		link = "1Tto5JBt6NsEOLFRWzyZEeV6kCCddc6wv"
		cmd = "gdown --id %s -O %s"%(link, args.videoFolder + '/col_labels.tar.gz')
		subprocess.call(cmd, shell=True, stdout=None)
		cmd = "tar -xzvf %s -C %s"%(args.videoFolder + '/col_labels.tar.gz', args.videoFolder)
		subprocess.call(cmd, shell=True, stdout=None)
		os.remove(args.videoFolder + '/col_labels.tar.gz')	
# else:
# 	args.videoPath = glob.glob(os.path.join(args.videoFolder, args.videoName + '.*'))[0]
# 	args.savePath = os.path.join(args.videoFolder, args.videoName)

def scene_detect(args):
	# CPU: Scene detection, output is the list of each shot's time duration
	videoManager = VideoManager([args.videoFilePath])
	statsManager = StatsManager()
	sceneManager = SceneManager(statsManager)
	sceneManager.add_detector(ContentDetector())
	baseTimecode = videoManager.get_base_timecode()
	videoManager.set_downscale_factor()
	videoManager.start()
	sceneManager.detect_scenes(frame_source = videoManager)
	sceneList = sceneManager.get_scene_list(baseTimecode)
	savePath = os.path.join(args.pyworkPath, 'scene.pckl')
	if sceneList == []:
		sceneList = [(videoManager.get_base_timecode(),videoManager.get_current_timecode())]
	with open(savePath, 'wb') as fil:
		# print("\n\n save scenelist \n\n")
		# print(sceneList)
		pickle.dump(sceneList, fil)
		sys.stderr.write('%s - scenes detected %d\n'%(args.videoFilePath, len(sceneList)))
	return sceneList

# def inference_video(args):
# 	# GPU: Face detection, output is the list contains the face location and score in this frame
# 	DET = S3FD(device='cuda')
# 	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
# 	flist.sort()
# 	dets = []
# 	for fidx, fname in enumerate(flist):
# 		image = cv2.imread(fname)
# 		imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# 		bboxes = DET.detect_faces(imageNumpy, conf_th=0.9, scales=[args.facedetScale])
# 		dets.append([])
# 		for bbox in bboxes:
# 		  dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]}) # dets has the frames info, bbox info, conf info
# 		sys.stderr.write('%s-%05d; %d dets\r' % (args.videoFilePath, fidx, len(dets[-1])))
# 	savePath = os.path.join(args.pyworkPath,'faces.pckl')
# 	with open(savePath, 'wb') as fil:
# 		# print("\n\n ** saving dets for face detection **\n")
# 		# print(dets)
# 		pickle.dump(dets, fil)
# 	return dets



def inference_video_worker(args,frame_list,frame_offset):
	# GPU: Face detection, output is the list contains the face location and score in this frame
	DET = S3FD(device='cuda')
	dets = []
	for fidx, fname in enumerate(frame_list):
		image = cv2.imread(fname)
		imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		bboxes = DET.detect_faces(imageNumpy, conf_th=0.9, scales=[args.facedetScale])
		dets.append([])
		for bbox in bboxes:

			dets[-1].append({'frame':fidx + frame_offset , 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]  } ) # dets has the frames info, bbox info, conf info
		# sys.stderr.write('%s-%05d; %d dets\r' % (args.videoFilePath, fidx+ frame_offset, len(dets[-1])))
	return dets

def inference_video(args, n_worker=8):
	# GPU: Face detection, output is the list contains the face location and score in this frame
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
	flist.sort()
	args.total_frames += len(flist)
	chunk_size = len(flist) // n_worker
	chunks =  [flist[i:i + chunk_size] for i in range(0, len(flist), chunk_size)]
	offsets = [sum([len(c) for c in chunks[:i]]) for i in range(len(chunks))]
	print(f"length of chunks flist is {len(flist)} , chunk_size is {chunk_size} , chunks[0] are {len(chunks[0])} , offsets is {offsets}")

	with Pool(processes=n_worker) as pool:
		results = pool.starmap(inference_video_worker, [(args, chunk, offset) for chunk, offset in zip(chunks, offsets)])

	final_results = []
	for result in results:
		final_results.extend(result)

	savePath = os.path.join(args.pyworkPath,'faces.pckl')
	with open(savePath, 'wb') as fil:
		# print("\n\n ** saving dets for face detection **\n")
		# print(dets)
		pickle.dump(final_results, fil)

	return final_results



def bb_intersection_over_union(boxA, boxB, evalCol = False):
	# CPU: IOU Function to calculate overlap between two image
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA) * max(0, yB - yA)
	boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
	if evalCol == True:
		iou = interArea / float(boxAArea)
	else:
		iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou


def frame_intersection(track1, track2):
	f1 = track1["frame"]
	f2 = track2["frame"]
	intersection = np.intersect1d(f1, f2)
	if intersection.size > 0:
		return True
	else:
		return False



def track_shot_work(args, sceneFaces):
    # CPU: Face tracking
    iouThres = 0.5  # Minimum IOU between consecutive face detections
    tracks = []
    while True:
        track = []
        for frameFaces in sceneFaces:
            for face in frameFaces:
                if track == []:
                    track.append(face)
                    frameFaces.remove(face)
                elif face['frame'] - track[-1]['frame'] <= args.numFailedDet:
                    iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                    if iou > iouThres:
                        track.append(face)
                        frameFaces.remove(face)
                        continue
                else:
                    break
        if track == []:
            break
        elif len(track) > args.minTrack:
            frameNum = np.array([f['frame'] for f in track])
            bboxes = np.array([np.array(f['bbox']) for f in track])
            multi_face = [f['frame'] for f in track]
            multifaceI = False
            if any(multi_face):
                multifaceI = True

            frameI = np.arange(frameNum[0], frameNum[-1] + 1)
            bboxesI = []
            for ij in range(0, 4):
                interpfn = interp1d(frameNum, bboxes[:, ij])
                bboxesI.append(interpfn(frameI))
            bboxesI = np.stack(bboxesI, axis=1)
            if max(np.mean(bboxesI[:, 2] - bboxesI[:, 0]), np.mean(bboxesI[:, 3] - bboxesI[:, 1])) > args.minFaceSize:
                tracks.append({'frame': frameI, 'bbox': bboxesI, 'multiface': multifaceI})

    return tracks

def process_shots(args, scenes, faces, n_workers=8):
    allTracks = []

    # 将场景拆分为任务
    tasks = [
        (args, faces[shot[0].frame_num:shot[1].frame_num])
        for shot in scenes
        if shot[1].frame_num - shot[0].frame_num >= args.minTrack
    ]
    # 使用多进程处理每个场景
    with Pool(processes=n_workers) as pool:
        results = pool.starmap(track_shot_work, tasks)
    print("len results", len(results) )
    for result in results:
        allTracks.extend(result)

    return allTracks

def track_shot(args, sceneFaces, n_worker=6):
	# CPU: Face tracking
	iouThres  = 0.5     # Minimum IOU between consecutive face detections
	tracks    = []
	while True:
		track     = []
		for frameFaces in sceneFaces:
			for face in frameFaces:
				if track == []:
					track.append(face)
					frameFaces.remove(face)
				elif face['frame'] - track[-1]['frame'] <= args.numFailedDet:
					iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
					if iou > iouThres:
						track.append(face)
						frameFaces.remove(face)
						continue
				else:
					break
		if track == []:
			break
		elif len(track) > args.minTrack:
			frameNum    = numpy.array([ f['frame'] for f in track ])
			bboxes      = numpy.array([numpy.array(f['bbox']) for f in track])
			multi_face = [f['frame'] for f in track]
			multifaceI = False
			if any(multi_face):
				multifaceI = True

			frameI      = numpy.arange(frameNum[0],frameNum[-1]+1)
			bboxesI    = []
			for ij in range(0,4):
				interpfn  = interp1d(frameNum, bboxes[:,ij])
				bboxesI.append(interpfn(frameI))
			bboxesI  = numpy.stack(bboxesI, axis=1)
			if max(numpy.mean(bboxesI[:,2]-bboxesI[:,0]), numpy.mean(bboxesI[:,3]-bboxesI[:,1])) > args.minFaceSize:
				tracks.append({'frame':frameI,'bbox':bboxesI , 'multiface':multifaceI})

	# check if the video for track encoutered multiple faces (discarded)
	for i in range(len(tracks)):
		intersec = False
		for j in range(len(tracks)):
			if i != j:
				if (frame_intersection(tracks[i],tracks[j])):
					intersec = True
		if intersec:
			tracks[i]['multiface'] = True
		else:
			tracks[i]['multiface'] = False

	return tracks

# def crop_video(args, track, cropFile):
# 	# CPU: crop the face clips
# 	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg')) # Read the frames
# 	flist.sort()
#
# 	dets = {'x':[], 'y':[], 's':[]}
# 	for det in track['bbox']: # Read the tracks
# 		dets['s'].append(  max((det[3]-det[1]), (det[2]-det[0]))  /2)
# 		dets['y'].append((det[1]+det[3])/2) # crop center x
# 		dets['x'].append((det[0]+det[2])/2) # crop center y
# 	dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections
# 	dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
# 	dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
# 	print("dets length is : ", len(dets['x']), dets['x'] )
#
# 	example_frame = track['frame'][0]
# 	image = cv2.imread(flist[example_frame])
# 	image_shape = image.shape
# 	print("image shape is ", image.shape)
# 	vOut = cv2.VideoWriter(cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (512,512))  # Write video
#
# 	for fidx, frame in enumerate(track['frame']):
# 		cs  = args.cropScale
# 		bs  = dets['s'][fidx]   # Detection box size
# 		bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount
# 		image = cv2.imread(flist[frame])
# 		frame = numpy.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
# 		#because of padding, we need to modify the center
# 		my  = dets['y'][fidx] + bsi  # BBox center Y
# 		mx  = dets['x'][fidx] + bsi  # BBox center X
# 		face = frame[int(my-bs*(1+cs)):int(my+bs*(1+cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
# 		vOut.write(cv2.resize(face, (512, 512)))
# 	audioTmp    = cropFile + '.wav'
# 	audioStart  = (track['frame'][0]) / 25
# 	audioEnd    = (track['frame'][-1]+1) / 25
# 	vOut.release()
# 	command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" % \
# 		      (args.audioFilePath, args.nDataLoaderThread, audioStart, audioEnd, audioTmp))
# 	output = subprocess.call(command, shell=True, stdout=None) # Crop audio file
# 	_, audio = wavfile.read(audioTmp)
# 	command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" % \
# 			  (cropFile, audioTmp, args.nDataLoaderThread, cropFile)) # Combine audio and video file
# 	output = subprocess.call(command, shell=True, stdout=None)
# 	os.remove(cropFile + 't.avi')
# 	return {'track':track, 'proc_track':dets}


def process_tracks(args, allTracks, n_workers=6):
    vidTracks = []

    # 创建任务列表
    tasks = [
        (args, track, os.path.join(args.pycropPath, '%05d' % ii),os.path.join(args.pywholePath, '%05d'%ii  ) )
        for ii, track in enumerate(allTracks)
    ]

    # 使用多进程处理每个任务
    with Pool(processes=n_workers) as pool:
        results = pool.starmap(crop_video_whole, tasks)

    # 收集所有进程的结果
    for result in results:
        vidTracks.append(result)

    return vidTracks

def crop_video(args, track, cropFile):
	# CPU: crop the face clips
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg')) # Read the frames
	flist.sort()
	vOut = cv2.VideoWriter(cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (224,224))# Write video
	face_shapes = []
	dets = {'x':[], 'y':[], 's':[] , 'w':[], 'h':[]}
	for det in track['bbox']: # Read the tracks
		w = det[2] - det[0]
		h = det[3] - det[1]
		# print("wid")
		dets['s'].append(max(w, h) /2)
		dets['w'].append(w)
		dets['h'].append(h)
		dets['y'].append((det[1]+det[3])/2) # crop center x
		dets['x'].append((det[0]+det[2])/2) # crop center y
	dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections
	dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
	dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
	for fidx, frame in enumerate(track['frame']):
		cs  = args.cropScale
		bs  = dets['s'][fidx]   # Detection box size
		bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount
		image = cv2.imread(flist[frame])
		frame = numpy.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
		my  = dets['y'][fidx] + bsi  # BBox center Y
		mx  = dets['x'][fidx] + bsi  # BBox center X
		face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
		vOut.write(cv2.resize(face, (224, 224)))
	audioTmp    = cropFile + '.wav'
	audioStart  = (track['frame'][0]) / 25
	audioEnd    = (track['frame'][-1]+1) / 25
	vOut.release()
	command = ("ffmpeg -y -i '%s' -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" % \
		      (args.audioFilePath, args.nDataLoaderThread, audioStart, audioEnd, audioTmp))
	output = subprocess.call(command, shell=True, stdout=None) # Crop audio file
	_, audio = wavfile.read(audioTmp)
	command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" % \
			  (cropFile, audioTmp, args.nDataLoaderThread, cropFile)) # Combine audio and video file
	output = subprocess.call(command, shell=True, stdout=None)
	os.remove(cropFile + 't.avi')
	return {'track':track, 'proc_track':dets }


def crop_video_whole(args, track, cropFile,cropFile_whole):
	# CPU: crop the face clips
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg')) # Read the frames
	flist.sort()

	dets = {'x': [], 'y': [], 's': [], 'w': [], 'h': []}
	for det in track['bbox']:  # Read the tracks
		w = det[2] - det[0]
		h = det[3] - det[1]
		# print("height and widths are ", h,w)
		dets['h'].append(h)
		dets['w'].append(w)
		dets['s'].append(max(h, w)/2)
		dets['y'].append((det[1]+det[3])/2) # crop center x
		dets['x'].append((det[0]+det[2])/2) # crop center y
	dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections
	dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
	dets['y'] = signal.medfilt(dets['y'], kernel_size=13)

	example_frame = track['frame'][0]
	image = cv2.imread(flist[example_frame])
	image_shape = image.shape

	vOut = cv2.VideoWriter(cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (224, 224))  # Write video
	vOut_whole = cv2.VideoWriter(cropFile_whole + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (image_shape[1], image_shape[0] ))  # Write video

	face_shapes = []

	for fidx, frame in enumerate(track['frame']):
		cs  = args.cropScale
		bs  = dets['s'][fidx]   # Detection box size
		bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount
		image = cv2.imread(flist[frame])
		frame = numpy.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
		my  = dets['y'][fidx] + bsi  # BBox center Y
		mx  = dets['x'][fidx] + bsi  # BBox center X
		face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
		# print("face shape is ", face.shape)
		face_shapes.append(face.shape)
		vOut.write(cv2.resize(face, (224, 224)))
		vOut_whole.write(image)
	audioTmp    = cropFile + '.wav'

	audioStart  = (track['frame'][0]) / 25
	audioEnd    = (track['frame'][-1]+1) / 25
	vOut.release()
	vOut_whole.release()
	command = ("ffmpeg -y -i '%s' -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" % \
		      (args.audioFilePath, 2, audioStart, audioEnd, audioTmp))
	output = subprocess.call(command, shell=True, stdout=None) # Crop audio file
	print(command)
	print(output)
	_, audio = wavfile.read(audioTmp)
	command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" % \
			  (cropFile, audioTmp, 2, cropFile)) # Combine audio and video file
	output = subprocess.call(command, shell=True, stdout=None)
	os.remove(cropFile + 't.avi')
	print(command)
	print(output)
	command_whole = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" % \
			   (cropFile_whole , audioTmp, 2, cropFile_whole ))  # Combine audio and video file
	output = subprocess.call(command_whole, shell=True, stdout=None)
	os.remove(cropFile_whole  + 't.avi')
	print(command)
	print(output)
	track["face_shapes"] = face_shapes

	return {'track':track, 'proc_track':dets}

def crop_whole_video(args, track, cropFile):
	# CPU: crop the face clips
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg')) # Read the frames
	flist.sort()

	dets = {'x':[], 'y':[], 's':[]}
	for det in track['bbox']: # Read the tracks
		dets['s'].append(  max((det[3]-det[1]), (det[2]-det[0]))  /2)
		dets['y'].append((det[1]+det[3])/2) # crop center x
		dets['x'].append((det[0]+det[2])/2) # crop center y
	dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections
	dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
	dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
	# print("dets length is : ", len(dets['x']), dets['x'] )

	example_frame = track['frame'][0]
	image = cv2.imread(flist[example_frame])
	image_shape = image.shape
	# print("image shape is ", image.shape)
	vOut = cv2.VideoWriter(cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (image_shape[1], image_shape[0] ))  # Write video

	for fidx, frame in enumerate(track['frame']):
		image = cv2.imread(flist[frame])
		vOut.write(image)
	audioTmp    = cropFile + '.wav'
	audioStart  = (track['frame'][0]) / 25
	audioEnd    = (track['frame'][-1]+1) / 25
	vOut.release()
	command = ("ffmpeg -y -i '%s' -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" % \
		      (args.audioFilePath, args.nDataLoaderThread, audioStart, audioEnd, audioTmp))
	output = subprocess.call(command, shell=True, stdout=None) # Crop audio file
	_, audio = wavfile.read(audioTmp)
	command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" % \
			  (cropFile, audioTmp, args.nDataLoaderThread, cropFile)) # Combine audio and video file
	output = subprocess.call(command, shell=True, stdout=None)
	os.remove(cropFile + 't.avi')
	return {'track':track, 'proc_track':dets}

def extract_MFCC(file, outPath):
	# CPU: extract mfcc
	sr, audio = wavfile.read(file)
	mfcc = python_speech_features.mfcc(audio,sr) # (N_frames, 13)   [1s = 100 frames]
	featuresPath = os.path.join(outPath, file.split('/')[-1].replace('.wav', '.npy'))
	numpy.save(featuresPath, mfcc)

def evaluate_network(files, args):
	# GPU: active speaker detection by pretrained TalkNet
	s = talkNet()
	s.loadParameters(args.pretrainModel)
	sys.stderr.write("Model %s loaded from previous state! \r\n"%args.pretrainModel)
	s.eval()
	allScores = []
	# durationSet = {1,2,4,6} # To make the result more reliable
	durationSet = {1,1,1,2,2,2,3,3,4,5,6} # Use this line can get more reliable result
	for file in tqdm.tqdm(files, total = len(files)):
		# fileName = os.path.splitext(file.split('\\')[-1])[0] # Load audio and
		directory, filename = os.path.split(file)
		# print("\n\n***\n\n", directory,filename)
		# fileName = os.path.splitext(file.split('/')[-1])[0]

		fileName = os.path.splitext(filename)[0]
		# print("fileName = ", fileName)
		_, audio = wavfile.read(os.path.join(args.pycropPath, fileName + '.wav'))
		audioFeature = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010)
		video = cv2.VideoCapture(os.path.join(args.pycropPath, fileName + '.avi'))
		videoFeature = []
		while video.isOpened():
			ret, frames = video.read()
			if ret == True:
				face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
				face = cv2.resize(face, (224,224))
				face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
				videoFeature.append(face)
			else:
				break
		video.release()
		videoFeature = numpy.array(videoFeature)
		length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0] / 25)
		audioFeature = audioFeature[:int(round(length * 100)),:]
		videoFeature = videoFeature[:int(round(length * 25)),:,:]
		allScore = [] # Evaluation use TalkNet
		for duration in durationSet:
			batchSize = int(math.ceil(length / duration))
			scores = []
			with torch.no_grad():
				for i in range(batchSize):
					inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i+1) * duration * 100,:]).unsqueeze(0).cuda()
					inputV = torch.FloatTensor(videoFeature[i * duration * 25: (i+1) * duration * 25,:,:]).unsqueeze(0).cuda()
					embedA = s.model.forward_audio_frontend(inputA)
					embedV = s.model.forward_visual_frontend(inputV)	
					embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
					out = s.model.forward_audio_visual_backend(embedA, embedV)
					score = s.lossAV.forward(out, labels = None)
					scores.extend(score)
			allScore.append(scores)
		allScore = numpy.round((numpy.mean(numpy.array(allScore), axis = 0)), 1).astype(float)
		allScores.append(allScore)	
	return allScores

def visualization(tracks, scores, args):
	# CPU: visulize the result for video format
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
	flist.sort()
	faces = [[] for i in range(len(flist))]
	for tidx, track in enumerate(tracks):
		score = scores[tidx]
		for fidx, frame in enumerate(track['track']['frame'].tolist()):
			s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)] # average smoothing
			s = numpy.mean(s)
			faces[frame].append({'track':tidx, 'score':float(s),'s':track['proc_track']['s'][fidx], 'x':track['proc_track']['x'][fidx], 'y':track['proc_track']['y'][fidx]})
	firstImage = cv2.imread(flist[0])
	fw = firstImage.shape[1]
	fh = firstImage.shape[0]
	vOut = cv2.VideoWriter(os.path.join(args.pyaviPath, 'video_only.avi'), cv2.VideoWriter_fourcc(*'XVID'), 25, (fw,fh))
	colorDict = {0: 0, 1: 255}
	for fidx, fname in tqdm.tqdm(enumerate(flist), total = len(flist)):
		image = cv2.imread(fname)
		for face in faces[fidx]:
			clr = colorDict[int((face['score'] >= 0))]
			txt = round(face['score'], 1)
			cv2.rectangle(image, (int(face['x']-face['s']), int(face['y']-face['s'])), (int(face['x']+face['s']), int(face['y']+face['s'])),(0,clr,255-clr),10)
			cv2.putText(image,'%s'%(txt), (int(face['x']-face['s']), int(face['y']-face['s'])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,clr,255-clr),5)
		vOut.write(image)
	vOut.release()
	command = ("ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" % \
		(os.path.join(args.pyaviPath, 'video_only.avi'), args.audioFilePath, \
		args.nDataLoaderThread, os.path.join(args.pyaviPath,'video_out.avi'))) 
	output = subprocess.call(command, shell=True, stdout=None)


def split_into_chunks(score_list, score_thresh=0, min_bad_frames = 25, min_good_frames = 20, debug=False):
	if debug:
		print("score list len : ", len(score_list))
	print("score_thresh" , score_thresh)
	chunks = []


	cur_unmatched_frames = 0
	cur_good = True
	start_f = 0
	end_f = 0
	for frame_id in range(len(score_list)):
		if cur_good:
			#if score_list[frame_id] >= score_thresh:
			#	print("good")
			if score_list[frame_id] < score_thresh:
				end_f = frame_id
				chunks.append([start_f,end_f,cur_good])
				cur_good = False
				start_f = frame_id

		elif not cur_good:
			if score_list[frame_id] >= score_thresh:
				end_f = frame_id
				chunks.append([start_f, end_f, cur_good])
				cur_good = True
				start_f = frame_id
	if start_f < frame_id:
		chunks.append([start_f,frame_id,cur_good])
	if debug:
		print("\n\n DEBUG chunks are \n\n ", chunks)

	result = []
	for chunk in chunks:
		if debug:
			print("\n chunk:",chunk, "\n Result:" ,result)
		if result == []:
			if chunk[2] and (chunk[1]- chunk[0])>0:
				result.append(chunk)
				# print(1)
		else: # there is item in result
			if not chunk[2]: # bad
				cur_chunk_size = chunk[1] - chunk[0]
				previous_chunk_size = result[-1][1] - result[-1][0]
				if debug:
					print("cur_chunk_size" , cur_chunk_size, "previous_chunk_size",  previous_chunk_size)
				if (chunk[1] - chunk[0]) < min_bad_frames and (chunk[0] - result[-1][1]) <= 3:
					# extend previous chunk
					# print("extend previous chunk", result[-1])
					result[-1][1] = chunk[1]
					# print("extended ", result[-1])
					# print(2)
				# print(2.5)

			elif chunk[2]: # good
				if (chunk[0] - result[-1][1]) <= 2:
					result[-1][1] = chunk[1]
					# print(3)
				else:
					result.append(chunk)
					# print(4)
	print("\n After connect, good chunks are : ", result)

	good_chunks = []
	for chunk in result:
		chunk_size = chunk[1] - chunk[0]
		if chunk_size > min_good_frames:
			good_chunks.append(chunk)
	print("\n Godd chunks are : ", good_chunks)

	return good_chunks




def filter_with_score(tracks, scores, args, score_threshold=0):


	files = glob.glob("%s/*.avi" % args.pywholePath)
	# print("\n\n *** \n\n ", files, args.pyworkPath)
	files.sort()

	data = []


	for tidx, track in enumerate(tracks):
		score = scores[tidx]
		# print("tidx ", tidx )
		# print("score", score)
		multiface = track['track']['multiface']
		file_path = files[tidx]

		print("\n\n*** DEBUGING *** \n\n\n current file is :", file_path)
		# print("\n track keys ", track.keys(), track['track'].keys() )
		# print(track['track'])
		# print(track['track']['frame'])

		# start_frame =
		# print("score:",score)


		good_chunks = split_into_chunks(score,score_thresh=score_threshold, min_bad_frames = 25, min_good_frames = 20, debug=False)


		if good_chunks:
			chunk_num = len(good_chunks)
			start_frame = track['track']['frame'][0]
			flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))  # Read the frames
			flist.sort()

			example_frame = track['track']['frame'][0]
			image = cv2.imread(flist[example_frame])
			image_shape = image.shape
			# print("Sample image shape is ," ,image_shape)
			chunk_id = 0
			for chunk in good_chunks:
				# we generate a video for each chunk
				chunk_start_frame = chunk[0]
				if chunk_start_frame <0 :
					chunk_start_frame = 0
				chunk_end_frame = chunk[1]+1
				chunk_file_path = file_path[:-4] +'_'+ ('%03d' % chunk_id)
				print("chunk_file_path is ", chunk_file_path)
				# print("chunk start frame :",chunk_start_frame,chunk_end_frame)
				vOut_whole = cv2.VideoWriter(chunk_file_path + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), 25,
											 (image_shape[1], image_shape[0]))

				# print("frames are :", track['track']['frame'][chunk_start_frame:chunk_end_frame])
				# gather the frames:
				for fidx, frame in enumerate(track['track']['frame'][chunk_start_frame:chunk_end_frame]):
					image = cv2.imread(flist[frame])
					vOut_whole.write(image)

				audioTmp = chunk_file_path + '.wav'
				audioStart = (track['track']['frame'][chunk_start_frame]) / 25
				audioEnd = (track['track']['frame'][chunk_end_frame] +1 ) / 25
				# print("audio start", audioStart, "audio end", audioEnd)
				vOut_whole.release()

				# Crop audio file
				command = ("ffmpeg -y -i '%s' -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" % \
							(args.audioFilePath, 2, audioStart, audioEnd, audioTmp))
				output = subprocess.call(command, shell=True, stdout=None)

				# Combine audio and video file
				command_whole = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" % \
								 (chunk_file_path, audioTmp, 2, chunk_file_path))
				output = subprocess.call(command_whole, shell=True, stdout=None)
				os.remove(chunk_file_path + 't.avi')

				# For each chunk, we store a meta-data
				h_mean = np.mean(track['proc_track']['h'][chunk[0]:chunk[1]])
				w_mean = np.mean(track['proc_track']['w'][chunk[0]:chunk[1]])
				args.h_mean.append(h_mean)
				args.w_mean.append(w_mean)
				num_frames = chunk[1] - chunk[0]
				avg_score = np.mean(score[chunk[0]:chunk[1]])
				# filtered_data.json format:
				meta_data_to_save = [chunk_file_path+".avi", num_frames, avg_score, h_mean, w_mean , int(chunk_start_frame), int(chunk_end_frame)]

				if multiface:
					meta_data_to_save.append(False)
				else:
					# we only do further face size check for singleface data! Don't care multiface for now
					meta_data_to_save.append(True)
				# Just for record
				if (h_mean + w_mean) / 2 >= 200:
					args.good_frames_200 += len(scores[tidx])
				if (h_mean + w_mean) / 2 >= 224:
					args.good_frames_224 += len(scores[tidx])
				if (h_mean + w_mean) / 2 >= 256:
					args.good_frames_256 += len(scores[tidx])
				if (h_mean + w_mean) / 2 >= 300:
					args.good_frames_300 += len(scores[tidx])
				data.append(meta_data_to_save)
				chunk_id+=1

	data = {'meta_data': data}

	out_path = os.path.join(args.savePath, "filtered_data.json" )
	with open(out_path,'w',encoding='utf-8') as f:
		json.dump(data,f)

	print("MetaData is saved at %s" % out_path)



def evaluate_col_ASD(tracks, scores, args):
	txtPath = args.videoFolder + '/col_labels/fusion/*.txt' # Load labels
	predictionSet = {}
	for name in {'long', 'bell', 'boll', 'lieb', 'sick', 'abbas'}:
		predictionSet[name] = [[],[]]
	dictGT = {}
	txtFiles = glob.glob("%s"%txtPath)
	for file in txtFiles:
		lines = open(file).read().splitlines()
		idName = file.split('/')[-1][:-4]
		for line in lines:
			data = line.split('\t')
			frame = int(int(data[0]) / 29.97 * 25)
			x1 = int(data[1])
			y1 = int(data[2])
			x2 = int(data[1]) + int(data[3])
			y2 = int(data[2]) + int(data[3])
			gt = int(data[4])
			if frame in dictGT:
				dictGT[frame].append([x1,y1,x2,y2,gt,idName])
			else:
				dictGT[frame] = [[x1,y1,x2,y2,gt,idName]]	
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg')) # Load files
	flist.sort()
	faces = [[] for i in range(len(flist))]
	for tidx, track in enumerate(tracks):
		score = scores[tidx]				
		for fidx, frame in enumerate(track['track']['frame'].tolist()):
			s = numpy.mean(score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)]) # average smoothing
			faces[frame].append({'track':tidx, 'score':float(s),'s':track['proc_track']['s'][fidx], 'x':track['proc_track']['x'][fidx], 'y':track['proc_track']['y'][fidx]})
	for fidx, fname in tqdm.tqdm(enumerate(flist), total = len(flist)):
		if fidx in dictGT: # This frame has label
			for gtThisFrame in dictGT[fidx]: # What this label is ?
				faceGT = gtThisFrame[0:4]
				labelGT = gtThisFrame[4]
				idGT = gtThisFrame[5]
				ious = []
				for face in faces[fidx]: # Find the right face in my result
					faceLocation = [int(face['x']-face['s']), int(face['y']-face['s']), int(face['x']+face['s']), int(face['y']+face['s'])]
					faceLocation_new = [int(face['x']-face['s']) // 2, int(face['y']-face['s']) // 2, int(face['x']+face['s']) // 2, int(face['y']+face['s']) // 2]
					iou = bb_intersection_over_union(faceLocation_new, faceGT, evalCol = True)
					if iou > 0.5:
						ious.append([iou, round(face['score'],2)])
				if len(ious) > 0: # Find my result
					ious.sort()
					labelPredict = ious[-1][1]
				else:					
					labelPredict = 0
				x1 = faceGT[0]
				y1 = faceGT[1]
				width = faceGT[2] - faceGT[0]
				predictionSet[idGT][0].append(labelPredict)
				predictionSet[idGT][1].append(labelGT)
	names = ['long', 'bell', 'boll', 'lieb', 'sick', 'abbas'] # Evaluate
	names.sort()
	F1s = 0
	for i in names:
		scores = numpy.array(predictionSet[i][0])
		labels = numpy.array(predictionSet[i][1])
		scores = numpy.int64(scores > 0)
		F1 = f1_score(labels, scores)
		ACC = accuracy_score(labels, scores)
		if i != 'abbas':
			F1s += F1
			print("%s, ACC:%.2f, F1:%.2f"%(i, 100 * ACC, 100 * F1))
	print("Average F1:%.2f"%(100 * (F1s / 5)))

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
	print(" args.minTrack is :" , args.minTrack)
	print(" numFailedDet is : ", args.numFailedDet)

	video_folder = args.videoFolder
	# target_folder = args.targetFolder

	args.total_frames = 0
	args.good_frames_200 = 0
	args.good_frames_224 = 0
	args.good_frames_256 = 0
	args.good_frames_300 = 0

	# target_name_list = os.listdir(args.targetFolder)
	# print(target_name_list)
	# current_name_list = os.listdir(video_folder)
	# print(current_name_list)
	# to_process_name = []
	# for item in current_name_list:
	# 	if item not in target_name_list:
	# 		to_process_name.append(item)
	# print(to_process_name)

	# to_process_name = current_name_list
	# to_process_name = np.load("to_process_name.npy")
	to_process_name = os.listdir(args.targetFolder)
	# print(to_process_name)
	# print(to_process_name_1)

	args.h_mean = []
	args.w_mean = []

	to_process_name.sort()

	if args.data_parallel_num:

		L = len(to_process_name)
		job_len = L // args.data_parallel_num
		start_idx = args.job_num * job_len
		end_idx = start_idx + job_len
		print("\n\n ** \n\n Data parallel total job is ",args.data_parallel_num , "job len " , job_len, "start_idx", start_idx, " end_idx", end_idx )
		to_process_name = to_process_name[start_idx:end_idx]



	for video_name in to_process_name:
		video_path = os.path.join(video_folder, video_name)

		args.videoPath = video_path
		args.savePath = os.path.join(args.targetFolder, video_name)

		# Initialization
		args.pyaviPath = os.path.join(args.savePath, 'pyavi')
		args.pyframesPath = os.path.join(args.savePath, 'pyframes')
		args.pyworkPath = os.path.join(args.savePath, 'pywork')
		args.pycropPath = os.path.join(args.savePath, 'pycrop')
		args.pywholePath = os.path.join(args.savePath, 'pywhole')


		if os.path.exists(args.pycropPath):
			rmtree(args.pycropPath)
		if os.path.exists(args.pyworkPath):
			rmtree(args.pyworkPath)
		if os.path.exists(args.pywholePath):
			rmtree(args.pywholePath)

		os.makedirs(args.pyaviPath, exist_ok = True) # The path for the input video, input audio, output video
		os.makedirs(args.pyframesPath, exist_ok = True) # Save all the video frames
		os.makedirs(args.pyworkPath, exist_ok = True) # Save the results in this process by the pckl method
		os.makedirs(args.pycropPath, exist_ok = True) # Save the detected face clips (audio+video) in this process
		os.makedirs(args.pywholePath, exist_ok=True)  # Save the detected face clips (audio+video) in this process



		# Extract video
		args.videoFilePath = os.path.join(args.pyaviPath, 'video.avi')
		# If duration did not set, extract the whole video, otherwise extract the video from 'args.start' to 'args.start + args.duration'
		# time_1 = time.time()
		# print(f"Extracting video {video_name}")
		# if args.duration == 0:
		# 	command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -async 1 -r 25 %s -loglevel panic" % \
		# 		(args.videoPath, args.nDataLoaderThread, args.videoFilePath))
		# else:
		# 	command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -ss %.3f -to %.3f -async 1 -r 25 %s -loglevel panic" % \
		# 		(args.videoPath, args.nDataLoaderThread, args.start, args.start + args.duration, args.videoFilePath))
		# subprocess.call(command, shell=True, stdout=None)
		# sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the video and save in %s \r\n" %(args.videoFilePath))
		#
		# print("Extract video done, time cost %.3f s"%(time.time() - time_1))
		# cost ~= 0.1 * video length , 60 min video will take about 6 min.

		# Extract audio

		# time_2 = time.time()

		# args.audioFilePath = os.path.join(args.pyaviPath, 'audio.wav')
		args.audioFilePath = os.path.join(args.pyaviPath,"audio_(Vocals)_Kim_Vocal_2.wav")

		# command = ("ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" % \
		# 	(args.videoFilePath, args.nDataLoaderThread, args.audioFilePath))
		# subprocess.call(command, shell=True, stdout=None)
		# sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the audio and save in %s \r\n" %(args.audioFilePath))
		# print("Extract audio done, time cost %.3f s"%(time.time() - time_2))
		# Tiny time cost, can be ignored


		# time_3 = time.time()
		# Extract the video frames
		# command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -f image2 %s -loglevel panic" % \
		# 	(args.videoFilePath, args.nDataLoaderThread, os.path.join(args.pyframesPath, '%06d.jpg')))
		# subprocess.call(command, shell=True, stdout=None)
		# sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the frames and save in %s \r\n" %(args.pyframesPath))
		# print("Extract video frame done, time cost %.3f s"%(time.time() - time_3))
		# cost ~= 0.12 * video length

		time_4 = time.time()
		# Scene detection for the video frames
		scene = scene_detect(args)
		sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scene detection and save in %s \r\n" %(args.pyworkPath))
		print("Scene detection done, time cost %.3f s"%(time.time() - time_4))
		# cost ~= 0.28 * video length  before multi process

		time_5 = time.time()
		# Face detection for the video frames
		faces = inference_video(args, n_worker=args.worker)
		sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face detection and save in %s \r\n" %(args.pyworkPath))
		print("face detection done, time cost %.3f s"%(time.time() - time_5))


		time_6 = time.time()

		# Face tracking multi process (not necessary)
		# allTracks = process_shots(args, scene, faces, n_worker=args.worker)

		allTracks= []
		for shot in scene:
			if shot[1].frame_num - shot[0].frame_num >= args.minTrack: # Discard the shot frames less than minTrack frames
				allTracks.extend(track_shot(args, faces[shot[0].frame_num:shot[1].frame_num])) # 'frames' to present this tracks' timestep, 'bbox' presents the location of the faces
		# track shot will calculate the IOU between consective frames, if IOU >  0.5 stop
		sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face track and detected %d tracks \r\n" %len(allTracks))
		print("face tracking, time cost %.3f s"%(time.time() - time_6))
		# print("\n\n ** all tracks detected \n\n")
		# print(allTracks)

		time_7 = time.time()
		# Face clips cropping

		# --------------non-multiprocess version ------------------------------
		# vidTracks = []
		# print("\n\n corping video \n\n")
		# for ii, track in tqdm.tqdm(enumerate(allTracks), total = len(allTracks)):
		# 	# print("\n\n corping video \n\n")
		# 	# print(ii, track)
		# 	# corped_video = crop_video(args, track, os.path.join(args.pycropPath, '%05d'%ii))
		# 	# whole_video = crop_whole_video(args, track, os.path.join(args.pywholePath, '%05d'%ii  ))
		# 	corped_video = crop_video_whole(args, track, os.path.join(args.pycropPath, '%05d'%ii),os.path.join(args.pywholePath, '%05d'%ii  ) )
		# 	vidTracks.append(  corped_video    )
		# --------------non-multiprocess version ------------------------------

		vidTracks = process_tracks(args, allTracks, n_workers=args.worker)

		savePath = os.path.join(args.pyworkPath, 'tracks.pckl')
		with open(savePath, 'wb') as fil:
			pickle.dump(vidTracks, fil)

		print("face clip crop done, time cost %.3f s"%(time.time() - time_7))

		sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face Crop and saved in %s tracks \r\n" %args.pycropPath)
		fil = open(savePath, 'rb')
		vidTracks = pickle.load(fil)

		# print("\n\n *** vid_trackers is  \n\n " , vidTracks )

		time_8 = time.time()
		# Active Speaker Detection by TalkNet
		files = glob.glob("%s/*.avi"%args.pycropPath)
		# print("\n\n *** \n\n " , files , args.pyworkPath)
		files.sort()
		scores = evaluate_network(files, args)
		score_thresh = 0.8
		print("\n\n score_thresh is : ", score_thresh)
		# FILTER tracks with high score without multiple faces
		filter_with_score(vidTracks, scores, args, score_thresh)
		print("face with score done, time cost %.3f s"%(time.time() - time_8))


		savePath = os.path.join(args.pyworkPath, 'scores.pckl')
		with open(savePath, 'wb') as fil:
			pickle.dump(scores, fil)
		sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scores extracted and saved in %s \r\n" %args.pyworkPath)

		# Generate visualization video
		# This is only for validation, you can remove this part for faster speed.
		time_9 = time.time()
		# visualization(vidTracks, scores, args)
		# print("visualization done , time cost %.3f s" % (time.time() - time_9))
		print(f"Total frames are {args.total_frames}, \n"
			  f" good frames > 200 , {args.good_frames_200} ,  data_ratio {args.good_frames_200 / args.total_frames},\n "
			  f" good frames > 224 , {args.good_frames_224} ,  data_ratio {args.good_frames_224 / args.total_frames},\n "
			  f" good frames > 256, {args.good_frames_256} ,  data_ratio {args.good_frames_256 / args.total_frames},\n"
			  f" good frames > 300, {args.good_frames_300} ,  data_ratio {args.good_frames_300 / args.total_frames},\n")
	print("\n\n h_mean = " ,args.h_mean, "\n\n w_mean = ", args.w_mean)


if __name__ == '__main__':
    main()
