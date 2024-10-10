import os
import shutil
import json
import argparse


parser = argparse.ArgumentParser(description = "move file to single folder")

parser.add_argument('--input_dir',             type=str, default="bili_data",   help='Demo video name')
parser.add_argument('--out_dir',           type=str, default="video",  help='Path for inputs')
parser.add_argument('--score_thresh',          type=float, default=0.5, help='score threshold')
parser.add_argument('--size_thresh',              type=int,   default=224,   help='face_size threshold')
parser.add_argument('--single_face',              type=bool,   default=True,   help='whether single face in frame')
parser.add_argument('--mode',              type=str,   default="single_video",   help='whether keep strcture or single video')
parser.add_argument('--frame_length_thresh',              type=int,   default=70,   help='face_size threshold')

# args = parser.parse_args()

def main():
    # upload_dir = 'NVAIE_Bili_24_09/'
    # upload_dir = "E:/Data/data_process/TalkNet-Data/data_upload/bili_video/"
    # processed_dir = "E:/Data/data_process/TalkNet-Data/data_processed/bili_data/"
    # score_threshold = 0.5
    # size_threshold = 224
    # mode = "video_folder"
    # mode = "keep_structure"
    args = parser.parse_args()
    print(args)
    mode = args.mode
    upload_dir = args.out_dir + "/"
    processed_dir = args.input_dir + "/"
    score_threshold = args.score_thresh
    size_threshold = args.size_thresh
    single_face_condition = args.single_face
    frame_threshold = args.frame_length_thresh

    file_uploaded = os.listdir(upload_dir)
    file_processed = os.listdir(processed_dir)


    print(file_processed)
    print(file_uploaded)

    file_todo = []

    for filename in file_processed:
        if filename not in file_uploaded:
            file_todo.append(filename)
    print(file_todo)

    for filename in file_todo:


        if mode == "keep_structure":
            target_dir = os.path.join(upload_dir, filename)
            os.makedirs(target_dir, exist_ok=True)
            os.makedirs(target_dir + "/pywhole", exist_ok=True)
            shutil.copy2(processed_dir + filename + "/filtered_data.json",
                         upload_dir + filename + "/filtered_data.json")
            shutil.copytree(processed_dir + filename + "/pywork",
                         upload_dir + filename + "/pywork")

        json_path = processed_dir + filename + "/filtered_data.json"
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(data)
        meta_data = data['meta_data']

        for item in meta_data:
            path = item[0]
            total_frames = item[1]
            score = item[2]
            single_face = item[-1]
            h = item[3]
            w = item[4]

            directory, file_name = os.path.split(path)
            print("directory: ", directory)
            print("file_name: ", file_name)

            size = (h+w) /2

            print(file_name, score, single_face, size)
            if total_frames > frame_threshold and score > score_threshold and size > size_threshold and (single_face or (not single_face_condition)):
                if mode == "keep_structure":
                    target_filename = filename.split(".")[0]
                    shutil.copy2(processed_dir + filename + "/pywhole/" + file_name,
                             upload_dir + filename + "/pywhole/" + file_name)
                elif mode == "single_folder":
                    target_filename = filename.split(".")[0]
                    target_file = upload_dir + target_filename + "_" + file_name
                    print(target_file)
                    shutil.copy2(processed_dir + filename + "/pywhole/" + file_name,
                                 target_file)


if __name__ == '__main__':
    main()
