# pylint: disable=C0301
import os
from audio_separator.separator import Separator
import subprocess

# from audio_separator.separator import Separator

audio_separator_model_file = "audio_separator_v1/Kim_Vocal_2.onnx"

audio_separator_model_path = os.path.dirname(audio_separator_model_file)
audio_separator_model_name = os.path.basename(audio_separator_model_file)


def resample_audio(input_audio_file: str, output_audio_file: str, sample_rate: int):
    p = subprocess.Popen([
        "ffmpeg", "-y", "-v", "error", "-i", input_audio_file, "-ar", str(sample_rate), output_audio_file
    ])
    ret = p.wait()
    assert ret == 0, "Resample audio failed!"
    return output_audio_file



#num = 1 or 2
# 1. separate vocals
# modify
base_dir = "E:/test_audio_sep_08_28"
audio_dir = os.path.join(base_dir, "audios")
seperated_audio_dir = os.path.join(base_dir, "seperated")
sample_rate = 16000

if not os.path.exists(seperated_audio_dir):
    os.makedirs(seperated_audio_dir)

print("separator init")

print(seperated_audio_dir, audio_separator_model_path)

audio_separator = Separator(
    output_dir=seperated_audio_dir,
    output_single_stem="vocals",
    model_file_dir=audio_separator_model_path,
)

print("model loading")
audio_separator.load_model(audio_separator_model_name)
assert audio_separator.model_instance is not None, "Fail to load audio separate model."

wav_files = os.listdir(audio_dir)
for wav_file in wav_files:

    wav_path = os.path.join(audio_dir, wav_file)
    outputs = audio_separator.separate(wav_path)

    if len(outputs) <= 0:
        print("Audio separate failed.")
    else:
        print("Audio separate succeeded.")

    vocal_audio_file = outputs[0]
    vocal_audio_name, _ = os.path.splitext(vocal_audio_file)
    vocal_audio_file = os.path.join(audio_separator.output_dir, vocal_audio_file)
    vocal_audio_file = resample_audio(vocal_audio_file,
                                      os.path.join(audio_separator.output_dir, f"{vocal_audio_name}-16k.wav"),
                                      sample_rate)
