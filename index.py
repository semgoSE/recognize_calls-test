from faster_whisper import WhisperModel
import time
import subprocess
import os;

path = "audio.wav"
model_size = "medium"

nemo_process = subprocess.Popen(
    ["python3", "nemo_process.py", "-a", path, "--device", "cuda"],
)
model = WhisperModel(model_size, device="cuda", compute_type="float16")

segments, info = model.transcribe(path, beam_size=1, best_of=3)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    
    
# nemo_process.communicate()
# ROOT = os.getcwd()
# temp_path = os.path.join(ROOT, "temp_outputs")

# speaker_ts = []
# with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
#     lines = f.readlines()
#     for line in lines:
#         line_list = line.split(" ")
#         s = int(float(line_list[5]) * 1000)
#         e = s + int(float(line_list[8]) * 1000)
#         speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])