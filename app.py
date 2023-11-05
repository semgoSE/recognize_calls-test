from flask import Flask, request, jsonify
import whisperx

device = "cpu" 
audio_file = "audio.wav"
batch_size = 1
compute_type = "int8"

app = Flask(__name__)

model = whisperx.load_model("medium", device, compute_type=compute_type)

@app.route('/')
def main(): 
    content = request.json

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=1)
    print(result["segments"])

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    print(result["segments"])
    
    diarize_model = whisperx.DiarizationPipeline(use_auth_token="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", device=device)

    diarize_segments = diarize_model(audio)

    result = whisperx.assign_word_speakers(diarize_segments, result)
    print(diarize_segments)
    print(result["segments"])    
    
    return 'Hello, World!'
app.run()