from flask import Flask, request, jsonify
import whisperx
import requests

device = "cuda" 
audio_file = "audio.wav"
batch_size = 1
compute_type = "float16"

app = Flask(__name__)

model = whisperx.load_model("medium", device, compute_type=compute_type)

@app.route('/', methods = ["POST"])
def main():
    content = request.json
    
    print(content)
    
    apiKey = request.headers.get("Authorization")
    
    if (apiKey != "maNTAmbrOpto"):
        return "Bad API key"
    
    r = requests.get(content['fileUrl'], allow_redirects=True)
    open('audio.wav', 'wb').write(r.content)

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=1, language=content['language'])
    print(result["segments"])

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    print(result["segments"])
    
    def only_text(segment): 
        return segment.text
    
    texts = map(only_text, result["segments"])
    
    return jsonify({ text:  " ".join(texts)})
    
    # diarize_model = whisperx.DiarizationPipeline(use_auth_token="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", device=device)

    # diarize_segments = diarize_model(audio)

    # result = whisperx.assign_word_speakers(diarize_segments, result)
    # print(diarize_segments)
    # print(result["segments"])    
if __name__ == '__main__':
    app.run(debug=False, port=4001, host="0.0.0.0")