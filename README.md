# video-emotion-detection
An example Streamlit app that uses a machine learning model to detect a user's emotion 

There are 3 python files:
- ***streamlit_app.py*** allows recording or audio file upload and will predict emotion
- ***continuous_audio_detection.py*** continuously records audio when button pressed and continuously predicts emotion
- ***audio_detection+video_v2.py*** records video and audio and analyzes/predicts emotion from audio thread

Thanks to Deepan Gautam for his [wav2vec2-emotion-recogniton model](https://huggingface.co/Dpngtm/wav2vec2-emotion-recognition)  
Thanks to Steven R. Livingstone for his [RAVDESS speech dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
