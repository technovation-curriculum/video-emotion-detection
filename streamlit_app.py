# this version just records audio or allows for audio file upload
# the audio is then processed by the model and the result is displayed

import streamlit as st
import torch
import torchaudio
import numpy as np
import tempfile
import os
import time
import soundfile as sf
import matplotlib.pyplot as plt
from datetime import datetime
import sounddevice as sd
from scipy import signal
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

# Page configuration
st.set_page_config(page_title="Voice Emotion Detection", layout="wide")

# Create directories if they don't exist
if not os.path.exists("recordings"):
    os.makedirs("recordings")
if not os.path.exists("results"):
    os.makedirs("results")

# Initialize session state
if 'recordings' not in st.session_state:
    st.session_state.recordings = []
if 'results' not in st.session_state:
    st.session_state.results = []
if 'audio_device' not in st.session_state:
    st.session_state.audio_device = None

def initialize_audio():
    """Initialize audio device and return available devices."""
    try:
        devices = sd.query_devices()
        input_devices = [i for i, d in enumerate(devices) if d['max_input_channels'] > 0]
        if input_devices:
            return devices, input_devices
        else:
            return None, []
    except Exception as e:
        st.error(f"Error initializing audio: {str(e)}")
        return None, []

def record_audio(duration=10, sample_rate=22050, device=None):
    """Record audio for the specified duration."""
    try:
        if device is not None:
            recording = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                device=device
            )
            sd.wait()
            return recording
        else:
            st.error("No audio input device selected")
            return None
    except Exception as e:
        st.error(f"Error recording audio: {str(e)}")
        return None

class EmotionDetector:
    def __init__(self, model_name="Dpngtm/wav2vec2-emotion-recognition"):
        """Initialize the emotion detector with a pre-trained model"""
        st.info("Loading emotion detection model... This may take a moment.")
        
        try:
            # Load pre-trained model and processor
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            
            # Check for GPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            
            # Map of emotions (adjust based on the specific model's output labels)
            self.emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            raise e
    
    def detect_emotion_from_numpy(self, audio_array, sample_rate=22050):
        """Process numpy array directly without file operations"""
        try:
            # Flatten array if it has more than one dimension (e.g., channels)
            if len(audio_array.shape) > 1:
                audio_array = audio_array.flatten()
                
            # Ensure audio is in float32 format and normalize between -1 and 1
            audio_array = audio_array.astype(np.float32)
            if audio_array.max() > 1.0 or audio_array.min() < -1.0:
                audio_array = audio_array / max(abs(audio_array.max()), abs(audio_array.min()))
            
            # Debug information
            st.write(f"Audio array shape: {audio_array.shape}")
            st.write(f"Audio array dtype: {audio_array.dtype}")
            st.write(f"Audio array range: {audio_array.min():.4f} to {audio_array.max():.4f}")
            
            # Resample to 16kHz directly on numpy array using scipy
            if sample_rate != 16000:
                number_of_samples = round(len(audio_array) * 16000 / sample_rate)
                audio_array = signal.resample(audio_array, number_of_samples)
                st.write(f"Resampled to 16kHz: {audio_array.shape}")
            
            # Process directly with processor - wav2vec2 expects raw waveform as input
            inputs = self.processor(
                audio_array, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            )
            
            # Move inputs to device (CPU/GPU)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get predicted emotion
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
            
            # Create emotion confidence mapping
            emotion_scores = {self.emotions[i]: round(probabilities[i].item() * 100, 2) 
                             for i in range(len(self.emotions))}
            
            return {
                "predicted_emotion": self.emotions[predicted_class],
                "confidence_scores": emotion_scores
            }
        except Exception as e:
            import traceback
            st.error(f"Error in emotion detection: {str(e)}")
            st.error(traceback.format_exc())
            return {"error": str(e)}
    
    def detect_emotion_from_file(self, file_path):
        """Process uploaded audio file with detailed debugging"""
        st.write("DEBUG: Starting file analysis")
        try:
            # Step 1: Check file existence
            if not os.path.exists(file_path):
                st.error(f"File not found: {file_path}")
                return {"error": f"File not found: {file_path}"}
            
            st.write(f"DEBUG: File exists: {file_path}, size: {os.path.getsize(file_path)} bytes")
            
            try:
                # Step 2: Try with soundfile first
                st.write("DEBUG: Attempting to load with soundfile")
                audio_array, sample_rate = sf.read(file_path, always_2d=False)
                st.write(f"DEBUG: Loaded with soundfile, shape: {audio_array.shape}, sample_rate: {sample_rate}")
            except Exception as sf_error:
                st.write(f"DEBUG: soundfile error: {str(sf_error)}")
                try:
                    # Step 3: Try with torchaudio as backup
                    st.write("DEBUG: Attempting to load with torchaudio")
                    waveform, sample_rate = torchaudio.load(file_path)
                    audio_array = waveform.squeeze().numpy()
                    st.write(f"DEBUG: Loaded with torchaudio, shape: {audio_array.shape}, sample_rate: {sample_rate}")
                except Exception as torch_error:
                    st.write(f"DEBUG: torchaudio error: {str(torch_error)}")
                    # Step 4: Last resort - try ffmpeg via custom subprocess if it's installed
                    try:
                        st.write("DEBUG: Attempting to use ffmpeg")
                        import subprocess
                        # Convert to WAV using ffmpeg
                        output_path = file_path + "_converted.wav"
                        subprocess.run(['ffmpeg', '-i', file_path, '-ar', '16000', '-ac', '1', '-y', output_path], 
                                      check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        st.write(f"DEBUG: Converted with ffmpeg to {output_path}")
                        # Try again with the converted file
                        audio_array, sample_rate = sf.read(output_path, always_2d=False)
                        st.write(f"DEBUG: Loaded converted file, shape: {audio_array.shape}, sample_rate: {sample_rate}")
                        # Clean up
                        os.remove(output_path)
                    except Exception as ffmpeg_error:
                        st.write(f"DEBUG: ffmpeg error: {str(ffmpeg_error)}")
                        return {"error": f"Could not load audio file with any available method. Original error: {str(sf_error)}"}
            
            # Step 5: Process the audio data
            st.write("DEBUG: Starting emotion detection on the loaded audio")
            result = self.detect_emotion_from_numpy(audio_array, sample_rate)
            st.write("DEBUG: Emotion detection completed")
            
            return result
            
        except Exception as e:
            import traceback
            st.write("DEBUG: General exception in detect_emotion_from_file")
            st.write(traceback.format_exc())
            return {"error": str(e)}

class ResponseGenerator:
    def __init__(self):
        """Initialize the response generator with templates for different emotions"""
        self.response_templates = {
            "angry": [
                "I notice you seem frustrated. Taking deep breaths can help calm your mind.",
                "It's okay to feel upset sometimes. Remember that challenges are temporary.",
                "I sense some frustration. Would you like to take a short break before continuing?"
            ],
            "calm": [
                "You have a wonderful sense of calm. That's perfect for learning origami!",
                "Your calm demeanor is impressive. It helps with focus and precision.",
                "I appreciate your peaceful approach. It makes learning new skills more enjoyable."
            ],
            "disgust": [
                "I sense you might be feeling uncomfortable. Let's try a different approach.",
                "It seems this might not be to your liking. We could try something else if you prefer.",
                "Sometimes things don't feel right at first. Would you like to try something different?"
            ],
            "fearful": [
                "It's completely normal to feel uncertain when trying something new. You're doing fine.",
                "Don't worry about making mistakes - they're part of the learning process.",
                "I sense some hesitation, which is natural. Take your time and proceed at your own pace."
            ],
            "happy": [
                "Your enthusiasm is wonderful! It makes learning so much more enjoyable.",
                "I love hearing the joy in your voice! You're doing great with this activity.",
                "Your positive energy is contagious! It's a pleasure to work with you."
            ],
            "neutral": [
                "You're maintaining good focus. That's perfect for learning new skills.",
                "Your steady approach is great for mastering techniques step by step.",
                "I appreciate your attentiveness. It helps make progress steady and consistent."
            ],
            "sad": [
                "I sense you might be feeling a bit down. Remember that it's okay to take things slowly.",
                "Sometimes we all feel a little blue. Would a simpler activity help lift your spirits?",
                "Your feelings are valid. Would you like to try something that might bring you some joy?"
            ],
            "surprised": [
                "That reaction is perfectly natural! New discoveries can be quite surprising.",
                "Surprises keep things interesting, don't they? Your curiosity is wonderful.",
                "I notice your surprise! It's exciting when things turn out unexpectedly."
            ]
        }
    
    def get_response(self, emotion):
        """Generate a response based on detected emotion"""
        if emotion in self.response_templates:
            responses = self.response_templates[emotion]
            return np.random.choice(responses)
        else:
            return "Thank you for sharing your voice. Let's continue our activity together."

def display_results(result, response_generator):
    """Display emotion detection results and supportive response"""
    st.subheader("Detected Emotion:")
    st.markdown(f"**{result['predicted_emotion'].capitalize()}**")
    
    # Generate and display supportive response
    response = response_generator.get_response(result['predicted_emotion'])
    st.subheader("Response:")
    st.markdown(f"*\"{response}\"*")
    
    # Display confidence scores
    st.subheader("Confidence Scores:")
    
    # Sort emotions by confidence score (descending)
    sorted_emotions = sorted(
        result['confidence_scores'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Create a bar chart of confidence scores
    emotion_names = [emotion.capitalize() for emotion, _ in sorted_emotions]
    confidence_values = [score for _, score in sorted_emotions]
    
    # Create a dictionary for the chart
    chart_data = {"Emotion": emotion_names, "Confidence (%)": confidence_values}
    
    # Display as a bar chart
    st.bar_chart(chart_data)
    
    # Display detailed confidence scores
    for emotion, score in sorted_emotions:
        st.write(f"{emotion.capitalize()}: {score}%")

def main():
    st.title("Voice Emotion Detection App")
    st.write("Record your voice or upload an audio file to detect emotions and receive a supportive response.")
    
    # Initialize emotion detector and response generator
    @st.cache_resource
    def load_emotion_detector():
        return EmotionDetector()
    
    emotion_detector = load_emotion_detector()
    response_generator = ResponseGenerator()
    
    # Audio input options
    input_option = st.radio("Choose input method:", ["Record Audio", "Upload Audio File"])
    
    if input_option == "Record Audio":
        st.write("Record your voice directly from your microphone")
        
        # Initialize audio devices
        devices, input_devices = initialize_audio()
        
        if devices is not None and input_devices:
            # Device selection
            device_options = [f"{i}: {devices[i]['name']}" for i in input_devices]
            selected_device_idx = st.selectbox(
                "Select audio input device:", 
                options=range(len(device_options)),
                format_func=lambda i: device_options[i]
            )
            st.session_state.audio_device = input_devices[selected_device_idx]
            
            # Recording duration
            duration = st.slider("Recording duration (seconds)", min_value=3, max_value=30, value=10)
            
            # Start recording button
            if st.button(f"Start Recording ({duration}s)"):
                with st.spinner(f"Recording for {duration} seconds..."):
                    # Record audio
                    audio_data = record_audio(
                        duration=duration, 
                        sample_rate=22050, 
                        device=st.session_state.audio_device
                    )
                    
                    if audio_data is not None:
                        # Save for playback
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"recordings/rec_{timestamp}.wav"
                        sf.write(filename, audio_data, 22050)
                        
                        # Add to session state
                        st.session_state.recordings.append({
                            "filename": filename,
                            "timestamp": timestamp
                        })
                        
                        # Display audio
                        st.audio(filename, format="audio/wav")
                        
                        # Visualize the audio
                        fig, ax = plt.subplots(figsize=(10, 2))
                        ax.plot(audio_data)
                        ax.set_title("Audio Waveform")
                        ax.set_xlabel("Sample")
                        ax.set_ylabel("Amplitude")
                        st.pyplot(fig)
                        
                        # Analyze emotions
                        with st.spinner("Analyzing emotions..."):
                            try:
                                result = emotion_detector.detect_emotion_from_numpy(audio_data, sample_rate=22050)
                                
                                if "error" in result:
                                    st.error(f"Error detecting emotion: {result['error']}")
                                else:
                                    # Display results
                                    display_results(result, response_generator)
                                    
                                    # Save results
                                    st.session_state.results.append({
                                        "timestamp": timestamp,
                                        "result": result
                                    })
                            except Exception as e:
                                st.error(f"Error in analysis: {str(e)}")
                    else:
                        st.error("Failed to record audio. Please check your microphone.")
        else:
            st.error("No audio input devices found. Please check your microphone connection.")
    
    else:  # Upload Audio File
        st.write("Upload an audio file for emotion analysis")
        
        # Simple file uploader
        uploaded_file = st.file_uploader("Choose a file", type=["wav"])
        
        if uploaded_file:
            # Display a message
            st.info("Processing your audio file...")
            
            # Display the audio
            st.audio(uploaded_file, format="audio/wav")
            
            # Save to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                audio_path = tmp_file.name
            
            try:
                # Load the file with scipy for better compatibility
                from scipy.io import wavfile
                sample_rate, audio_array = wavfile.read(audio_path)
                
                # Convert to float32 and normalize if needed
                if audio_array.dtype != np.float32:
                    audio_array = audio_array.astype(np.float32)
                    if audio_array.dtype.kind in 'iu':  # If it's an integer type
                        max_val = np.iinfo(audio_array.dtype).max
                        audio_array = audio_array / max_val
                
                # Visualize the audio
                fig, ax = plt.subplots(figsize=(10, 2))
                if len(audio_array.shape) > 1:  # If stereo, just plot first channel
                    ax.plot(audio_array[:, 0])
                else:
                    ax.plot(audio_array)
                ax.set_title("Audio Waveform")
                ax.set_xlabel("Sample")
                ax.set_ylabel("Amplitude")
                st.pyplot(fig)
                
                # Analyze emotions
                with st.spinner("Analyzing emotions..."):
                    # Trim if very long
                    if len(audio_array.shape) > 1:  # If stereo
                        # Convert to mono by averaging channels
                        audio_array = np.mean(audio_array, axis=1)
                    
                    # Trim if too long
                    if len(audio_array) > 480000:  # ~30 seconds at 16kHz
                        st.warning("Audio file is very long. Analyzing only the first 30 seconds.")
                        audio_array = audio_array[:480000]
                    
                    # Process
                    result = emotion_detector.detect_emotion_from_numpy(audio_array, sample_rate)
                
                # Clean up
                os.unlink(audio_path)
                
                # Display results
                if "error" in result:
                    st.error(f"Error detecting emotion: {result['error']}")
                else:
                    display_results(result, response_generator)
                
            except Exception as e:
                import traceback
                st.error(f"Error processing audio file: {str(e)}")
                st.error(traceback.format_exc())
                try:
                    os.unlink(audio_path)
                except:
                    pass

if __name__ == "__main__":
    main()