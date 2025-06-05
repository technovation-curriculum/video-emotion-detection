# this version of the app will continuously record audio and run it through the emotion detection model
# user presses a button to start and stop the audio capture and analysis

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
import threading
import queue

if 'initialized' not in st.session_state:
    st.session_state['initialized'] = True
    st.session_state['recordings'] = []
    st.session_state['results'] = []
    st.session_state['audio_device'] = None
    st.session_state['is_recording'] = False
    st.session_state['audio_queue'] = queue.Queue()
    st.session_state['current_emotion'] = "neutral"
    st.session_state['emotion_history'] = []
    st.session_state['recording_thread'] = None
    st.session_state['analysis_thread'] = None

class ThreadController:
    def __init__(self):
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.current_emotion = "neutral"
        self.emotion_history = []

# At the beginning of your script, after imports
if 'controller' not in st.session_state:
    st.session_state['controller'] = ThreadController()

# Page configuration
st.set_page_config(page_title="Voice Emotion Detection", layout="wide")

# Create directories if they don't exist
if not os.path.exists("recordings"):
    os.makedirs("recordings")
if not os.path.exists("results"):
    os.makedirs("results")


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

def record_audio_segment(duration=3, sample_rate=22050, device=None, queue=None):
    """Record audio for the specified duration and add to queue."""
    try:
        if device is not None:
            recording = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                device=device
            )
            sd.wait()
            if queue is not None:
                queue.put(recording)
            return recording
        else:
            return None
    except Exception as e:
        print(f"Error recording audio: {str(e)}")
        return None

def continuous_recording(controller, duration=3, sample_rate=22050, device=None):
    """Continuously record audio in segments while recording flag is True."""
    print(f"Recording thread started with device: {device}")
    try:
        while controller.is_recording:
            try:
                print(f"Attempting to record {duration}s segment")
                recording = sd.rec(
                    int(duration * sample_rate),
                    samplerate=sample_rate,
                    channels=1,
                    device=device
                )
                sd.wait()
                print(f"Recording complete. Shape: {recording.shape}")
                
                if controller.audio_queue is not None:
                    controller.audio_queue.put(recording)
                    print(f"Added recording to queue. Queue size: {controller.audio_queue.qsize()}")
            except Exception as e:
                print(f"Error in continuous recording: {str(e)}")
                import traceback
                print(traceback.format_exc())
                time.sleep(0.5)
    except Exception as e:
        print(f"Recording thread error: {str(e)}")
    print("Recording thread exiting")

def continuous_analysis(controller, emotion_detector, response_generator, segment_duration=3):
    """Continuously analyze audio segments from the queue."""
    print("Analysis thread started")
    try:
        while controller.is_recording:
            try:
                if not controller.audio_queue.empty():
                    print("Processing audio segment")
                    audio_data = controller.audio_queue.get()
                    
                    if audio_data is not None:
                        print(f"Audio data shape: {audio_data.shape}")
                        result = emotion_detector.detect_emotion_from_numpy(audio_data, sample_rate=22050)
                        print(f"Detection result: {result}")
                        
                        if "error" not in result:
                            # Use controller instead of session state
                            controller.current_emotion = result['predicted_emotion']
                            print(f"Updated current emotion to: {controller.current_emotion}")
                            controller.emotion_history.append({
                                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                                "emotion": result['predicted_emotion'],
                                "scores": result['confidence_scores']
                            })
                            
                            # Limit history size
                            if len(controller.emotion_history) > 20:
                                controller.emotion_history = controller.emotion_history[-20:]
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error in analysis loop: {str(e)}")
                import traceback
                print(traceback.format_exc())
                time.sleep(0.5)
    except Exception as e:
        print(f"Analysis thread error: {str(e)}")
    print("Analysis thread exiting")
    
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
        print("ENTERING EMOTION DETECTION FUNCTION") 
        """Process numpy array directly without file operations"""
        try:
            # Flatten array if it has more than one dimension (e.g., channels)
            if len(audio_array.shape) > 1:
                audio_array = audio_array.flatten()

            # Check for silence/no speech
            audio_energy = np.mean(np.abs(audio_array))
            rms = np.sqrt(np.mean(np.square(audio_array)))
            zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_array)))) / len(audio_array)
            peak_amplitude = np.max(np.abs(audio_array))
            # Print diagnostic info to console
            print(f"Audio diagnostics - RMS: {rms:.5f}, Energy: {audio_energy:.5f}, "
              f"Zero crossing rate: {zero_crossings:.5f}, Peak: {peak_amplitude:.5f}")


            # Silence detection threshold - may need adjustment based on your microphone
            # A typical value for silence is when RMS is below 0.02-0.05 for normalized audio
            if rms < 0.01 or audio_energy < 0.008 or peak_amplitude < 0.05:
                print("silence detected"),
                return {
                    "predicted_emotion": "silence",
                    "confidence_scores": {emotion: 0 for emotion in self.emotions}
                }
            if audio_energy < 0.01:  # This threshold may need adjustment based on your microphone
                return {
                    "predicted_emotion": "silence",
                    "confidence_scores": {emotion: 0 for emotion in self.emotions}
                }                
            # Ensure audio is in float32 format and normalize between -1 and 1
            audio_array = audio_array.astype(np.float32)
            if audio_array.max() > 1.0 or audio_array.min() < -1.0:
                audio_array = audio_array / max(abs(audio_array.max()), abs(audio_array.min()))
            
            # Resample to 16kHz directly on numpy array using scipy
            if sample_rate != 16000:
                number_of_samples = round(len(audio_array) * 16000 / sample_rate)
                audio_array = signal.resample(audio_array, number_of_samples)
            
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
            print(f"Error in emotion detection: {str(e)}")
            print(traceback.format_exc())
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
            ],
            "silence": [
                ""
            ]
        }
    def get_response(self, emotion):
        """Generate a response based on detected emotion"""
        print(f"Generating response for emotion: {emotion}")  # Add this line
        if emotion in self.response_templates:
            responses = self.response_templates[emotion]
            response = np.random.choice(responses)
            # print(f"Selected response: {response}")  # Add this line
            return response
        else:
            print(f"No template for emotion: {emotion}, using default")  # Add this line
            return "Thank you for sharing your voice. Let's continue our activity together."

# def display_current_emotion(emotion, response_generator):
#     print(f"Displaying emotion: {controller.current_emotion}")  # Add this line
#     """Display current detected emotion and supportive response"""
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.subheader("Current Emotion:")
#         st.markdown(f"**{emotion.capitalize()}**")
        
#         # Generate and display supportive response
#         response = response_generator.get_response(emotion)
#         st.subheader("Response:")
#         st.markdown(f"*\"{response}\"*")
    
#     with col2:
#         st.subheader("Emotion History:")
        
#         # Create emotion history chart
#         if st.session_state.emotion_history:
#             # Extract emotions and timestamps
#             emotions = [entry["emotion"].capitalize() for entry in st.session_state.emotion_history]
#             timestamps = [entry["timestamp"][11:] for entry in st.session_state.emotion_history]  # Just time portion
            
#             # Create a simple line chart showing emotion transitions
#             emotion_to_num = {
#                 'Angry': 0, 'Disgust': 1, 'Fearful': 2, 'Sad': 3, 
#                 'Neutral': 4, 'Calm': 5, 'Happy': 6, 'Surprised': 7, 'Silence': 8
#             }
            
#             emotion_values = [emotion_to_num.get(e, 4) for e in emotions]  # Default to neutral (4)
            
#             # Plot the emotion transitions
#             fig, ax = plt.subplots(figsize=(10, 3))
#             ax.plot(range(len(emotions)), emotion_values, 'o-')
#             ax.set_yticks(list(emotion_to_num.values()))
#             ax.set_yticklabels(list(emotion_to_num.keys()))
#             ax.set_title("Emotion Transitions")
#             ax.set_xlabel("Time")
#             ax.set_xticks([])  # Hide x-ticks for cleaner look
#             st.pyplot(fig)

# def display_emotion_confidence(emotion_history):
#     """Display confidence scores for the most recent emotion detection"""
#     if emotion_history:
#         latest_entry = emotion_history[-1]
#         scores = latest_entry["scores"]
        
#         # Sort emotions by confidence score (descending)
#         sorted_emotions = sorted(
#             scores.items(),
#             key=lambda x: x[1],
#             reverse=True
#         )
        
#         # Create a bar chart of confidence scores
#         emotion_names = [emotion.capitalize() for emotion, _ in sorted_emotions]
#         confidence_values = [score for _, score in sorted_emotions]
        
#         st.subheader("Latest Confidence Scores:")
        
#         # Create chart data
#         chart_data = {"Emotion": emotion_names, "Confidence (%)": confidence_values}
        
#         # Display as a bar chart
#         st.bar_chart(chart_data)

def main():
    st.title("Continuous Voice Emotion Detection")
    st.write("This app continuously records your voice and detects emotions in real-time.")
    
    # Initialize emotion detector and response generator
    @st.cache_resource
    def load_emotion_detector():
        return EmotionDetector()
    
    emotion_detector = load_emotion_detector()
    response_generator = ResponseGenerator()

    # Initialize the controller if not already done
    if 'controller' not in st.session_state:
        st.session_state['controller'] = ThreadController()

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
        
        # Recording segment duration
        segment_duration = st.slider(
            "Recording segment duration (seconds)", 
            min_value=1, 
            max_value=5, 
            value=3,
            help="Each segment of audio will be this long before being analyzed"
        )
        
    # Start recording button
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Continuous Recording", key="start_button"):
            print("Starting recording process")
            
            # Create a new controller or reset the existing one
            controller = st.session_state['controller']
            controller.is_recording = True
            controller.audio_queue = queue.Queue()
            
            # Clear history
            st.session_state['emotion_history'] = []
            
            # Create new threads with controller
            recording_thread = threading.Thread(
                target=continuous_recording,
                args=(controller, segment_duration, 22050, st.session_state['audio_device'])
            )
            recording_thread.daemon = True
            
            analysis_thread = threading.Thread(
                target=continuous_analysis,
                args=(controller, emotion_detector, response_generator, segment_duration)
            )
            analysis_thread.daemon = True
            
            # Start threads
            recording_thread.start()
            analysis_thread.start()
            
            # Store threads
            st.session_state['recording_thread'] = recording_thread
            st.session_state['analysis_thread'] = analysis_thread
            
            st.rerun()

    with col2:
        if st.button("Stop Recording", key="stop_button"):
            print("Stopping recording process")
            controller = st.session_state['controller']
            controller.is_recording = False
            time.sleep(0.5)  # Give threads a moment to clean up
            st.rerun()
        
        # Display recording status
        if st.session_state['is_recording'] :
            st.warning("Recording in progress... Speak normally and your emotions will be analyzed continuously.")
            
            # Create a placeholder for the results that will be updated
            # results_placeholder = st.empty()
            # confidence_placeholder = st.empty()
            
            # # Use a container for the results to update in real-time
            # with results_placeholder.container():
            #     display_current_emotion(st.session_state.current_emotion, response_generator)
            
            # with confidence_placeholder.container():
            #     if st.session_state.emotion_history:
            #         display_emotion_confidence(st.session_state.emotion_history)
            
            # Auto-refresh the app to show updated results
            time.sleep(segment_duration)
            st.rerun()

    
    # Display current state regardless of recording status
    st.subheader("Current Emotion:")
    current_emotion = st.session_state['controller'].current_emotion
    st.markdown(f"**{current_emotion.capitalize()}**")
    
    # Generate and display response
    st.subheader("Response:")
    response = response_generator.get_response(current_emotion)
    st.markdown(response) 
    
    # Display emotions history if available
    if st.session_state['controller'].emotion_history:
        # Display confidence scores for the most recent detection
        # st.subheader("Latest Confidence Scores:")
        # latest_entry = st.session_state['controller'].emotion_history[-1]
        # scores = latest_entry["scores"]
        
        # # Sort emotions by confidence score (descending)
        # sorted_emotions = sorted(
        #     scores.items(),
        #     key=lambda x: x[1],
        #     reverse=True
        # )
        
        # # Create a bar chart
        # emotion_names = [emotion.capitalize() for emotion, _ in sorted_emotions]
        # confidence_values = [score for _, score in sorted_emotions]
        
        # # Create chart data
        # chart_data = {"Emotion": emotion_names, "Confidence (%)": confidence_values}
        # st.bar_chart(chart_data)
        
        # Display emotion history chart
        st.subheader("Emotion History:")
        emotions = [entry["emotion"].capitalize() for entry in st.session_state['controller'].emotion_history]
        
        # Create emotion mapping
        emotion_to_num = {
            'Angry': 0, 'Disgust': 1, 'Fearful': 2, 'Sad': 3, 
            'Neutral': 4, 'Calm': 5, 'Happy': 6, 'Surprised': 7, 'Silence': 8
        }
        
        # Map emotions to numeric values
        emotion_values = [emotion_to_num.get(e, 4) for e in emotions]  # Default to neutral (4)
        
        # Plot the emotion transitions
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(range(len(emotions)), emotion_values, 'o-')
        ax.set_yticks(list(emotion_to_num.values()))
        ax.set_yticklabels(list(emotion_to_num.keys()))
        ax.set_title("Emotion Transitions")
        ax.set_xlabel("Time")
        ax.set_xticks([])  # Hide x-ticks for cleaner look
        st.pyplot(fig)
    
    # If recording is active, set up automatic refresh
    if st.session_state['controller'].is_recording:
        st.warning("Recording in progress... Speak normally and your emotions will be analyzed continuously.")
        time.sleep(segment_duration)
        st.rerun()
if __name__ == "__main__":
    main()