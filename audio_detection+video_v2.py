# This is the completed app that detects emotions from video and audio, and provides supportive responses for origami sessions
# It runs the video and audio capture in 2 separate threads, and uses just the audio thread for emotion analysis

import streamlit as st
import torch
import torchaudio
import numpy as np
import time
import soundfile as sf
import matplotlib.pyplot as plt
from datetime import datetime
import sounddevice as sd
from scipy import signal
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import threading
import queue
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import warnings

warnings.filterwarnings("ignore")

if 'initialized' not in st.session_state:
    st.session_state['initialized'] = True
if 'audio_device' not in st.session_state:
    st.session_state['audio_device'] = None
if 'audio_queue' not in st.session_state:
    st.session_state['audio_queue'] = queue.Queue()
    # st.session_state['current_emotion'] = "silence"
if 'recording_thread' not in st.session_state:
    st.session_state['recording_thread'] = None
if 'analysis_thread' not in st.session_state:
    st.session_state['analysis_thread'] = None
if 'last_emotion' not in st.session_state:
    st.session_state['last_emotion'] = "Neutral"
if 'current_emotion' not in st.session_state:
    st.session_state['current_emotion'] = "Neutral"
if 'emotion_history' not in st.session_state:
    st.session_state['emotion_history'] = []
if 'show_history' not in st.session_state:
    st.session_state['show_history'] = False
  

class ThreadController:
    def __init__(self):
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.results_queue = queue.Queue()

# At the beginning of your script, after imports
if 'controller' not in st.session_state:
    st.session_state['controller'] = ThreadController()

# Page configuration
st.set_page_config(page_title="Voice Emotion Detection", layout="wide")

@st.cache_data
def initialize_audio():
    """Initialize audio device with better error handling."""
    try:
        # Reset PortAudio
        sd._terminate()
        sd._initialize()
        
        devices = sd.query_devices()
        input_devices = []
        
        print("Scanning audio devices...")
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0:
                print(f"Found input device {i}: {d['name']}")
                # Test if device actually works
                try:
                    # Try a very short test recording
                    test_rec = sd.rec(
                        frames=1024, 
                        samplerate=22050, 
                        channels=1, 
                        device=i
                    )
                    sd.wait()
                    input_devices.append(i)
                    print(f"  ✓ Device {i} works")
                except Exception as e:
                    print(f"  ✗ Device {i} failed: {e}")
        
        return devices, input_devices
    except Exception as e:
        st.error(f"Error initializing audio: {str(e)}")
        return None, []

# def record_audio_segment(duration=5, sample_rate=22050, device=None, queue=None):
#     """Record audio for the specified duration and add to queue."""
#     try:
#         if device is not None:
#             recording = sd.rec(
#                 int(duration * sample_rate),
#                 samplerate=sample_rate,
#                 channels=1,
#                 device=device
#             )
#             sd.wait()
#             if queue is not None:
#                 queue.put(recording)
#             return recording
#         else:
#             return None
#     except Exception as e:
#         print(f"Error recording audio: {str(e)}")
#         return None

def continuous_recording(controller, duration=5, sample_rate=22050, device=None):
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
                    # print("Processing audio segment")
                    audio_data = controller.audio_queue.get()
                    
                    if audio_data is not None:
                        result = emotion_detector.detect_emotion_from_numpy(audio_data, sample_rate=22050)
                        # print(f"Detection result: {result}")
                        
                        if "error" not in result:
                            # Create complete emotion result
                            emotion_result = {
                                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                                "emotion": result['predicted_emotion'],
                                "scores": result['confidence_scores']
                            }
                            
                            # Put everything in queue - let main thread handle session_state
                            controller.results_queue.put(emotion_result)
                            
                            print(f"Queued emotion result: {result['predicted_emotion']}")
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
            # zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_array)))) / len(audio_array)
            peak_amplitude = np.max(np.abs(audio_array))
            # Print diagnostic info to console
            # print(f"Audio diagnostics - RMS: {rms:.5f}, Energy: {audio_energy:.5f}, "
            #  f"Zero crossing rate: {zero_crossings:.5f}, Peak: {peak_amplitude:.5f}")


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
                "Are you a bit frustrated? Take a few deep breaths. Maybe take a short break.",
                "It's okay to feel upset sometimes. Remember that challenges are temporary.",
                "I sense some frustration. Would you like to take a short break before continuing?"
            ],
            "calm": [
                "You have a wonderful sense of calm. That's perfect for learning origami!",
                "Your calm demeanor is impressive. It helps with focus and precision.",
                "I appreciate your peaceful approach. It makes learning new skills more enjoyable."
            ],
            "disgust": [
                "I sense you might be feeling uncomfortable. Let's try a different shape.",
                "It seems you are not enjoying this. Maybe you could try taking a break",
                "Sometimes things don't feel right at first. Would you like to try a different shape?"
            ],
            "fearful": [
                "It's completely normal to feel uncertain when trying something new. You're doing fine.",
                "Don't worry about making mistakes - they're part of the learning process.",
                "I sense some hesitation, which is natural. Take your time and proceed at your own pace."
            ],
            "happy": [
                "Your enthusiasm is wonderful! It makes learning so much more enjoyable.",
                "I love hearing the joy in your voice! You're doing great with this activity.",
                "Your positive energy is contagious!"
            ],
            "neutral": [
                "You're maintaining good focus. That's perfect for learning new skills.",
                "Your steady approach is great for mastering techniques step by step.",
                "I appreciate your attentiveness. It helps make progress steady and consistent."
            ],
            "sad": [
                "Remember that it's okay to take things slowly.",
                "Are you not enjoying this activity? Would a simpler shape help lift your spirits?",
                "You seem a bit down. Would you like to try something that might bring you some joy?"
            ],
            "surprised": [
                "Fun, isn't it!",
                "Your curiosity is wonderful.",
                "I notice your surprise! It's exciting when things turn out unexpectedly."
            ],
            "silence": [
                ""
            ]
        }
    def get_response(self, emotion):
        """Generate a response based on detected emotion"""
        print(f"Generating response for emotion: {emotion}")  
        if emotion in self.response_templates:
            responses = self.response_templates[emotion]
            response = np.random.choice(responses)
            return response
        else:
            print(f"No template for emotion: {emotion}, using default") 
            return "Thank you for sharing your voice. Let's continue our activity together."


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
    queue_size = 0

    # Initialize the controller if not already done
    if 'controller' not in st.session_state:
        st.session_state['controller'] = ThreadController()

    # Create columns to constrain the video width to 60%
    _, video_col, _ = st.columns([2, 6, 2])  # This creates a 60% middle column

    with video_col:
        if 'webrtc_key' not in st.session_state:
            st.session_state['webrtc_key'] = "origami-video-stable"
        webrtc_ctx = webrtc_streamer(
                key=st.session_state['webrtc_key'],
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTCConfiguration(
                    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                ),
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,  # enable async processing
        )

        # Create a placeholder for the WebRTC component
        # video_placeholder = st.empty()
        if webrtc_ctx.state.playing:
            st.success("Video is active")
                # if current_emotion not in ["neutral", "silence"]:
                #     st.info(f"Detected emotion: {current_emotion}")
                    # Add your response_generator.get_response(current_emotion) here
        else:
            st.info("Video is inactive. Press the START button above to begin.")

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
        # segment_duration = st.slider(
        #     "Recording segment duration (seconds)", 
        #     min_value=1, 
        #     max_value=5, 
        #     value=3,
        #     help="Each segment of audio will be this long before being analyzed"
        # )
        segment_duration = 5  # Set the default segment duration to 5 seconds

    # Check if video is active and sync audio recording
    video_is_active = webrtc_ctx.state.playing if webrtc_ctx else False

    # Create a new controller or reset the existing one
    controller = st.session_state['controller']

    # Start audio recording when video starts
    if video_is_active and not controller.is_recording:
        print("Video started - starting audio recording")
        
        controller.is_recording = True
        controller.audio_queue = queue.Queue()
        
        # Create new threads
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
        
        # Store threads and state
        st.session_state['recording_thread'] = recording_thread
        st.session_state['analysis_thread'] = analysis_thread

    # Stop audio recording when video stops
    elif not video_is_active and controller.is_recording:
        print("Video stopped - stopping audio recording")
        controller.is_recording = False
        time.sleep(0.5)  # Give threads a moment to clean up

    print(f"MAIN THREAD: video active is now {video_is_active}")
    # Display synchronized status
    if video_is_active:
        controller = st.session_state['controller']  
        queue_size = controller.results_queue.qsize()   
        
        print(f"MAIN THREAD: Results queue size: {controller.results_queue.qsize()}")
        if not controller.results_queue.empty():
            try:
                emotion_result = controller.results_queue.get_nowait()
                print(f"MAIN THREAD: Got emotion: {emotion_result['emotion']}")
                
                st.session_state['current_emotion'] = emotion_result['emotion']
                # Still add ALL to history for tracking
                st.session_state['emotion_history'].append(emotion_result)
                
            except queue.Empty:
                pass

            
            # Limit history size
            if len(st.session_state['emotion_history']) > 20:
                st.session_state['emotion_history'] = st.session_state['emotion_history'][-20:]
        
        # Show current emotion and immediate feedback
        current_emotion = st.session_state['current_emotion']
        last_emotion = st.session_state.get('last_emotion', 'neutral')
        print(f"*** MAIN THREAD: Current emotion is {current_emotion} and last emotion is {last_emotion}")

            # Create placeholders for conditional display
        emotion_placeholder = st.empty()
        guidance_placeholder = st.empty()

        response = st.session_state.get('emotion_response', None) # last emotion response to user

        # Give immediate feedback based on current emotion
        if current_emotion == "silence":
            # Clear the display for silence
            emotion_placeholder.empty()
            guidance_placeholder.empty()
            print("**** MAIN THREAD: Cleared display for silence")
        else:
            if current_emotion != last_emotion:
                print(f"**** MAIN THREAD: Emotion changed from {last_emotion} to {current_emotion}")
                response = response_generator.get_response(current_emotion)
                st.session_state['emotion_response'] = response
            else:
                print(f"**** MAIN THREAD: Emotion unchanged: {current_emotion}")
            with emotion_placeholder.container():
                st.subheader(f"Current Emotion: {current_emotion.capitalize()}")
            with guidance_placeholder.container():
                if response:
                    st.info(f"💡 {response}")

        st.session_state['last_emotion'] = current_emotion  

    # Display emotions history if available
    if st.button("Show/Hide Emotion History"):
        st.session_state['show_history'] = not st.session_state.get('show_history', False)
        # if st.session_state.get('show_history', False):
        #     st.success("📊 History: ON")
        # else:
        #     st.info("📊 History: OFF")

        chart_placeholder = st.empty()

        if st.session_state.get('show_history', False):
            with chart_placeholder.container():
                if st.session_state['emotion_history']:
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
                    emotions = [entry["emotion"].capitalize() for entry in st.session_state['emotion_history']]
                    
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
                    # Clean up
                    plt.close(fig)
                    plt.clf()
                else:
                    st.info("No emotion history available yet.")
        else:
            chart_placeholder.empty()

    time.sleep(5.0)  # Wait longer to allow time for emotion processing
    st.rerun()

if __name__ == "__main__":
    main()