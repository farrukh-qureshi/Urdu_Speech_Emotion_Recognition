import streamlit as st
import torch
import torchaudio
import os
import numpy as np
from models import UrduClinicalEmotionTransformer
from audio_preprocessing import AudioPreprocessor
import tempfile
from audio_recorder_streamlit import audio_recorder
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import queue
import threading
from typing import List, NamedTuple
import logging
import time

# Constants
SAMPLE_RATE = 16000
EMOTIONS = ['Angry', 'Happy', 'Neutral', 'Sad']
WINDOW_SIZE = 16000  # 1 second of audio at 16kHz

# Create a class for audio frames
class AudioFrame(NamedTuple):
    data: np.ndarray
    timestamp: int

# Global variables for real-time processing
audio_buffer = queue.Queue()
result_buffer = queue.Queue()
lock = threading.Lock()

def load_model():
    model = UrduClinicalEmotionTransformer(num_emotions=len(EMOTIONS))
    checkpoint = torch.load(
        'checkpoints/best_model.pt', 
        map_location=torch.device('cpu'),
        weights_only=True
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def process_audio(audio_file, preprocessor):
    # Load and preprocess audio with backend specified
    waveform, sr = torchaudio.load(audio_file, backend="soundfile")
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Preprocess using the same preprocessing pipeline as training
    features = preprocessor.preprocess(waveform, augment=False)
    return features.unsqueeze(0)  # Add batch dimension

def process_audio_bytes(audio_bytes, preprocessor):
    # Save bytes to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name
    
    try:
        # Process the temporary file
        features = process_audio(tmp_path, preprocessor)
    finally:
        # Clean up
        os.unlink(tmp_path)
    
    return features

def audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
    """Callback for processing audio frames in real-time"""
    audio_data = frame.to_ndarray()
    audio_buffer.put(AudioFrame(data=audio_data, timestamp=frame.time))
    return frame

def process_audio_buffer(model, preprocessor):
    """Process audio buffer in real-time"""
    while True:
        if not audio_buffer.empty():
            # Get audio frame
            audio_frame = audio_buffer.get()
            
            try:
                # Convert to tensor
                waveform = torch.FloatTensor(audio_frame.data)
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Ensure correct sample rate
                if SAMPLE_RATE != 16000:
                    waveform = torchaudio.functional.resample(waveform, SAMPLE_RATE, 16000)
                
                # Preprocess
                features = preprocessor.preprocess(waveform, augment=False)
                features = features.unsqueeze(0)
                
                # Make prediction
                with torch.no_grad():
                    outputs = model(features)
                    probabilities = torch.softmax(outputs, dim=1)
                    prediction = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][prediction].item() * 100
                
                # Put results in buffer
                result_buffer.put({
                    'emotion': EMOTIONS[prediction],
                    'confidence': confidence,
                    'probabilities': probabilities[0].tolist()
                })
            
            except Exception as e:
                logging.error(f"Error processing audio frame: {str(e)}")

def main():
    st.title("Urdu Clinical Emotion Classification")
    
    # Initialize model and preprocessor
    try:
        model = load_model()
        preprocessor = AudioPreprocessor()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["Upload Audio", "Record Audio", "Real-time Detection"])

    with tab1:
        st.write("Upload an audio file to classify the emotion")
        audio_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3'])

        if audio_file is not None:
            # Create a temporary file to save the uploaded audio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_file.getvalue())
                tmp_path = tmp_file.name

            try:
                # Display audio player
                st.audio(audio_file)

                # Process when button is clicked
                if st.button("Classify Uploaded Audio"):
                    with st.spinner("Processing audio..."):
                        features = process_audio(tmp_path, preprocessor)
                        
                        # Make prediction
                        with torch.no_grad():
                            outputs = model(features)
                            probabilities = torch.softmax(outputs, dim=1)
                            prediction = torch.argmax(probabilities, dim=1).item()
                            confidence = probabilities[0][prediction].item() * 100

                        # Display results
                        st.success(f"Detected Emotion: {EMOTIONS[prediction]}")
                        st.info(f"Confidence: {confidence:.2f}%")

                        # Display probability distribution
                        st.write("Probability Distribution:")
                        probs = probabilities[0].tolist()
                        for emotion, prob in zip(EMOTIONS, probs):
                            st.progress(prob)
                            st.write(f"{emotion}: {prob*100:.2f}%")

            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
            finally:
                os.unlink(tmp_path)

    with tab2:
        st.write("Record your voice to classify the emotion")
        
        # Add a more visible recording interface
        st.write("👇 Click the microphone button below to start/stop recording")
        
        # Add audio recorder with more visible styling
        audio_bytes = audio_recorder(
            text="🎤 Click to Record",
            recording_color="#FF0000",  # Red color while recording
            neutral_color="#1976D2",    # Blue color when not recording
            icon_name="microphone",
            icon_size="6x",             # Larger icon
            key="audio_recorder"        # Unique key for the component
        )

        # Add recording instructions
        if not audio_bytes:
            st.info("Instructions:\n"
                   "1. Click the microphone button to start recording\n"
                   "2. Speak your message\n"
                   "3. Click again to stop recording")

        if audio_bytes:
            st.audio(audio_bytes)
            
            if st.button("Classify Recorded Audio"):
                with st.spinner("Processing audio..."):
                    try:
                        features = process_audio_bytes(audio_bytes, preprocessor)
                        
                        # Make prediction
                        with torch.no_grad():
                            outputs = model(features)
                            probabilities = torch.softmax(outputs, dim=1)
                            prediction = torch.argmax(probabilities, dim=1).item()
                            confidence = probabilities[0][prediction].item() * 100

                        # Display results
                        st.success(f"Detected Emotion: {EMOTIONS[prediction]}")
                        st.info(f"Confidence: {confidence:.2f}%")

                        # Display probability distribution
                        st.write("Probability Distribution:")
                        probs = probabilities[0].tolist()
                        for emotion, prob in zip(EMOTIONS, probs):
                            st.progress(prob)
                            st.write(f"{emotion}: {prob*100:.2f}%")

                    except Exception as e:
                        st.error(f"Error processing audio: {str(e)}")

    with tab3:
        st.write("Real-time Emotion Detection")
        st.info("Click 'START' to begin real-time emotion detection. Speak continuously and see emotions detected in real-time.")
        
        # Create placeholder for real-time results
        result_placeholder = st.empty()
        
        # WebRTC Configuration
        rtc_configuration = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        # Create WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="real-time-emotion",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=WINDOW_SIZE,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": False, "audio": True},
            audio_frame_callback=audio_frame_callback,
        )
        
        # Start processing thread when streaming begins
        if webrtc_ctx.state.playing:
            process_thread = threading.Thread(
                target=process_audio_buffer,
                args=(model, preprocessor),
                daemon=True
            )
            process_thread.start()
            
            # Display real-time results
            while webrtc_ctx.state.playing:
                if not result_buffer.empty():
                    result = result_buffer.get()
                    
                    # Update display
                    with result_placeholder.container():
                        st.success(f"Detected Emotion: {result['emotion']}")
                        st.info(f"Confidence: {result['confidence']:.2f}%")
                        
                        st.write("Probability Distribution:")
                        for emotion, prob in zip(EMOTIONS, result['probabilities']):
                            st.progress(prob)
                            st.write(f"{emotion}: {prob*100:.2f}%")
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)

if __name__ == "__main__":
    main() 