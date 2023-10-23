import random
import json
import threading
import wave
import os
from winsound import PlaySound
import pyaudio
import speech_recognition as sr
import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import QTimer
import pyttsx3
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize  # Added this import

# Initialize PyAudio for audio input
p = pyaudio.PyAudio()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Initialize the speech recognizer
recognizer = sr.Recognizer()

class ConversationalUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.is_recording = False
        self.is_recording_completed = False
        self.audio_input_file = ""  # Initialize as an empty string
        self.user_name = 'User'

        self.record_timer = QTimer(self)
        self.record_timer.timeout.connect(self.record_audio)

        self.engine = pyttsx3.init()  # Initialize pyttsx3

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("FitHub")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: #2c3e50;")

        self.output_text = QtWidgets.QTextEdit(self)
        self.output_text.setGeometry(20, 20, 760, 400)
        self.output_text.setStyleSheet("background-color: #34495e; color: #ecf0f1;")

        self.input_text = QtWidgets.QLineEdit(self)
        self.input_text.setGeometry(20, 440, 600, 40)
        self.input_text.setStyleSheet("background-color: #34495e; color: #ecf0f1;")

        self.send_text_button = QtWidgets.QPushButton("Send Text", self)
        self.send_text_button.setGeometry(640, 440, 140, 40)
        self.send_text_button.setStyleSheet("background-color: #2c3e50; color: #ecf0f1;")
        self.send_text_button.clicked.connect(self.send_text)

        self.start_recording_button = QtWidgets.QPushButton("Start Recording", self)
        self.start_recording_button.setGeometry(20, 500, 140, 40)
        self.start_recording_button.setStyleSheet("background-color: #2c3e50; color: #ecf0f1;")
        self.start_recording_button.clicked.connect(self.start_recording)

        self.stop_recording_button = QtWidgets.QPushButton("Stop Recording", self)
        self.stop_recording_button.setGeometry(180, 500, 140, 40)
        self.stop_recording_button.setStyleSheet("background-color: #e74c3c; color: #ecf0f1;")
        self.stop_recording_button.clicked.connect(self.stop_recording)
        #self.stop_recording_button.hide()

        self.quit_button = QtWidgets.QPushButton("Quit", self)
        self.quit_button.setGeometry(640, 500, 140, 40)
        self.quit_button.setStyleSheet("background-color: #2c3e50; color: #ecf0f1;")
        self.quit_button.clicked.connect(QtWidgets.QApplication.quit)

        self.show()

    def display_response(self, speaker, response):
        self.output_text.append(f"{speaker}: {response}")

    def audio_to_text(self, audio_file):
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            return text

    def record_audio(self):
        while self.is_recording:
            data = self.stream.read(1024)
            self.frames.append(data)
            QtWidgets.QApplication.processEvents()  # Process events to keep the UI responsive

    def start_recording(self):
        self.is_recording = True
        self.is_recording_completed = False
        self.start_recording_button.setEnabled(False)
        self.stop_recording_button.setEnabled(True)

        self.frames = []  # Store audio frames
        self.audio_input_file = "user_audio.wav"  # Set the audio input file path

        # Initialize the PyAudio stream for audio recording
        self.stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024
        )

        # Start the record_timer to call record_audio periodically
        interval_msec = int(1000 / 16000)  # Convert float to integer
        self.record_timer.start(interval_msec)  # Adjust the interval as needed

    def stop_recording(self):
        print("Stop Recording button clicked")
        if self.is_recording:
            self.is_recording = False

            # Stop the record_timer
            self.record_timer.stop()

            # Stop the PyAudio stream
            self.stream.stop_stream()
            self.stream.close()

            # Save recorded audio to file
            wf = wave.open(self.audio_input_file, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)
            wf.writeframes(b''.join(self.frames))
            wf.close()

            # Reset the audio frames
            self.frames = []

            # Convert the recorded audio to text and process the response
            text_input = self.audio_to_text(self.audio_input_file)
            self.display_response(self.user_name, text_input)

            # Process text_input and get a response from the bot
            bot_response = self.get_bot_response(text_input)
            self.display_response("FitHub", bot_response)

            # Speak the bot's response using pyttsx3
            self.speak(bot_response)

            # Toggle button visibility after stopping recording
            if not self.is_recording_completed:
                self.stop_recording_button.setEnabled(False)
                self.start_recording_button.setEnabled(True)



    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def send_text(self):
        user_input = self.input_text.text()
        self.display_response(self.user_name, user_input)

        # Process user_input and get a response from the bot
        bot_response = self.get_bot_response(user_input)
        self.display_response("FitHub", bot_response)

       

    def record_and_send(self):
        audio_input_file = "user_audio.wav"
        self.record_audio()  # Start recording audio frames
        PlaySound(audio_input_file)

        # Stop recording audio
        self.is_recording = False
        self.stream.stop_stream()
        self.stream.close()

        # Save recorded audio to file
        wf = wave.open(self.audio_input_file, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        # Convert the audio to text using speech recognition
        text_input = self.audio_to_text(self.audio_input_file)
        self.display_response(self.user_name, text_input)

        # Process text_input and get a response from the bot
        bot_response = self.get_bot_response(text_input)
        self.display_response("FitHub", bot_response)


    def get_bot_response(self, user_input):
        sentence = tokenize(user_input)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device).to(dtype=torch.float32)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    response = random.choice(intent['responses'])
                    return response
        else:
            return "I do not understand..."


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = ConversationalUI()
    sys.exit(app.exec_())
