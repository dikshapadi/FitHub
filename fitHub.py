import random
import json
import wave
import os
import pyaudio
import speech_recognition as sr
import tkinter as tk
from tkinter import scrolledtext
from playsound import playsound
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

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

def record_audio(output_file):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print(f"Recording audio to {output_file}. Press Ctrl+C to stop recording...")
    frames = []

    try:
        while True:
            data = stream.read(CHUNK)
            frames.append(data)
    except KeyboardInterrupt:
        print("Recording stopped.")

    stream.stop_stream()
    stream.close()

    wf = wave.open(output_file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def audio_to_text(audio_file):
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return text

class ConversationalUI:
    def __init__(self, root):
        self.is_recording = False  # Flag to indicate if recording is in progress
        self.recording_thread = None
        self.root = root
        self.root.title("FitHub")
        self.user_name = 'User'

        

        self.chat_frame = tk.Frame(self.root, bg="#2c3e50")
        self.chat_frame.pack(fill=tk.BOTH, expand=True)

        self.output_text = scrolledtext.ScrolledText(self.chat_frame, wrap=tk.WORD, width=60, height=15, bg="#34495e", fg="#ecf0f1")
        self.output_text.pack(fill=tk.BOTH, expand=True)

        self.input_text = tk.Entry(self.chat_frame, width=60, bg="#34495e", fg="#ecf0f1", bd=0)
        self.input_text.pack(fill=tk.BOTH, expand=True)

        
        self.record_button = tk.Button(self.chat_frame, text="Send text" ,command=self.send_text, bg="#2c3e50",fg="#ecf0f1")
        self.record_button.pack()
        self.record_button.pack(pady=5)

        self.record_audio_button = tk.Button(self.chat_frame, text="Record & Send Audio", command=self.record_and_send, bg="#2c3e50", fg="#ecf0f1")
        self.record_audio_button.pack()
        self.record_audio_button.pack(pady=5)

        #self.end_recording_button = tk.Button(self.chat_frame, text="End Recording", command=self.end_recording, bg="#e74c3c", fg="#ecf0f1")
        #self.end_recording_button.pack(pady=5)  

        self.quit_button = tk.Button(self.chat_frame, text="Quit", command=self.root.quit, bg="#2c3e50", fg="#ecf0f1")
        self.quit_button.pack(pady =5)
        

        self.init_bot()
    

    def init_bot(self):
        pass  # Initialize the chatbot components here

    def display_response(self, speaker, response):
        self.output_text.insert(tk.END, f"{speaker}: {response}\n")
        self.output_text.see(tk.END)

    def send_text(self):
        user_input = self.input_text.get()
        self.display_response(self.user_name, user_input)

        # Process user_input and get a response from the bot
        bot_response = self.get_bot_response(user_input)
        self.display_response("FitHub", bot_response)

    def record_and_send(self):
        audio_input_file = "user_audio.wav"
        record_audio(audio_input_file)
        playsound(audio_input_file)

        # Convert the audio to text using speech recognition
        text_input = audio_to_text(audio_input_file)
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
                    return random.choice(intent['responses'])
        else:
            return "I do not understand..."

if __name__ == "__main__":
    root = tk.Tk()
    root.configure(bg="#2c3e50")
    app = ConversationalUI(root)
    root.mainloop()