import speech_recognition as sr
import os

sr.AudioFile.flac_command = '/opt/homebrew/bin/flac' 
r = sr.Recognizer()

def transcribe_and_save():
    print("Listening... Say 'stop recording' to end.")
    
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        
        while True:
            try:
                audio = r.listen(source, timeout=None)
                text = r.recognize_google(audio).lower()
                print(f"You said: {text}")

                if "stop recording" in text:
                    print("Stopping recording.")
                    break  
                    with open("notes.txt", "a") as f:
                    f.write(text + "\n")
                print("Note saved.")

            except sr.WaitTimeoutError:
                continue  
            except sr.UnknownValueError:
                print("Could not understand audio, continuing...")
                continue
            except sr.RequestError as e:
                print(f"Error; {e}")
                break

def save_note(text):
    notes_file = "notes.txt"
    with open(notes_file, "a") as f:
        f.write(text + "\n")
    print(f"Note saved to {notes_file}")

if __name__ == "__main__":
    transcribe_and_save()
