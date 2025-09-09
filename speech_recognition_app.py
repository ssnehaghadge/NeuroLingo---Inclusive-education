import speech_recognition as sr
import os

sr.AudioFile.flac_command = '/opt/homebrew/bin/flac' 
# Create a recognizer instance
r = sr.Recognizer()

def transcribe_and_save():
    print("Listening... Say 'stop recording' to end.")
    
    with sr.Microphone() as source:
        # Optional: Adjust for ambient noise for better accuracy
        r.adjust_for_ambient_noise(source)
        
        while True:
            try:
                # Listen for a phrase
                audio = r.listen(source, timeout=None)
                text = r.recognize_google(audio).lower()
                print(f"You said: {text}")

                if "stop recording" in text:
                    print("Stopping recording.")
                    break  # Exit the loop

                # Save the transcribed text
                with open("notes.txt", "a") as f:
                    f.write(text + "\n")
                print("Note saved.")

            except sr.WaitTimeoutError:
                continue  # Continue listening if no speech is detected
            except sr.UnknownValueError:
                print("Could not understand audio, continuing...")
                continue
            except sr.RequestError as e:
                print(f"Error; {e}")
                break

def save_note(text):
    """
    Appends the transcribed text to a file named notes.txt.
    """
    notes_file = "notes.txt"
    with open(notes_file, "a") as f:
        # Add the transcribed text followed by a new line
        f.write(text + "\n")
    print(f"Note saved to {notes_file}")

if __name__ == "__main__":
    transcribe_and_save()
