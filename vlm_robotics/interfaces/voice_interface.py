"""Voice recognition interface using speech_recognition and microphone input."""

import threading
import speech_recognition as sr
from typing import Callable, Optional, Tuple


class VoiceInterface:
    """Manages audio capture from microphone and speech-to-text transcription."""

    def __init__(self, energy_threshold: int = 300, dynamic_energy: bool = True):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.dynamic_energy_threshold = dynamic_energy
        self.recognizer.pause_threshold = 0.8
        self._is_listening = False
        self._listen_thread: Optional[threading.Thread] = None

    def listen_and_transcribe(
        self,
        on_status: Optional[Callable[[str], None]] = None,
        on_result: Optional[Callable[[bool, str], None]] = None,
        timeout: float = 7.0,
        phrase_time_limit: float = 9.0,
    ):
        """
        Record audio from default microphone and transcribe in a background thread.
        on_status(status_str) is invoked with progress updates.
        on_result(success, transcribed_text_or_error) is invoked upon completion.
        """
        if self._is_listening:
            if on_status:
                on_status("Already listening...")
            return

        def _worker():
            self._is_listening = True
            try:
                if on_status:
                    on_status("Adjusting for ambient noise...")

                with sr.Microphone() as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.6)

                    if on_status:
                        on_status("Listening... Speak your command now!")

                    audio = self.recognizer.listen(
                        source,
                        timeout=timeout,
                        phrase_time_limit=phrase_time_limit,
                    )

                if on_status:
                    on_status("Processing speech to text...")

                # Transcribe using Google Web Speech API (free, reliable, zero-config)
                text = self.recognizer.recognize_google(audio)
                clean_text = text.strip()

                if on_status:
                    on_status(f"Heard: '{clean_text}'")

                if on_result:
                    on_result(True, clean_text)

            except sr.WaitTimeoutError:
                err = "No speech detected (timeout)."
                if on_status:
                    on_status(err)
                if on_result:
                    on_result(False, err)

            except sr.UnknownValueError:
                err = "Could not understand audio. Please speak clearly."
                if on_status:
                    on_status(err)
                if on_result:
                    on_result(False, err)

            except sr.RequestError as e:
                err = f"Speech recognition service error: {e}"
                if on_status:
                    on_status(err)
                if on_result:
                    on_result(False, err)

            except Exception as e:
                err = f"Microphone error: {e}"
                if on_status:
                    on_status(err)
                if on_result:
                    on_result(False, err)

            finally:
                self._is_listening = False

        self._listen_thread = threading.Thread(target=_worker, daemon=True)
        self._listen_thread.start()

    def is_listening(self) -> bool:
        return self._is_listening

