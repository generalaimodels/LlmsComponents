import pyttsx3
from typing import Optional, Union
from functools import lru_cache


class TextToSpeechAPI:
    def __init__(self):
        self.engine = pyttsx3.init()

    @lru_cache(maxsize=128)
    def get_voices(self):
        """Cache the available voices to improve performance."""
        return self.engine.getProperty('voices')

    def set_voice(self, voice_id: Optional[str] = None) -> None:
        """
        Set the voice for the text-to-speech engine.

        :param voice_id: The ID of the voice to use. If None, use the default voice.
        """
        voices = self.get_voices()
        if voice_id:
            matching_voices = [v for v in voices if v.id == voice_id]
            if matching_voices:
                self.engine.setProperty('voice', matching_voices[0].id)
            else:
                raise ValueError(f"Voice ID '{voice_id}' not found.")
        elif voices:
            self.engine.setProperty('voice', voices[0].id)

    def set_rate(self, rate: int) -> None:
        """
        Set the speech rate.

        :param rate: The speed of speech (words per minute).
        """
        self.engine.setProperty('rate', rate)

    def set_volume(self, volume: float) -> None:
        """
        Set the speech volume.

        :param volume: The volume of speech (0.0 to 1.0).
        """
        if 0.0 <= volume <= 1.0:
            self.engine.setProperty('volume', volume)
        else:
            raise ValueError("Volume must be between 0.0 and 1.0")

    def speak(
        self,
        text: str,
        rate: Optional[int] = None,
        volume: Optional[float] = None,
        voice_id: Optional[str] = None
    ) -> None:
        """
        Convert text to speech and play it.

        :param text: The text to be converted to speech.
        :param rate: The speed of speech (words per minute).
        :param volume: The volume of speech (0.0 to 1.0).
        :param voice_id: The ID of the voice to use.
        """
        if rate is not None:
            self.set_rate(rate)
        if volume is not None:
            self.set_volume(volume)
        if voice_id is not None:
            self.set_voice(voice_id)

        self.engine.say(text)
        self.engine.runAndWait()

    def save_to_file(
        self,
        text: str,
        filename: str,
        rate: Optional[int] = None,
        volume: Optional[float] = None,
        voice_id: Optional[str] = None
    ) -> None:
        """
        Convert text to speech and save it to a file.

        :param text: The text to be converted to speech.
        :param filename: The name of the file to save the audio to.
        :param rate: The speed of speech (words per minute).
        :param volume: The volume of speech (0.0 to 1.0).
        :param voice_id: The ID of the voice to use.
        """
        if rate is not None:
            self.set_rate(rate)
        if volume is not None:
            self.set_volume(volume)
        if voice_id is not None:
            self.set_voice(voice_id)

        self.engine.save_to_file(text, filename)
        self.engine.runAndWait()

    def list_available_voices(self) -> list:
        """Return a list of available voice IDs."""
        return [voice.id for voice in self.get_voices()]

