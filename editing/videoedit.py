
from typing import Optional, Tuple, List, Union
from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    CompositeVideoClip,
    CompositeAudioClip,
    TextClip,
    concatenate_videoclips,
    vfx,
)
import os


class VideoEditor:
    """
    A reusable, generalized, and robust video editing module that supports text overlay,
    audio replacement or mixing, and basic video manipulation functionalities.
    """

    def __init__(self, video_path: str) -> None:
        """
        Initialize the VideoEditor with a video file.

        :param video_path: Path to the input video file.
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"The video file '{video_path}' does not exist.")
        try:
            self.video_clip: VideoFileClip = VideoFileClip(video_path)
        except Exception as e:
            raise Exception(f"Failed to load video file '{video_path}': {e}")

    def add_text_overlay(
        self,
        text: str,
        position: Tuple[Union[int, str], Union[int, str]] = ("center", "bottom"),
        fontsize: int = 24,
        color: str = "white",
        start_time: float = 0.0,
        duration: Optional[float] = None,
    ) -> None:
        """
        Add text overlay to the video.

        :param text: Text to display.
        :param position: Position of the text on the video.
        :param fontsize: Font size of the text.
        :param color: Color of the text.
        :param start_time: Start time in seconds for the text to appear.
        :param duration: Duration in seconds for which the text will appear.
        """
        try:
            txt_clip = TextClip(
                text,
                fontsize=fontsize,
                color=color,
                size=self.video_clip.size,
                method="caption",
            )
            txt_clip = txt_clip.set_position(position).set_start(start_time)
            if duration is not None:
                txt_clip = txt_clip.set_duration(duration)
            else:
                txt_clip = txt_clip.set_duration(self.video_clip.duration - start_time)
            self.video_clip = CompositeVideoClip([self.video_clip, txt_clip])
        except Exception as e:
            raise Exception(f"Failed to add text overlay: {e}")

    def replace_audio(self, audio_path: str) -> None:
        """
        Replace the video's audio with a new audio track.

        :param audio_path: Path to the new audio file.
        """
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"The audio file '{audio_path}' does not exist.")
        try:
            new_audio = AudioFileClip(audio_path)
            self.video_clip = self.video_clip.set_audio(new_audio)
        except Exception as e:
            raise Exception(f"Failed to replace audio: {e}")

    def mix_audio(self, audio_path: str) -> None:
        """
        Mix the video's audio with a new audio track.

        :param audio_path: Path to the new audio file.
        """
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"The audio file '{audio_path}' does not exist.")
        try:
            original_audio = self.video_clip.audio
            new_audio = AudioFileClip(audio_path)
            mixed_audio = CompositeAudioClip([original_audio, new_audio])
            self.video_clip = self.video_clip.set_audio(mixed_audio)
        except Exception as e:
            raise Exception(f"Failed to mix audio: {e}")

    def trim(self, start_time: float, end_time: float) -> None:
        """
        Trim the video between the start_time and end_time.

        :param start_time: Start time in seconds.
        :param end_time: End time in seconds.
        """
        try:
            self.video_clip = self.video_clip.subclip(start_time, end_time)
        except Exception as e:
            raise Exception(f"Failed to trim video: {e}")

    def rotate(self, angle: float) -> None:
        """
        Rotate the video by a specified angle.

        :param angle: Angle in degrees to rotate the video.
        """
        try:
            self.video_clip = self.video_clip.rotate(angle)
        except Exception as e:
            raise Exception(f"Failed to rotate video: {e}")

    def merge_videos(self, video_paths: List[str]) -> None:
        """
        Merge the current video with other videos.

        :param video_paths: List of paths to video files to be merged.
        """
        clips = [self.video_clip]
        try:
            for path in video_paths:
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"The video file '{path}' does not exist.")
                clip = VideoFileClip(path)
                clips.append(clip)
            self.video_clip = concatenate_videoclips(clips)
        except Exception as e:
            raise Exception(f"Failed to merge videos: {e}")

    def change_speed(self, factor: float) -> None:
        """
        Speed up or slow down the video.

        :param factor: Factor by which to change the speed (>1 speeds up, <1 slows down).
        """
        try:
            self.video_clip = self.video_clip.fx(vfx.speedx, factor)
        except Exception as e:
            raise Exception(f"Failed to change video speed: {e}")

    def save(self, output_path: str, codec: str = "libx264") -> None:
        """
        Save the edited video to a file.

        :param output_path: Path to save the output video file.
        :param codec: Codec to use for encoding the video.
        """
        try:
            self.video_clip.write_videofile(output_path, codec=codec)
        except Exception as e:
            raise Exception(f"Failed to save video to '{output_path}': {e}")

    def close(self) -> None:
        """
        Close the video clip and free resources.
        """
        try:
            self.video_clip.close()
        except Exception as e:
            raise Exception(f"Failed to close video clip: {e}")
        



from typing import Optional, Tuple, List, Union
from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    CompositeVideoClip,
    CompositeAudioClip,
    concatenate_videoclips,
    TextClip,
    vfx
)
import os


class VideoEditor1:
    """
    A reusable, generalized, and robust video editing module that supports
    text overlay, audio replacement or mixing, and basic video manipulation
    functionalities.
    """

    def __init__(self, video_path: str) -> None:
        """
        Initialize the VideoEditor with a video file.

        :param video_path: Path to the input video file.
        :raises FileNotFoundError: If the video file does not exist.
        :raises ValueError: If the video file cannot be loaded.
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file '{video_path}' does not exist.")
        try:
            self.video_clip: VideoFileClip = VideoFileClip(video_path)
        except Exception as e:
            raise ValueError(f"Failed to load video file '{video_path}': {e}")

    
    def add_text_overlay(
        self,
        text: str,
        position: Tuple[Union[int, str], Union[int, str]] = ("center", "bottom"),
        fontsize: int = 24,
        color: str = "white",
        start_time: float = 0.0,
        duration: Optional[float] = None
    ) -> None:
        """
        Add a text overlay to the video.
    
        :param text: Text to overlay on the video.
        :param position: Position of the text on the video frame.
        :param fontsize: Font size of the text.
        :param color: Color of the text.
        :param start_time: Time (in seconds) when the text appears.
        :param duration: Duration (in seconds) the text stays on screen.
        :raises ValueError: If text overlay fails.
        """
        try:
            txt_clip = TextClip(
                txt=text,   # Corrected parameter name
                fontsize=fontsize,
                color=color,
                size=self.video_clip.size,
                method='caption'
            )
            txt_clip = txt_clip.set_position(position).set_start(start_time)
            if duration is not None:
                txt_clip = txt_clip.set_duration(duration)
            else:
                txt_clip = txt_clip.set_duration(self.video_clip.duration - start_time)
            self.video_clip = CompositeVideoClip([self.video_clip, txt_clip])
        except Exception as e:
            raise ValueError(f"Failed to add text overlay: {e}")

    def replace_audio(self, audio_path: str) -> None:
        """
        Replace the current audio track with a new audio file.

        :param audio_path: Path to the new audio file.
        :raises FileNotFoundError: If the audio file does not exist.
        :raises ValueError: If audio replacement fails.
        """
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file '{audio_path}' does not exist.")
        try:
            new_audio: AudioFileClip = AudioFileClip(audio_path)
            self.video_clip = self.video_clip.set_audio(new_audio)
        except Exception as e:
            raise ValueError(f"Failed to replace audio: {e}")

    def mix_audio(self, audio_path: str, volume: float = 1.0) -> None:
        """
        Mix the current audio with a new audio file.

        :param audio_path: Path to the new audio file.
        :param volume: Volume level for the new audio track.
        :raises FileNotFoundError: If the audio file does not exist.
        :raises ValueError: If audio mixing fails.
        """
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file '{audio_path}' does not exist.")
        try:
            original_audio = self.video_clip.audio
            new_audio: AudioFileClip = AudioFileClip(audio_path).volumex(volume)
            mixed_audio = CompositeAudioClip([original_audio, new_audio])
            self.video_clip = self.video_clip.set_audio(mixed_audio)
        except Exception as e:
            raise ValueError(f"Failed to mix audio: {e}")

    def trim(self, start_time: float, end_time: float) -> None:
        """
        Trim the video between specified start and end times.

        :param start_time: Start time in seconds.
        :param end_time: End time in seconds.
        :raises ValueError: If trimming fails.
        """
        try:
            self.video_clip = self.video_clip.subclip(start_time, end_time)
        except Exception as e:
            raise ValueError(f"Failed to trim video: {e}")

    def rotate(self, angle: float) -> None:
        """
        Rotate the video by a given angle.

        :param angle: Angle in degrees to rotate the video.
        :raises ValueError: If rotation fails.
        """
        try:
            self.video_clip = self.video_clip.rotate(angle)
        except Exception as e:
            raise ValueError(f"Failed to rotate video: {e}")

    def merge_videos(self, video_paths: List[str]) -> None:
        """
        Merge the current video with additional videos.

        :param video_paths: List of paths to video files to merge.
        :raises FileNotFoundError: If any of the video files do not exist.
        :raises ValueError: If merging fails.
        """
        try:
            clips: List[VideoFileClip] = [self.video_clip]
            for path in video_paths:
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"Video file '{path}' does not exist.")
                clip = VideoFileClip(path)
                clips.append(clip)
            self.video_clip = concatenate_videoclips(clips, method="compose")
        except Exception as e:
            raise ValueError(f"Failed to merge videos: {e}")

    def change_speed(self, factor: float) -> None:
        """
        Change the playback speed of the video.

        :param factor: Speed factor (>1.0 speeds up, <1.0 slows down).
        :raises ValueError: If speed change fails.
        """
        try:
            self.video_clip = self.video_clip.fx(vfx.speedx, factor)
        except Exception as e:
            raise ValueError(f"Failed to change video speed: {e}")

    def save(self, output_path: str, codec: str = "libx264") -> None:
        """
        Save the edited video to a file.

        :param output_path: Path to save the output video file.
        :param codec: Video codec to use for encoding.
        :raises ValueError: If saving fails.
        """
        try:
            self.video_clip.write_videofile(output_path, codec=codec, audio_codec="aac")
        except Exception as e:
            raise ValueError(f"Failed to save video to '{output_path}': {e}")

    def close(self) -> None:
        """
        Close the video clip and release resources.

        :raises ValueError: If closing fails.
        """
        try:
            self.video_clip.close()
        except Exception as e:
            raise ValueError(f"Failed to close video clip: {e}")


def main():
    editor = VideoEditor1(r"C:\Users\heman\Desktop\Coding\videos\8c399HPb01s.webm")
    editor.add_text_overlay(
        text="I LOVE YOU ...",
        position=('center', 'bottom'),
        fontsize=50,
        color='white',
        start_time=5,
        duration=10
    )
    # editor.replace_audio('new_audio.mp3')
    editor.trim(start_time=0, end_time=60)
    # editor.rotate(angle=90)
    editor.change_speed(factor=1)
    editor.save('output_video.mp4')
    editor.close()
if __name__ == "__main__":
    main()