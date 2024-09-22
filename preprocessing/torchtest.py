import torch
import torchaudio
from torchaudio import (
    get_audio_backend,
    info,
    list_audio_backends,
    load,
    save,
    set_audio_backend,
)
from torchaudio.transforms import (
    AddNoise, AmplitudeToDB, ComputeDeltas, Convolve, Deemphasis,
    Fade, FFTConvolve, FrequencyMasking, GriffinLim, InverseMelScale,
    InverseSpectrogram, LFCC, Loudness, MelScale, MelSpectrogram, MFCC,
    MuLawDecoding, MuLawEncoding, PitchShift, Preemphasis, Resample,
    RNNTLoss, SlidingWindowCmn, SpecAugment, SpectralCentroid,
    Spectrogram, Speed, SpeedPerturbation, TimeMasking, TimeStretch,
    Vad, Vol,
)
from typing import List, Optional, Dict, Union, Callable, Tuple
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AudioProcessingError(Exception):
    """Custom exception class for audio processing errors."""
    pass


class AudioProcessor:
    """
    A robust and scalable advanced API wrapper for torchaudio.
    Simplifies audio processing, transformations, and manipulation.
    """

    def __init__(self, backend: Optional[str] = None) -> None:
        """
        Initialize the AudioProcessor.

        Args:
            backend (Optional[str]): The audio backend to use.
                If None, uses the current backend.
        """
        if backend:
            try:
                set_audio_backend(backend)
                logger.info(f"Audio backend set to {backend}")
            except Exception as e:
                logger.error(f"Failed to set audio backend: {e}")
                raise AudioProcessingError(f"Failed to set audio backend: {e}") from e
        else:
            self.backend = get_audio_backend()
            logger.info(f"Using audio backend: {self.backend}")

    def load_audio(self, filepath: str) -> Tuple[torch.Tensor, int]:
        """
        Load an audio file.

        Args:
            filepath (str): Path to the audio file.

        Returns:
            Tuple[torch.Tensor, int]: A tuple containing the audio waveform tensor and the sample rate.
        """
        try:
            waveform, sample_rate = load(filepath)
            logger.info(f"Loaded audio file {filepath} with sample rate {sample_rate}")
            return waveform, sample_rate
        except Exception as e:
            logger.error(f"Failed to load audio file {filepath}: {e}")
            raise AudioProcessingError(f"Failed to load audio file {filepath}: {e}") from e

    def save_audio(
        self,
        filepath: str,
        waveform: torch.Tensor,
        sample_rate: int,
        encoding: Optional[str] = None,
        bits_per_sample: Optional[int] = None,
    ) -> None:
        """
        Save an audio file.

        Args:
            filepath (str): Path to save the audio file.
            waveform (torch.Tensor): Audio waveform tensor.
            sample_rate (int): Sample rate of the audio.
            encoding (Optional[str]): Audio encoding format.
            bits_per_sample (Optional[int]): Bits per sample.

        Raises:
            AudioProcessingError: If saving the audio fails.
        """
        try:
            save(
                filepath,
                waveform,
                sample_rate,
                encoding=encoding,
                bits_per_sample=bits_per_sample
            )
            logger.info(f"Saved audio file {filepath} at sample rate {sample_rate}")
        except Exception as e:
            logger.error(f"Failed to save audio file {filepath}: {e}")
            raise AudioProcessingError(f"Failed to save audio file {filepath}: {e}") from e

    def get_metadata(self, filepath: str) -> torchaudio.AudioMetaData:
        """
        Retrieve metadata of an audio file.

        Args:
            filepath (str): Path to the audio file.

        Returns:
            AudioMetaData: Metadata of the audio file.

        Raises:
            AudioProcessingError: If retrieving metadata fails.
        """
        try:
            metadata = info(filepath)
            logger.info(f"Retrieved metadata for {filepath}")
            return metadata
        except Exception as e:
            logger.error(f"Failed to retrieve metadata for {filepath}: {e}")
            raise AudioProcessingError(f"Failed to retrieve metadata for {filepath}: {e}") from e

    def apply_transforms(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        transforms: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None
    ) -> torch.Tensor:
        """
        Apply a sequence of transformations to the audio waveform.

        Args:
            waveform (torch.Tensor): Audio waveform tensor.
            sample_rate (int): Sample rate of the audio.
            transforms (Optional[List[Callable[[torch.Tensor], torch.Tensor]]]):
                A list of transformation functions to apply.

        Returns:
            torch.Tensor: The transformed audio waveform.

        Raises:
            AudioProcessingError: If applying transformations fails.
        """
        if transforms is None:
            logger.warning("No transformations provided; returning original waveform.")
            return waveform
        try:
            transformed_waveform = waveform
            for i, transform in enumerate(transforms):
                transformed_waveform = transform(transformed_waveform)
                logger.debug(f"Applied transform {i + 1}/{len(transforms)}: {transform}")
            logger.info("Successfully applied all transformations.")
            return transformed_waveform
        except Exception as e:
            logger.error(f"Failed to apply transformations: {e}")
            raise AudioProcessingError(f"Failed to apply transformations: {e}") from e

    def resample_audio(
        self,
        waveform: torch.Tensor,
        orig_sample_rate: int,
        new_sample_rate: int
    ) -> torch.Tensor:
        """
        Resample the audio waveform to a new sample rate.

        Args:
            waveform (torch.Tensor): Audio waveform tensor.
            orig_sample_rate (int): Original sample rate.
            new_sample_rate (int): New sample rate to resample to.

        Returns:
            torch.Tensor: The resampled audio waveform.

        Raises:
            AudioProcessingError: If resampling fails.
        """
        try:
            resampler = Resample(orig_freq=orig_sample_rate, new_freq=new_sample_rate)
            resampled_waveform = resampler(waveform)
            logger.info(f"Resampled audio from {orig_sample_rate}Hz to {new_sample_rate}Hz")
            return resampled_waveform
        except Exception as e:
            logger.error(f"Failed to resample audio: {e}")
            raise AudioProcessingError(f"Failed to resample audio: {e}") from e

    def compute_spectrogram(
        self,
        waveform: torch.Tensor,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        power: Optional[float] = 2.0,
    ) -> torch.Tensor:
        """
        Compute the spectrogram of the audio waveform.

        Args:
            waveform (torch.Tensor): Audio waveform tensor.
            n_fft (int): Size of FFT, creates n_fft // 2 + 1 bins.
            win_length (Optional[int]): Window size. If None, defaults to n_fft.
            hop_length (Optional[int]): Length of hop between STFT windows. If None, defaults to win_length // 2.
            power (Optional[float]): Exponent for the magnitude spectrogram.

        Returns:
            torch.Tensor: Spectrogram tensor.

        Raises:
            AudioProcessingError: If spectrogram computation fails.
        """
        try:
            spectrogram = Spectrogram(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                power=power,
            )
            spec = spectrogram(waveform)
            logger.info("Computed spectrogram.")
            return spec
        except Exception as e:
            logger.error(f"Failed to compute spectrogram: {e}")
            raise AudioProcessingError(f"Failed to compute spectrogram: {e}") from e

    def add_noise(
        self,
        waveform: torch.Tensor,
        noise_waveform: torch.Tensor,
        snr: float
    ) -> torch.Tensor:
        """
        Add noise to the audio waveform at a specified Signal-to-Noise Ratio (SNR).

        Args:
            waveform (torch.Tensor): Original audio waveform.
            noise_waveform (torch.Tensor): Noise audio waveform.
            snr (float): Desired Signal-to-Noise Ratio in decibels.

        Returns:
            torch.Tensor: The noisy audio waveform.

        Raises:
            AudioProcessingError: If adding noise fails.
        """
        try:
            if noise_waveform.size(-1) < waveform.size(-1):
                repeats = (waveform.size(-1) + noise_waveform.size(-1) - 1) // noise_waveform.size(-1)
                noise_waveform = noise_waveform.repeat(1, repeats)[:, :waveform.size(-1)]
            else:
                noise_waveform = noise_waveform[:, :waveform.size(-1)]

            signal_power = waveform.pow(2).mean()
            noise_power = noise_waveform.pow(2).mean()
            snr_linear = 10 ** (snr / 10)
            scaling_factor = torch.sqrt(signal_power / (snr_linear * noise_power))

            noisy_waveform = waveform + scaling_factor * noise_waveform
            logger.info(f"Added noise at {snr} dB SNR.")
            return noisy_waveform
        except Exception as e:
            logger.error(f"Failed to add noise: {e}")
            raise AudioProcessingError(f"Failed to add noise: {e}") from e

    def compute_mfcc(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        n_mfcc: int = 40,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute MFCCs from the audio waveform.

        Args:
            waveform (torch.Tensor): Audio waveform tensor.
            sample_rate (int): Sample rate of the audio waveform.
            n_mfcc (int): Number of MFCCs to return.
            **kwargs: Additional keyword arguments for MFCC.

        Returns:
            torch.Tensor: MFCC tensor.

        Raises:
            AudioProcessingError: If MFCC computation fails.
        """
        try:
            mfcc_transform = MFCC(
                sample_rate=sample_rate,
                n_mfcc=n_mfcc,
                **kwargs
            )
            mfcc = mfcc_transform(waveform)
            logger.info("Computed MFCCs.")
            return mfcc
        except Exception as e:
            logger.error(f"Failed to compute MFCCs: {e}")
            raise AudioProcessingError(f"Failed to compute MFCCs: {e}") from e

    def batch_process(
        self,
        filepaths: List[str],
        process_function: Callable[[torch.Tensor, int], None],
        num_workers: int = 1
    ) -> None:
        """
        Batch process multiple audio files concurrently.

        Args:
            filepaths (List[str]): List of audio file paths to process.
            process_function (Callable[[torch.Tensor, int], None]):
                A processing function that takes waveform and sample rate.
            num_workers (int): Number of worker threads to use for parallel processing.

        Raises:
            AudioProcessingError: If batch processing fails.
        """
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(self._process_file, filepath, process_function): filepath
                    for filepath in filepaths
                }
                for future in as_completed(futures):
                    filepath = futures[future]
                    try:
                        future.result()
                        logger.info(f"Processed file {filepath}")
                    except Exception as e:
                        logger.error(f"Failed to process file {filepath}: {e}")
        except Exception as e:
            logger.error(f"Failed to batch process files: {e}")
            raise AudioProcessingError(f"Failed to batch process files: {e}") from e

    def _process_file(
        self,
        filepath: str,
        process_function: Callable[[torch.Tensor, int], None]
    ) -> None:
        """
        Helper function to process a single audio file.

        Args:
            filepath (str): Path to the audio file.
            process_function (Callable[[torch.Tensor, int], None]): Processing function.

        Raises:
            Exception: Propagates exceptions from processing.
        """
        try:
            waveform, sample_rate = self.load_audio(filepath)
            process_function(waveform, sample_rate)
        except Exception as e:
            logger.error(f"Error processing file {filepath}: {e}")
            raise e

    def augment_audio(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        pitch_shift_semitones: Optional[float] = None,
        time_stretch_rate: Optional[float] = None,
        add_noise_params: Optional[Dict[str, Union[torch.Tensor, float]]] = None,
    ) -> torch.Tensor:
        """
        Apply common augmentation techniques to the audio waveform.

        Args:
            waveform (torch.Tensor): Original audio waveform.
            sample_rate (int): Sample rate of the audio waveform.
            pitch_shift_semitones (Optional[float]): Semitones to shift pitch.
            time_stretch_rate (Optional[float]): Rate for time stretching (>1.0 speeds up, <1.0 slows down).
            add_noise_params (Optional[Dict[str, Union[torch.Tensor, float]]]):
                Parameters for adding noise - requires 'noise_waveform' and 'snr'.

        Returns:
            torch.Tensor: Augmented audio waveform.

        Raises:
            AudioProcessingError: If augmentation fails.
        """
        try:
            augmented_waveform = waveform
            if pitch_shift_semitones is not None:
                pitch_transform = PitchShift(
                    sample_rate=sample_rate,
                    n_steps=pitch_shift_semitones
                )
                augmented_waveform = pitch_transform(augmented_waveform)
                logger.info(f"Applied pitch shift of {pitch_shift_semitones} semitones.")
            if time_stretch_rate is not None:
                spectrogram = Spectrogram()(augmented_waveform)
                time_stretch_transform = TimeStretch()
                stretched_spectrogram = time_stretch_transform(
                    spectrogram,
                    rate=time_stretch_rate
                )
                istft = InverseSpectrogram()
                augmented_waveform = istft(stretched_spectrogram)
                logger.info(f"Applied time stretch with rate {time_stretch_rate}.")
            if add_noise_params is not None:
                noise_waveform = add_noise_params['noise_waveform']
                snr = add_noise_params['snr']
                augmented_waveform = self.add_noise(
                    augmented_waveform,
                    noise_waveform,
                    snr
                )
            logger.info("Completed audio augmentation.")
            return augmented_waveform
        except Exception as e:
            logger.error(f"Failed to augment audio: {e}")
            raise AudioProcessingError(f"Failed to augment audio: {e}") from e

    # Additional methods can be added as needed to cover more functionalities.


# Example usage:
# if __name__ == '__main__':
#     audio_processor = AudioProcessor()
#     waveform, sample_rate = audio_processor.load_audio('path/to/audio.wav')

#     # Apply some transformations
#     transformed_waveform = audio_processor.apply_transforms(
#         waveform,
#         sample_rate,
#         transforms=[
#             lambda x: Resample(orig_freq=sample_rate, new_freq=16000)(x),
#             lambda x: AmplitudeToDB()(x),
#         ]
#     )

#     # Save the transformed audio
#     audio_processor.save_audio('path/to/transformed_audio.wav', transformed_waveform, 16000)