from typing import List, Dict, Union, Optional, Any, Tuple
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
import torch
import numpy as np
from datasets import Audio
import numpy as np
from audioutils import ffmpeg_read, ffmpeg_microphone, ffmpeg_microphone_live

class SpeechToTextPipeline:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        chunk_length_s: float = 30.0,
        stride_length_s: Optional[float] = None,
        *model_args: Any,
        **kwargs: Any
    ):
        self.model_name = model_name
        self.device = device
        self.chunk_length_s = chunk_length_s
        self.stride_length_s = stride_length_s or chunk_length_s / 6
        self.model_args = model_args
        self.kwargs = kwargs
        self.processor, self.model = self.load_model_and_processor()
        self.sampling_rate = self.processor.feature_extractor.sampling_rate

    def load_model_and_processor(self) -> Tuple[AutoProcessor, AutoModelForSpeechSeq2Seq]:
        processor = AutoProcessor.from_pretrained(self.model_name, *self.model_args, **self.kwargs)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            *self.model_args,
            **self.kwargs
        ).to(self.device)
        return processor, model

    def preprocess_audio(self, audio_file: Union[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        if isinstance(audio_file, str):
            audio = Audio().decode_audio(audio_file)
        else:
            audio = audio_file

        if audio["sampling_rate"] != self.sampling_rate:
            raise ValueError(f"Audio sampling rate must be {self.sampling_rate}")

        return self.processor(
            audio["array"],
            sampling_rate=self.sampling_rate,
            return_tensors="pt"
        ).to(self.device)

    def transcribe(
        self,
        audio_file: Union[str, np.ndarray],
        return_timestamps: bool = False,
        **generation_kwargs: Any
    ) -> Dict[str, Any]:
        inputs = self.preprocess_audio(audio_file)

        forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="en", task="transcribe")

        transcription = self.model.generate(
            **inputs,
            forced_decoder_ids=forced_decoder_ids,
            return_timestamps=return_timestamps,
            **generation_kwargs
        )

        result = self.processor.batch_decode(transcription, skip_special_tokens=True)

        if return_timestamps:
            timestamps = self.processor.batch_decode(transcription, output_timestamps=True)
            return {"text": result[0], "timestamps": timestamps[0]["timestamps"]}
        else:
            return {"text": result[0]}

    def create_pipeline(
        self,
        task: str = "automatic-speech-recognition",
        **kwargs: Any
    ) -> pipeline:
        return pipeline(
            task=task,
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=self.chunk_length_s,
            stride_length_s=self.stride_length_s,
            device=self.device,
            **kwargs
        )

    def batch_transcribe(
        self,
        audio_files: List[Union[str, np.ndarray]],
        batch_size: int = 8,
        return_timestamps: bool = False,
        **generation_kwargs: Any
    ) -> List[Dict[str, Any]]:
        results = []
        for i in range(0, len(audio_files), batch_size):
            batch = audio_files[i:i+batch_size]
            inputs = [self.preprocess_audio(audio) for audio in batch]
            
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="en", task="transcribe")
            
            transcriptions = self.model.generate(
                **{k: torch.cat([inp[k] for inp in inputs]) for k in inputs[0].keys()},
                forced_decoder_ids=forced_decoder_ids,
                return_timestamps=return_timestamps,
                **generation_kwargs
            )
            
            batch_results = self.processor.batch_decode(transcriptions, skip_special_tokens=True)
            
            if return_timestamps:
                timestamps = self.processor.batch_decode(transcriptions, output_timestamps=True)
                results.extend([{"text": text, "timestamps": ts["timestamps"]} for text, ts in zip(batch_results, timestamps)])
            else:
                results.extend([{"text": text} for text in batch_results])
        
        return results


# # Initialize the pipeline
# stt_pipeline = SpeechToTextPipeline("openai/whisper-large-v2")

# # Transcribe a single audio file
# result = stt_pipeline.transcribe("path/to/audio.wav", return_timestamps=True)
# print(result["text"])
# print(result["timestamps"])

# # Batch transcribe multiple audio files
# audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
# batch_results = stt_pipeline.batch_transcribe(audio_files, batch_size=2)
# for result in batch_results:
#     print(result["text"])

# # Create a Hugging Face pipeline
# hf_pipeline = stt_pipeline.create_pipeline()




def test_file_transcription(pipeline, audio_file):
    print("Testing file transcription:")
    audio = ffmpeg_read(open(audio_file, "rb").read(), pipeline.sampling_rate)
    result = pipeline.transcribe(audio)
    print(f"Transcription: {result['text']}")
    print()

def test_microphone_transcription(pipeline, duration=5):
    print(f"Testing microphone transcription (speaking for {duration} seconds):")
    microphone = ffmpeg_microphone(pipeline.sampling_rate, duration)
    audio = np.frombuffer(next(microphone), dtype=np.float32)
    result = pipeline.transcribe(audio)
    print(f"Transcription: {result['text']}")
    print()

def test_streaming_transcription(pipeline, duration=30, chunk_length_s=5):
    print(f"Testing streaming transcription (speaking for {duration} seconds):")
    stream = ffmpeg_microphone_live(
        sampling_rate=pipeline.sampling_rate,
        chunk_length_s=chunk_length_s,
        stream_chunk_s=1,
    )

    for i, chunk in enumerate(stream):
        if i * chunk_length_s >= duration:
            break

        if not chunk["partial"]:
            result = pipeline.transcribe(chunk["raw"])
            print(f"Chunk {i + 1} transcription: {result['text']}")

    print()

# def main():
#     # Initialize the pipeline
#     model_name = "openai/whisper-small"
#     pipeline = SpeechToTextPipeline(model_name)

#     # Test file transcription
#     audio_file = "path/to/your/audio/file.wav"  # Replace with your audio file path
#     test_file_transcription(pipeline, audio_file)

#     # Test microphone transcription
#     test_microphone_transcription(pipeline)

#     # Test streaming transcription
#     test_streaming_transcription(pipeline)

# if __name__ == "__main__":
#     main()