# Install dependencies: pip install -U qwen-tts soundfile torch
# Example: python tts.py "Hello! This is Qwen3 TTS." --output hello.wav

import argparse
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

def main():
	parser = argparse.ArgumentParser(description="Generate speech with Qwen3-TTS.")
	parser.add_argument("text", help="Text to speak")
	parser.add_argument("--output", type=Path, default=Path("speech.wav"))
	parser.add_argument("--language", default="English")
	parser.add_argument("--speaker", default="Vivian")
	parser.add_argument("--instruct", default="", help="Optional speaking style instruction")
	parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
	parser.add_argument(
		"--device", choices=["cpu", "cuda"],
		default="cuda" if torch.cuda.is_available() else "cpu",
	)
	args = parser.parse_args()
	if not args.text.strip():
		parser.error("text must not be empty")
	if args.device == "cuda" and not torch.cuda.is_available():
		parser.error("CUDA is not available; use --device cpu")

	dtype = torch.float32
	if args.device == "cuda":
		dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

	# Downloads the model on first use; CPU inference can be slow.
	model = Qwen3TTSModel.from_pretrained(
		args.model,
		device_map=args.device,
		dtype=dtype,
		attn_implementation="sdpa",
	)
	with torch.inference_mode():
		waveforms, sample_rate = model.generate_custom_voice(
			text=args.text,
			language=args.language,
			speaker=args.speaker,
			instruct=args.instruct,
		)

	args.output.parent.mkdir(parents=True, exist_ok=True)
	sf.write(str(args.output), waveforms[0], sample_rate, format="WAV")
	print(f"Saved audio to {args.output.resolve()}")

if __name__ == "__main__":
	main()
