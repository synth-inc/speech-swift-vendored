# speech-swift-vendored

On-device speech inference for Apple platforms, built on
[MLX-Swift](https://github.com/ml-explore/mlx-swift).

This package bundles:

- A **vendored subset** of [soniqo/speech-swift](https://github.com/soniqo/speech-swift)
  (Qwen3-ASR transcription, voice-activity detection, and shared audio/MLX
  helpers), with the platform floor lowered to **macOS 14 / iOS 18**.
- An **original** native MLX-Swift port of
  [IBM Granite-Speech 4.1-2b](https://huggingface.co/ibm-granite/granite-speech-4.1-2b).

See [`NOTICE.md`](NOTICE.md) for full attribution and the provenance of each
target.

## Why this repository exists

Upstream `soniqo/speech-swift` declares a `.macOS("15.0")` minimum, which
prevents SwiftPM resolution from a macOS 14 (Sonoma) consumer even though the
inference code runs fine on macOS 14. This vendor lowers the floor to
`.macOS("14.0")` by gating the only macOS 15-only code path (the CoreML
stateful prediction path) behind `@available`. When upstream lowers its own
requirement, the vendored subset should be dropped in favour of it.

## Products

| Product | Description |
|---|---|
| `GraniteSpeech` | Native MLX-Swift port of IBM Granite-Speech 4.1-2b (original work). |
| `Qwen3ASR` | Qwen3-ASR speech recognition, with optional streaming and forced alignment. |
| `SpeechVAD` | Voice-activity detection / diarization (Silero, Pyannote, Sortformer, WeSpeaker). |
| `AudioCommon` | Audio I/O, resampling, tokenizers, HuggingFace weight download. |
| `MLXCommon` | Shared MLX building blocks (SDPA, quantized layers, memory helpers). |

## Requirements

- macOS 14+ / iOS 18+
- Swift 5.10+ (Xcode 15.3+)
- Apple silicon (MLX runs on the Metal GPU)

## Installation

Add the package to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/synth-inc/speech-swift-vendored.git", branch: "main")
]
```

> No versioned release is tagged yet - pin to `branch: "main"` or a specific
> revision until a tag is published.

Then add the products you need to a target:

```swift
.target(
    name: "YourApp",
    dependencies: [
        .product(name: "GraniteSpeech", package: "speech-swift-vendored"),
        .product(name: "Qwen3ASR", package: "speech-swift-vendored")
    ]
)
```

## Models

Model weights are **not** bundled. Download them from HuggingFace at runtime -
`AudioCommon.HuggingFaceDownloader` is provided for this. GraniteSpeech expects
a local directory containing the model's safetensors shards, `config.json`, and
tokenizer files (e.g. the contents of
[`ibm-granite/granite-speech-4.1-2b`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b)).

## Quickstart - GraniteSpeech

```swift
import GraniteSpeech
import MLX

// `modelDir` is a local folder with the safetensors shards + tokenizer.
let model = try await GraniteSpeechModel.load(from: modelDir)

// 16 kHz mono waveform as Float samples.
let audio = MLXArray(samples) // [Float], 16 kHz mono

let transcript = model.transcribe(audio)
print(transcript)
```

`transcribe(_:userPrompt:maxTokens:)` runs the full pipeline
(preprocessing → encoder → projector → greedy decode → detokenize). Pass a
custom `userPrompt` to override the default transcription instruction.

## Quickstart - Qwen3-ASR

Load 16 kHz mono audio with `AudioFileLoader`, then transcribe through a loaded
`Qwen3ASRModel`:

```swift
import Qwen3ASR
import AudioCommon

let samples = try AudioFileLoader.load(url: wavURL, targetSampleRate: 16000)

let transcript = model.transcribe(
    audio: samples,
    sampleRate: 16000,
    options: Qwen3DecodingOptions(language: "en", repetitionPenalty: 1.1)
)
```

Weight loading is wired through `Qwen3ASR.WeightLoading` /
`AudioCommon.HuggingFaceDownloader`; `StreamingASR` provides chunked streaming
transcription gated by `SpeechVAD`.

## License

Apache License 2.0 - see [`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md).
