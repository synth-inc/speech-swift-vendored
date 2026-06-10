# NOTICE

This repository contains two distinct bodies of code, both distributed under
the Apache License 2.0 — see [`LICENSE`](LICENSE):

1. A **vendored subset** of
   [soniqo/speech-swift](https://github.com/soniqo/speech-swift)
   (`Qwen3ASR`, `AudioCommon`, `MLXCommon`, `SpeechVAD`).
   Copyright 2025 Ivan Digital (original work).
2. **Original work** — the `GraniteSpeech` target — authored in this
   repository and not derived from soniqo/speech-swift.
   Copyright 2026 Onit.

## Vendored content

Only the targets required by `Qwen3ASR` inference were copied:

- `Sources/Qwen3ASR/`
- `Sources/AudioCommon/`
- `Sources/MLXCommon/`
- `Sources/SpeechVAD/`

Tests, executables, examples, and other targets (`Qwen3TTS`, `CosyVoiceTTS`,
`PersonaPlex`, `SpeechCore`, `KokoroTTS`, `ParakeetASR`, `ParakeetStreamingASR`,
`NemotronStreamingASR`, `OmnilingualASR`, `VibeVoiceTTS`, `Qwen3Chat`,
`MADLADTranslation`, `SpeechUI`, `SpeechWakeWord`, `AudioCLI`,
`AudioServerCLI`, …) are not vendored.

## Modifications relative to upstream

The Swift source files under `Sources/` are reproduced from
`soniqo/speech-swift@0.0.15` and carry their original copyright notices.

### `Package.swift`

- Platform requirements lowered from `.macOS("15.0")` / `.iOS("18.0")` to
  `.macOS("14.0")` / `.iOS("18.0")`. The upstream `15.0` minimum is enforced
  only by the CoreML stateful prediction path (`CoreMLTextDecoder`,
  `CoreMLASRModel`), which uses `MLState` / `MLModel.makeState()` /
  `MLModel.prediction(from:using:)` — APIs introduced in macOS 15 / iOS 18.
  The MLX inference path (`Qwen3ASRModel`) does not depend on those APIs
  and runs on macOS 14 once the affected types are properly gated.
- The product, target, and dependency surface has been pruned to the four
  vendored targets (`Qwen3ASR`, `AudioCommon`, `MLXCommon`, `SpeechVAD`)
  and the public products consumed downstream.
- A `GraniteSpeech` target was added — this is original work, not part of the
  vendored subset (see *Original work* below).

### `Sources/Qwen3ASR/CoreMLTextDecoder.swift`, `Sources/Qwen3ASR/CoreMLASRModel.swift`

- Added `@available(macOS 15, iOS 18, *)` to `CoreMLTextDecoder`,
  `CoreMLASRModel`, and both extensions of `CoreMLASRModel`. This makes the
  CoreML inference path accessible only on macOS 15+ / iOS 18+ — the only
  platforms where its required CoreML APIs exist — while letting the rest
  of the package (including the MLX-based `Qwen3ASRModel` entry point) build
  on macOS 14+.

No other vendored source code has been modified.

## Original work

`Sources/GraniteSpeech/` is original work authored in this repository: a native
MLX-Swift port of [IBM Granite-Speech 4.1-2b](https://huggingface.co/ibm-granite/granite-speech-4.1-2b),
structurally mirroring the reference [mlx-audio](https://github.com/Blaizzy/mlx-audio)
implementation. It is **not** derived from soniqo/speech-swift; it only depends
on the vendored `AudioCommon` and `MLXCommon` targets. Copyright 2026
Onit, licensed under the Apache License 2.0.

## Purpose

This vendor exists to allow downstream consumers to depend on Qwen3-ASR
inference while keeping a `MACOSX_DEPLOYMENT_TARGET` of `14.0` (macOS Sonoma
support). The upstream package declares `.macOS("15.0")`, which prevents SPM
resolution from a Sonoma-compatible consumer even though the underlying code
runs fine on macOS 14.

When upstream lowers its platform requirement, the vendored subset should be
removed in favour of the upstream package.
