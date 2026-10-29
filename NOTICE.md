# NOTICE

This repository contains a vendored subset of
[soniqo/speech-swift](https://github.com/soniqo/speech-swift), licensed under
the Apache License 2.0 - see [`LICENSE`](LICENSE).

Copyright 2025 Ivan Digital (original work).

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
  `MLModel.prediction(from:using:)` - APIs introduced in macOS 15 / iOS 18.
  The MLX inference path (`Qwen3ASRModel`) does not depend on those APIs
  and runs on macOS 14 once the affected types are properly gated.
- The product, target, and dependency surface has been pruned to the four
  vendored targets (`Qwen3ASR`, `AudioCommon`, `MLXCommon`, `SpeechVAD`)
  and the public products consumed downstream.

### `Sources/Qwen3ASR/CoreMLTextDecoder.swift`, `Sources/Qwen3ASR/CoreMLASRModel.swift`

- Added `@available(macOS 15, iOS 18, *)` to `CoreMLTextDecoder`,
  `CoreMLASRModel`, and both extensions of `CoreMLASRModel`. This makes the
  CoreML inference path accessible only on macOS 15+ / iOS 18+ - the only
  platforms where its required CoreML APIs exist - while letting the rest
  of the package (including the MLX-based `Qwen3ASRModel` entry point) build
  on macOS 14+.

### `Sources/Qwen3ASR/TrieBoost.swift` (added), `Sources/Qwen3ASR/Qwen3ASR.swift`, `Tests/Qwen3ASRTests/` (added)

- Added decode-time contextual biasing ("token-trie shallow fusion") for
  custom-dictionary terms: `Qwen3TokenTrie`, `Qwen3TrieBoostController`, and
  a `generateTrieBoosted` greedy decode loop with snapshot/rollback
  revocation, an EOS guard, and a term-chaining cooldown. `Qwen3DecodingOptions`
  gained `boostTerms` / `boostStrength` / `boostMaxBacktracks`; the greedy
  fast path is unchanged when boosting is off. Not present upstream.

No other source code has been modified.

## Purpose

This vendor exists to allow the Cloudless Voice app (and other consumers) to depend on
Qwen3-ASR inference while keeping a `MACOSX_DEPLOYMENT_TARGET` of `14.0`
(macOS Sonoma support). The upstream package declares `.macOS("15.0")`,
which prevents SPM resolution from a Sonoma-compatible consumer even though
the underlying code runs fine on macOS 14.

When upstream lowers its platform requirement, this vendor should be removed
in favour of the upstream package.
