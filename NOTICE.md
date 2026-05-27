# NOTICE

This repository contains a vendored subset of
[soniqo/speech-swift](https://github.com/soniqo/speech-swift), licensed under
the Apache License 2.0 — see [`LICENSE`](LICENSE).

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

The Swift source files under `Sources/` are reproduced **verbatim** from
`soniqo/speech-swift@0.0.15` and carry their original copyright notices.

The only modification is to `Package.swift`:

- Platform requirements lowered from `.macOS("15.0")` / `.iOS("18.0")` to
  `.macOS("14.0")` / `.iOS("18.0")`. An audit of the four vendored targets
  found no macOS 15-only APIs; the upstream `15.0` declaration is defensive
  rather than mandatory.
- The product, target, and dependency surface has been pruned to the four
  vendored targets and a single library product (`Qwen3ASR`).

No source code under `Sources/` has been modified.

## Purpose

This vendor exists to allow `onit-beacon` (and other consumers) to depend on
Qwen3-ASR inference while keeping a `MACOSX_DEPLOYMENT_TARGET` of `14.0`
(macOS Sonoma support). The upstream package declares `.macOS("15.0")`,
which prevents SPM resolution from a Sonoma-compatible consumer even though
the underlying code runs fine on macOS 14.

When upstream lowers its platform requirement, this vendor should be removed
in favour of the upstream package.
