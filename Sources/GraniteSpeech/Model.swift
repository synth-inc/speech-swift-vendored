//
//  Model.swift
//  GraniteSpeech
//
//  Created by Kévin Naudin on 2026-06-03.
//
//  Top-level Granite-Speech model (Phase 4f): ties preprocessing -> encoder ->
//  projector -> prompt build (tokenizer) -> audio splicing -> greedy decode ->
//  detokenize. Mirrors mlx-audio `Model.generate`. The prompt uses the model's
//  canonical chat template `USER: {content}\n ASSISTANT:` (note the space before
//  ASSISTANT) — the same string `apply_chat_template(add_generation_prompt=true)`
//  produces — rendered directly here rather than via Jinja.
//

import Foundation
import MLX
import MLXNN
import Tokenizers

public final class GraniteSpeechModel {
    public let config: GraniteSpeechConfig
    let preprocessor: GranitePreprocessor
    let encoder: CTCEncoder
    let projector: EncoderProjector
    let decoder: GraniteForCausalLM
    let tokenizer: Tokenizer
    let audioTokenId: Int
    let eosTokenId: Int

    public static let defaultPrompt = "can you transcribe the speech into a written format?"

    public init(
        config: GraniteSpeechConfig,
        encoder: CTCEncoder,
        projector: EncoderProjector,
        decoder: GraniteForCausalLM,
        tokenizer: Tokenizer
    ) {
        self.config = config
        self.preprocessor = GranitePreprocessor()
        self.encoder = encoder
        self.projector = projector
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.audioTokenId = config.audioTokenIndex
        self.eosTokenId = tokenizer.eosTokenId ?? 100257
    }

    /// Load weights (3 shards) + tokenizer from a local model directory.
    public static func load(
        from modelDir: URL, config: GraniteSpeechConfig = .granite41_2b
    ) async throws -> GraniteSpeechModel {
        let weights = try GraniteWeights.loadSanitized(from: modelDir)

        // Verify every model parameter is filled and shapes match, so a future
        // key-mapping drift fails loudly here instead of producing garbage
        // transcripts (or a cryptic MLX crash) at inference time.
        let verify: Module.VerifyUpdate = [.allModelKeysSet, .shapeMismatch]

        let encoder = CTCEncoder(config.encoderConfig)
        try encoder.update(parameters: ModuleParameters.unflattened(
            GraniteWeights.subset(weights, prefix: "encoder.")), verify: verify)

        let projector = EncoderProjector(config)
        try projector.update(parameters: ModuleParameters.unflattened(
            GraniteWeights.subset(weights, prefix: "projector.")), verify: verify)

        let decoder = GraniteForCausalLM(config.textConfig)
        try decoder.update(parameters: ModuleParameters.unflattened(
            GraniteWeights.subset(weights, prefix: "language_model.")), verify: verify)

        eval(encoder, projector, decoder)

        let tokenizer = try await AutoTokenizer.from(modelFolder: modelDir, strict: false)

        return GraniteSpeechModel(
            config: config, encoder: encoder, projector: projector,
            decoder: decoder, tokenizer: tokenizer)
    }

    /// Canonical chat-template prompt `USER: <|audio|>*n{prompt}\n ASSISTANT:`
    /// (space before ASSISTANT) -> token ids.
    public func buildPrompt(numAudioTokens: Int, userPrompt: String? = nil) -> [Int] {
        let prompt = userPrompt ?? Self.defaultPrompt
        let placeholder = String(repeating: "<|audio|>", count: numAudioTokens)
        return tokenizer.encode(text: "USER: \(placeholder)\(prompt)\n ASSISTANT:")
    }

    /// Embed `promptIds` (audio placeholders -> 0) then splice `audioFeatures`
    /// `(1, numAudio, D)` into the contiguous `<|audio|>` positions.
    public func buildInputsEmbeds(promptIds: [Int], audioFeatures: MLXArray) -> MLXArray {
        let llmIds = promptIds.map { Int32($0 == audioTokenId ? 0 : $0) }
        let embeds = decoder.embed(MLXArray(llmIds).reshaped([1, llmIds.count])) // (1, L, D)

        guard let start = promptIds.firstIndex(of: audioTokenId) else { return embeds }
        let count = promptIds.reduce(0) { $0 + ($1 == audioTokenId ? 1 : 0) }
        let numAudio = min(count, audioFeatures.dim(1))

        let pre = embeds[0..., 0 ..< start]
        let audio = audioFeatures[0..., 0 ..< numAudio]
        let post = embeds[0..., (start + numAudio)...]
        return concatenated([pre, audio, post], axis: 1)
    }

    /// Audio features `(1, numAudioTokens, D)` from a 16 kHz mono waveform.
    public func audioFeatures(_ audio: MLXArray) -> MLXArray {
        projector(encoder(preprocessor.melFeatures(audio)))
    }

    /// Full pipeline: 16 kHz mono waveform -> transcript.
    public func transcribe(_ audio: MLXArray, userPrompt: String? = nil, maxTokens: Int = 256) -> String {
        let projected = audioFeatures(audio)
        let promptIds = buildPrompt(numAudioTokens: projected.dim(1), userPrompt: userPrompt)
        let inputsEmbeds = buildInputsEmbeds(promptIds: promptIds, audioFeatures: projected)
        let tokens = decoder.generate(
            inputsEmbeds: inputsEmbeds, eosTokenId: eosTokenId, maxTokens: maxTokens)
        return tokenizer.decode(tokens: tokens, skipSpecialTokens: true)
    }
}
