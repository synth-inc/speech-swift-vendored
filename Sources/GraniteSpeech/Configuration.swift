//
//  Configuration.swift
//  GraniteSpeech
//
//  Created by Onit on 2026-06-03.
//
//  Native MLX-Swift port of IBM Granite-Speech 4.1-2b.
//  Config mirrors mlx-audio `granite_speech/config.py`. Field values are the
//  resolved runtime config of `ibm-granite/granite-speech-4.1-2b` (the dataclass
//  defaults in config.py are placeholders; real values come from the HF
//  config.json). Decodable from that config.json; `.granite41_2b` is the
//  hardcoded fallback used when no JSON is available.
//

import Foundation

/// CTC Conformer encoder configuration.
public struct GraniteEncoderConfig: Codable, Sendable {
    public var inputDim: Int = 160
    public var numLayers: Int = 16
    public var hiddenDim: Int = 1024
    public var feedforwardMult: Int = 4
    public var numHeads: Int = 8
    public var dimHead: Int = 128
    public var outputDim: Int = 348
    public var contextSize: Int = 200
    public var maxPosEmb: Int = 512
    public var convKernelSize: Int = 15
    public var convExpansionFactor: Int = 2

    enum CodingKeys: String, CodingKey {
        case inputDim = "input_dim"
        case numLayers = "num_layers"
        case hiddenDim = "hidden_dim"
        case feedforwardMult = "feedforward_mult"
        case numHeads = "num_heads"
        case dimHead = "dim_head"
        case outputDim = "output_dim"
        case contextSize = "context_size"
        case maxPosEmb = "max_pos_emb"
        case convKernelSize = "conv_kernel_size"
        case convExpansionFactor = "conv_expansion_factor"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func d<T: Decodable>(_ k: CodingKeys, _ fallback: T) -> T {
            (try? c.decode(T.self, forKey: k)) ?? fallback
        }
        inputDim = d(.inputDim, inputDim)
        numLayers = d(.numLayers, numLayers)
        hiddenDim = d(.hiddenDim, hiddenDim)
        feedforwardMult = d(.feedforwardMult, feedforwardMult)
        numHeads = d(.numHeads, numHeads)
        dimHead = d(.dimHead, dimHead)
        outputDim = d(.outputDim, outputDim)
        contextSize = d(.contextSize, contextSize)
        maxPosEmb = d(.maxPosEmb, maxPosEmb)
        convKernelSize = d(.convKernelSize, convKernelSize)
        convExpansionFactor = d(.convExpansionFactor, convExpansionFactor)
    }
}

/// BLIP-2 Q-Former projector configuration.
public struct GraniteProjectorConfig: Codable, Sendable {
    public var hiddenSize: Int = 1024
    public var numHiddenLayers: Int = 2
    public var numAttentionHeads: Int = 16
    public var intermediateSize: Int = 4096
    public var layerNormEps: Float = 1e-12
    public var encoderHiddenSize: Int = 1024
    public var crossAttentionFrequency: Int = 1

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case layerNormEps = "layer_norm_eps"
        case encoderHiddenSize = "encoder_hidden_size"
        case crossAttentionFrequency = "cross_attention_frequency"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func d<T: Decodable>(_ k: CodingKeys, _ fallback: T) -> T {
            (try? c.decode(T.self, forKey: k)) ?? fallback
        }
        hiddenSize = d(.hiddenSize, hiddenSize)
        numHiddenLayers = d(.numHiddenLayers, numHiddenLayers)
        numAttentionHeads = d(.numAttentionHeads, numAttentionHeads)
        intermediateSize = d(.intermediateSize, intermediateSize)
        layerNormEps = d(.layerNormEps, layerNormEps)
        encoderHiddenSize = d(.encoderHiddenSize, encoderHiddenSize)
        crossAttentionFrequency = d(.crossAttentionFrequency, crossAttentionFrequency)
    }
}

/// Granite dense-transformer decoder (LLM) configuration.
public struct GraniteTextConfig: Codable, Sendable {
    public var vocabSize: Int = 100353
    public var hiddenSize: Int = 2048
    public var intermediateSize: Int = 4096
    public var numHiddenLayers: Int = 40
    public var numAttentionHeads: Int = 16
    public var numKeyValueHeads: Int = 4
    public var maxPositionEmbeddings: Int = 4096
    public var rmsNormEps: Float = 1e-5
    public var ropeTheta: Float = 10000.0
    public var attentionBias: Bool = false
    public var mlpBias: Bool = false
    public var attentionMultiplier: Float = 0.0078125
    public var embeddingMultiplier: Float = 12.0
    public var residualMultiplier: Float = 0.22
    public var logitsScaling: Float = 8.0
    public var tieWordEmbeddings: Bool = false

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case attentionMultiplier = "attention_multiplier"
        case embeddingMultiplier = "embedding_multiplier"
        case residualMultiplier = "residual_multiplier"
        case logitsScaling = "logits_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func d<T: Decodable>(_ k: CodingKeys, _ fallback: T) -> T {
            (try? c.decode(T.self, forKey: k)) ?? fallback
        }
        vocabSize = d(.vocabSize, vocabSize)
        hiddenSize = d(.hiddenSize, hiddenSize)
        intermediateSize = d(.intermediateSize, intermediateSize)
        numHiddenLayers = d(.numHiddenLayers, numHiddenLayers)
        numAttentionHeads = d(.numAttentionHeads, numAttentionHeads)
        numKeyValueHeads = d(.numKeyValueHeads, numKeyValueHeads)
        maxPositionEmbeddings = d(.maxPositionEmbeddings, maxPositionEmbeddings)
        rmsNormEps = d(.rmsNormEps, rmsNormEps)
        ropeTheta = d(.ropeTheta, ropeTheta)
        attentionBias = d(.attentionBias, attentionBias)
        mlpBias = d(.mlpBias, mlpBias)
        attentionMultiplier = d(.attentionMultiplier, attentionMultiplier)
        embeddingMultiplier = d(.embeddingMultiplier, embeddingMultiplier)
        residualMultiplier = d(.residualMultiplier, residualMultiplier)
        logitsScaling = d(.logitsScaling, logitsScaling)
        tieWordEmbeddings = d(.tieWordEmbeddings, tieWordEmbeddings)
    }
}

/// Top-level Granite-Speech model configuration.
public struct GraniteSpeechConfig: Codable, Sendable {
    public var encoderConfig: GraniteEncoderConfig = .init()
    public var projectorConfig: GraniteProjectorConfig = .init()
    public var textConfig: GraniteTextConfig = .init()
    public var audioTokenIndex: Int = 100352
    public var downsampleRate: Int = 5
    public var windowSize: Int = 15
    public var hasLoraAdapter: Bool = false

    enum CodingKeys: String, CodingKey {
        case encoderConfig = "encoder_config"
        case projectorConfig = "projector_config"
        case textConfig = "text_config"
        case audioTokenIndex = "audio_token_index"
        case downsampleRate = "downsample_rate"
        case windowSize = "window_size"
        case hasLoraAdapter = "has_lora_adapter"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func d<T: Decodable>(_ k: CodingKeys, _ fallback: T) -> T {
            (try? c.decode(T.self, forKey: k)) ?? fallback
        }
        encoderConfig = d(.encoderConfig, encoderConfig)
        projectorConfig = d(.projectorConfig, projectorConfig)
        textConfig = d(.textConfig, textConfig)
        audioTokenIndex = d(.audioTokenIndex, audioTokenIndex)
        downsampleRate = d(.downsampleRate, downsampleRate)
        windowSize = d(.windowSize, windowSize)
        hasLoraAdapter = d(.hasLoraAdapter, hasLoraAdapter)
    }

    /// Number of learnable Q-Former queries per window (`window_size / downsample_rate`).
    public var numQueries: Int { windowSize / downsampleRate }

    /// Hardcoded config for `ibm-granite/granite-speech-4.1-2b`.
    public static var granite41_2b: GraniteSpeechConfig { GraniteSpeechConfig() }

    /// Decode from a HuggingFace `config.json` payload.
    public static func load(from url: URL) throws -> GraniteSpeechConfig {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(GraniteSpeechConfig.self, from: data)
    }
}
