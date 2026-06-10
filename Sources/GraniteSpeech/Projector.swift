//
//  Projector.swift
//  GraniteSpeech
//
//  Created by Onit on 2026-06-03.
//
//  BLIP-2 Q-Former projector (Phase 4d), ported 1:1 from mlx-audio
//  `granite_speech.py` (EncoderProjector + QFormer* modules). Maps the encoder
//  output (B, L, 1024) to audio embeddings (B, nblocks*num_queries, 2048) via
//  windowed cross-attention between learnable queries and encoder states.
//

import Foundation
import MLX
import MLXNN

// MARK: - Multi-head attention (self or cross)

final class QFormerMultiHeadAttention: Module {
    @ModuleInfo(key: "query") var query: Linear
    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear

    let numHeads: Int
    let headDim: Int

    init(hiddenSize: Int, numHeads: Int, kvHiddenSize: Int? = nil) {
        self.numHeads = numHeads
        self.headDim = hiddenSize / numHeads
        let kvDim = kvHiddenSize ?? hiddenSize
        self._query = ModuleInfo(wrappedValue: Linear(hiddenSize, hiddenSize), key: "query")
        self._key = ModuleInfo(wrappedValue: Linear(kvDim, hiddenSize), key: "key")
        self._value = ModuleInfo(wrappedValue: Linear(kvDim, hiddenSize), key: "value")
    }

    func callAsFunction(_ hiddenStates: MLXArray, encoderHiddenStates: MLXArray? = nil) -> MLXArray {
        let B = hiddenStates.dim(0)
        let L = hiddenStates.dim(1)
        let kvInput = encoderHiddenStates ?? hiddenStates

        var q = query(hiddenStates)
        var k = key(kvInput)
        var v = value(kvInput)

        q = q.reshaped([B, L, numHeads, headDim]).transposed(0, 2, 1, 3)
        k = k.reshaped([B, -1, numHeads, headDim]).transposed(0, 2, 1, 3)
        v = v.reshaped([B, -1, numHeads, headDim]).transposed(0, 2, 1, 3)

        let scale = pow(Float(headDim), -0.5)
        var attn = matmul(q * scale, k.transposed(0, 1, 3, 2))
        attn = softmax(attn, axis: -1)
        let out = matmul(attn, v).transposed(0, 2, 1, 3).reshaped([B, L, -1])
        return out
    }
}

final class QFormerSelfOutput: Module {
    @ModuleInfo(key: "dense") var dense: Linear
    @ModuleInfo(key: "LayerNorm") var layerNorm: LayerNorm

    init(hiddenSize: Int, eps: Float) {
        self._dense = ModuleInfo(wrappedValue: Linear(hiddenSize, hiddenSize), key: "dense")
        self._layerNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: hiddenSize, eps: eps), key: "LayerNorm")
    }

    func callAsFunction(_ hiddenStates: MLXArray, _ inputTensor: MLXArray) -> MLXArray {
        layerNorm(dense(hiddenStates) + inputTensor)
    }
}

final class QFormerAttention: Module {
    @ModuleInfo(key: "attention") var attention: QFormerMultiHeadAttention
    @ModuleInfo(key: "output") var output: QFormerSelfOutput

    init(hiddenSize: Int, numHeads: Int, kvHiddenSize: Int? = nil, eps: Float) {
        self._attention = ModuleInfo(
            wrappedValue: QFormerMultiHeadAttention(
                hiddenSize: hiddenSize, numHeads: numHeads, kvHiddenSize: kvHiddenSize),
            key: "attention")
        self._output = ModuleInfo(wrappedValue: QFormerSelfOutput(hiddenSize: hiddenSize, eps: eps), key: "output")
    }

    func callAsFunction(_ hiddenStates: MLXArray, encoderHiddenStates: MLXArray? = nil) -> MLXArray {
        let attnOut = attention(hiddenStates, encoderHiddenStates: encoderHiddenStates)
        return output(attnOut, hiddenStates)
    }
}

final class QFormerIntermediate: Module, UnaryLayer {
    @ModuleInfo(key: "dense") var dense: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self._dense = ModuleInfo(wrappedValue: Linear(hiddenSize, intermediateSize), key: "dense")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { gelu(dense(x)) }
}

final class QFormerOutput: Module {
    @ModuleInfo(key: "dense") var dense: Linear
    @ModuleInfo(key: "LayerNorm") var layerNorm: LayerNorm

    init(intermediateSize: Int, hiddenSize: Int, eps: Float) {
        self._dense = ModuleInfo(wrappedValue: Linear(intermediateSize, hiddenSize), key: "dense")
        self._layerNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: hiddenSize, eps: eps), key: "LayerNorm")
    }

    func callAsFunction(_ hiddenStates: MLXArray, _ inputTensor: MLXArray) -> MLXArray {
        layerNorm(dense(hiddenStates) + inputTensor)
    }
}

final class QFormerLayer: Module {
    @ModuleInfo(key: "attention") var attention: QFormerAttention
    @ModuleInfo(key: "crossattention") var crossattention: QFormerAttention
    @ModuleInfo(key: "intermediate_query") var intermediateQuery: QFormerIntermediate
    @ModuleInfo(key: "output_query") var outputQuery: QFormerOutput

    init(_ c: GraniteProjectorConfig) {
        self._attention = ModuleInfo(
            wrappedValue: QFormerAttention(
                hiddenSize: c.hiddenSize, numHeads: c.numAttentionHeads, eps: c.layerNormEps),
            key: "attention")
        self._crossattention = ModuleInfo(
            wrappedValue: QFormerAttention(
                hiddenSize: c.hiddenSize, numHeads: c.numAttentionHeads,
                kvHiddenSize: c.encoderHiddenSize, eps: c.layerNormEps),
            key: "crossattention")
        self._intermediateQuery = ModuleInfo(
            wrappedValue: QFormerIntermediate(hiddenSize: c.hiddenSize, intermediateSize: c.intermediateSize),
            key: "intermediate_query")
        self._outputQuery = ModuleInfo(
            wrappedValue: QFormerOutput(
                intermediateSize: c.intermediateSize, hiddenSize: c.hiddenSize, eps: c.layerNormEps),
            key: "output_query")
    }

    func callAsFunction(_ hiddenStates: MLXArray, encoderHiddenStates: MLXArray) -> MLXArray {
        var h = attention(hiddenStates)
        h = crossattention(h, encoderHiddenStates: encoderHiddenStates)
        let intermediate = intermediateQuery(h)
        return outputQuery(intermediate, h)
    }
}

final class QFormerEncoder: Module {
    @ModuleInfo(key: "layer") var layer: [QFormerLayer]

    init(_ c: GraniteProjectorConfig) {
        self._layer = ModuleInfo(
            wrappedValue: (0 ..< c.numHiddenLayers).map { _ in QFormerLayer(c) }, key: "layer")
    }

    func callAsFunction(_ hiddenStates: MLXArray, encoderHiddenStates: MLXArray) -> MLXArray {
        var h = hiddenStates
        for l in layer {
            h = l(h, encoderHiddenStates: encoderHiddenStates)
        }
        return h
    }
}

final class QFormerModel: Module {
    @ModuleInfo(key: "layernorm") var layernorm: LayerNorm
    @ModuleInfo(key: "encoder") var encoder: QFormerEncoder

    init(_ c: GraniteProjectorConfig) {
        self._layernorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps), key: "layernorm")
        self._encoder = ModuleInfo(wrappedValue: QFormerEncoder(c), key: "encoder")
    }

    func callAsFunction(_ queryEmbeds: MLXArray, encoderHiddenStates: MLXArray) -> MLXArray {
        encoder(layernorm(queryEmbeds), encoderHiddenStates: encoderHiddenStates)
    }
}

// MARK: - Projector

public final class EncoderProjector: Module {
    @ParameterInfo(key: "query") var query: MLXArray
    @ModuleInfo(key: "qformer") var qformer: QFormerModel
    @ModuleInfo(key: "linear") var linear: Linear

    let hiddenSize: Int
    let windowSize: Int
    let numQueries: Int

    public init(_ config: GraniteSpeechConfig) {
        let pc = config.projectorConfig
        self.hiddenSize = pc.hiddenSize
        self.windowSize = config.windowSize
        self.numQueries = config.numQueries
        self._query = ParameterInfo(
            wrappedValue: MLXArray.zeros([1, config.numQueries, pc.hiddenSize]), key: "query")
        self._qformer = ModuleInfo(wrappedValue: QFormerModel(pc), key: "qformer")
        self._linear = ModuleInfo(
            wrappedValue: Linear(pc.hiddenSize, config.textConfig.hiddenSize), key: "linear")
    }

    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let B = hiddenStates.dim(0)
        let L = hiddenStates.dim(1)
        let D = hiddenStates.dim(2)

        let nblocks = (L + windowSize - 1) / windowSize
        let pad = nblocks * windowSize - L
        var hs = hiddenStates
        if pad > 0 {
            hs = padded(hs, widths: [.init((0, 0)), .init((0, pad)), .init((0, 0))])
        }
        hs = hs.reshaped([B * nblocks, windowSize, D])

        let q = broadcast(query, to: [B * nblocks, numQueries, hiddenSize])
        let queryOutput = qformer(q, encoderHiddenStates: hs)
        return linear(queryOutput.reshaped([B, nblocks * numQueries, -1]))
    }
}
