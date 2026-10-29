//
//  Encoder.swift
//  GraniteSpeech
//
//  Created by Kévin Naudin on 2026-06-03.
//
//  CTC Conformer encoder (Phase 4c), ported 1:1 from mlx-audio
//  `granite_speech.py` (CTCEncoder + ConformerBlock and submodules). Module
//  keys mirror the HF safetensors names via @ModuleInfo/@ParameterInfo so the
//  whole tree loads with `update(parameters: .unflattened(weights))`.
//

import Foundation
import MLX
import MLXNN

// MARK: - BatchNorm1d (inference-only, running stats)

final class GraniteBatchNorm1d: Module, UnaryLayer {
    let weight: MLXArray
    let bias: MLXArray
    @ParameterInfo(key: "running_mean") var runningMean: MLXArray
    @ParameterInfo(key: "running_var") var runningVar: MLXArray
    let eps: Float

    init(_ numFeatures: Int, eps: Float = 1e-5) {
        self.weight = MLXArray.ones([numFeatures])
        self.bias = MLXArray.zeros([numFeatures])
        self._runningMean = ParameterInfo(wrappedValue: MLXArray.zeros([numFeatures]), key: "running_mean")
        self._runningVar = ParameterInfo(wrappedValue: MLXArray.ones([numFeatures]), key: "running_var")
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        (x - runningMean) / sqrt(runningVar + eps) * weight + bias
    }
}

// MARK: - Feed forward

final class ConformerFeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "pre_norm") var preNorm: LayerNorm
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ c: GraniteEncoderConfig) {
        self._preNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: c.hiddenDim), key: "pre_norm")
        self._upProj = ModuleInfo(
            wrappedValue: Linear(c.hiddenDim, c.hiddenDim * c.feedforwardMult), key: "up_proj")
        self._downProj = ModuleInfo(
            wrappedValue: Linear(c.hiddenDim * c.feedforwardMult, c.hiddenDim), key: "down_proj")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(upProj(preNorm(x))))
    }
}

// MARK: - Attention (block-wise context + Shaw relative position)

final class ConformerAttention: Module {
    @ModuleInfo(key: "pre_norm") var preNorm: LayerNorm
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_kv") var toKV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    @ModuleInfo(key: "rel_pos_emb") var relPosEmb: Embedding

    let numHeads: Int
    let dimHead: Int
    let contextSize: Int
    let scale: Float

    init(_ c: GraniteEncoderConfig) {
        let inner = c.dimHead * c.numHeads
        self.numHeads = c.numHeads
        self.dimHead = c.dimHead
        self.contextSize = c.contextSize
        self.scale = pow(Float(c.dimHead), -0.5)
        self._preNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: c.hiddenDim), key: "pre_norm")
        self._toQ = ModuleInfo(wrappedValue: Linear(c.hiddenDim, inner, bias: false), key: "to_q")
        self._toKV = ModuleInfo(wrappedValue: Linear(c.hiddenDim, inner * 2, bias: false), key: "to_kv")
        self._toOut = ModuleInfo(wrappedValue: Linear(inner, c.hiddenDim), key: "to_out")
        self._relPosEmb = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: 2 * c.maxPosEmb + 1, dimensions: c.dimHead),
            key: "rel_pos_emb")
    }

    func callAsFunction(_ input: MLXArray, attentionDists: MLXArray) -> MLXArray {
        var x = preNorm(input)
        let B = x.dim(0)
        let N = x.dim(1)
        let C = contextSize

        let numBlocks = (N + C - 1) / C
        let remainder = N % C
        if remainder > 0 {
            x = padded(x, widths: [.init((0, 0)), .init((0, C - remainder)), .init((0, 0))])
        }

        var q = toQ(x)
        let kv = toKV(x)
        let parts = kv.split(parts: 2, axis: -1)
        var k = parts[0]
        var v = parts[1]

        // (B, numBlocks, C, numHeads, dimHead) -> (B, numBlocks, numHeads, C, dimHead)
        q = q.reshaped([B, numBlocks, C, numHeads, dimHead]).transposed(0, 1, 3, 2, 4)
        k = k.reshaped([B, numBlocks, C, numHeads, dimHead]).transposed(0, 1, 3, 2, 4)
        v = v.reshaped([B, numBlocks, C, numHeads, dimHead]).transposed(0, 1, 3, 2, 4)

        let relPos = relPosEmb(attentionDists) // (C, C, dimHead)
        let qe = expandedDimensions(q, axis: 4) // (B, nb, nh, C, 1, dh)
        let re = relPos.reshaped([1, 1, 1, C, C, dimHead])
        var posAttn = (qe * re).sum(axis: -1) * scale // (B, nb, nh, C, C)

        if remainder > 0 {
            // Mask cross/keys beyond `remainder` in the LAST block only.
            let idx = MLXArray(Int32(0) ..< Int32(C))
            let rowValid = expandedDimensions(idx, axis: 1) .< Int32(remainder) // (C,1)
            let colValid = expandedDimensions(idx, axis: 0) .< Int32(remainder) // (1,C)
            let validF = rowValid.asType(.float32) * colValid.asType(.float32) // (C,C)
            let blockIsLast = (MLXArray(Int32(0) ..< Int32(numBlocks)) .== Int32(numBlocks - 1))
                .reshaped([numBlocks, 1, 1]).asType(.float32)
            let invalid = (blockIsLast * (validF .< 0.5).asType(.float32)) .> 0.5 // (nb,C,C)
            let maskB = invalid.reshaped([1, numBlocks, 1, C, C])
            // Python uses finfo(dtype).min; -greatestFiniteMagnitude is fp32
            // and, if cast to bf16, saturates to -inf. Either way it's added to
            // the qk term and immediately softmaxed -> masked weights become 0.
            let minV = MLXArray(-Float.greatestFiniteMagnitude)
            posAttn = which(maskB, minV, posAttn)
        }

        var attn = matmul(q, k.transposed(0, 1, 2, 4, 3)) * scale + posAttn
        attn = softmax(attn, axis: -1)

        var out = matmul(attn, v) // (B, nb, nh, C, dh)
        out = out.transposed(0, 1, 3, 2, 4) // (B, nb, C, nh, dh)
        out = out.reshaped([B, -1, numHeads * dimHead])
        out = out[0..., 0 ..< N, 0...]
        return toOut(out)
    }
}

// MARK: - Convolution module

final class DepthWiseConv1d: Module, UnaryLayer {
    @ModuleInfo(key: "conv") var conv: Conv1d
    let padLeft: Int
    let padRight: Int

    init(_ chanIn: Int, _ chanOut: Int, kernelSize: Int) {
        let pad = kernelSize / 2
        let padOffset = (kernelSize + 1) % 2
        self.padLeft = pad
        self.padRight = pad - padOffset
        self._conv = ModuleInfo(
            wrappedValue: Conv1d(
                inputChannels: chanIn, outputChannels: chanOut,
                kernelSize: kernelSize, groups: chanIn, bias: false),
            key: "conv")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let p = padded(x, widths: [.init((0, 0)), .init((padLeft, padRight)), .init((0, 0))])
        return conv(p)
    }
}

final class ConformerConvModule: Module, UnaryLayer {
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "up_conv") var upConv: Conv1d
    @ModuleInfo(key: "depth_conv") var depthConv: DepthWiseConv1d
    @ModuleInfo(key: "batch_norm") var batchNorm: GraniteBatchNorm1d
    @ModuleInfo(key: "down_conv") var downConv: Conv1d

    init(_ c: GraniteEncoderConfig) {
        let inner = c.hiddenDim * c.convExpansionFactor
        self._norm = ModuleInfo(wrappedValue: LayerNorm(dimensions: c.hiddenDim), key: "norm")
        self._upConv = ModuleInfo(
            wrappedValue: Conv1d(inputChannels: c.hiddenDim, outputChannels: inner * 2, kernelSize: 1),
            key: "up_conv")
        self._depthConv = ModuleInfo(
            wrappedValue: DepthWiseConv1d(inner, inner, kernelSize: c.convKernelSize),
            key: "depth_conv")
        self._batchNorm = ModuleInfo(wrappedValue: GraniteBatchNorm1d(inner), key: "batch_norm")
        self._downConv = ModuleInfo(
            wrappedValue: Conv1d(inputChannels: inner, outputChannels: c.hiddenDim, kernelSize: 1),
            key: "down_conv")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = norm(x)
        h = upConv(h)
        let parts = h.split(parts: 2, axis: -1)
        h = parts[0] * sigmoid(parts[1])
        h = depthConv(h)
        h = silu(batchNorm(h))
        h = downConv(h)
        return h
    }
}

// MARK: - Block

final class ConformerBlock: Module {
    @ModuleInfo(key: "ff1") var ff1: ConformerFeedForward
    @ModuleInfo(key: "attn") var attn: ConformerAttention
    @ModuleInfo(key: "conv") var conv: ConformerConvModule
    @ModuleInfo(key: "ff2") var ff2: ConformerFeedForward
    @ModuleInfo(key: "post_norm") var postNorm: LayerNorm

    init(_ c: GraniteEncoderConfig) {
        self._ff1 = ModuleInfo(wrappedValue: ConformerFeedForward(c), key: "ff1")
        self._attn = ModuleInfo(wrappedValue: ConformerAttention(c), key: "attn")
        self._conv = ModuleInfo(wrappedValue: ConformerConvModule(c), key: "conv")
        self._ff2 = ModuleInfo(wrappedValue: ConformerFeedForward(c), key: "ff2")
        self._postNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: c.hiddenDim), key: "post_norm")
    }

    func callAsFunction(_ input: MLXArray, attentionDists: MLXArray) -> MLXArray {
        var x = 0.5 * ff1(input) + input
        x = attn(x, attentionDists: attentionDists) + x
        x = conv(x) + x
        x = 0.5 * ff2(x) + x
        return postNorm(x)
    }
}

// MARK: - CTC Encoder

public final class CTCEncoder: Module {
    @ModuleInfo(key: "input_linear") var inputLinear: Linear
    @ModuleInfo(key: "layers") var layers: [ConformerBlock]
    @ModuleInfo(key: "out") var out: Linear
    @ModuleInfo(key: "out_mid") var outMid: Linear

    let numLayers: Int
    let contextSize: Int
    let maxPosEmb: Int

    public init(_ c: GraniteEncoderConfig) {
        self.numLayers = c.numLayers
        self.contextSize = c.contextSize
        self.maxPosEmb = c.maxPosEmb
        self._inputLinear = ModuleInfo(
            wrappedValue: Linear(c.inputDim, c.hiddenDim), key: "input_linear")
        self._layers = ModuleInfo(
            wrappedValue: (0 ..< c.numLayers).map { _ in ConformerBlock(c) }, key: "layers")
        self._out = ModuleInfo(wrappedValue: Linear(c.hiddenDim, c.outputDim), key: "out")
        self._outMid = ModuleInfo(wrappedValue: Linear(c.outputDim, c.hiddenDim), key: "out_mid")
    }

    /// Shaw relative-position index matrix `(contextSize, contextSize)`:
    /// `clip(seq[:,None] - seq[None,:], -C, C) + max_pos_emb`. Computed per
    /// forward (a tiny int matrix, negligible vs the 16 conformer blocks)
    /// rather than stored - a stored `MLXArray` would be reflected as a model
    /// parameter and break `.allModelKeysSet` weight-load verification (it's a
    /// constant, not a checkpoint weight).
    private func attentionDists() -> MLXArray {
        let seq = MLXArray(Int32(0) ..< Int32(contextSize))
        let dist = expandedDimensions(seq, axis: 1) - expandedDimensions(seq, axis: 0)
        return clip(dist, min: Int32(-contextSize), max: Int32(contextSize)) + Int32(maxPosEmb)
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        let dists = attentionDists()
        var x = inputLinear(input)
        for (i, layer) in layers.enumerated() {
            x = layer(x, attentionDists: dists)
            if (i + 1) == numLayers / 2 {
                let xMid = out(x)
                x = x + outMid(softmax(xMid, axis: -1))
            }
        }
        return x
    }

    /// Per-block outputs (saved BEFORE the mid-layer injection, matching the
    /// `encoder_block_{k}.npy` fixtures). The returned array also includes the
    /// post-injection running state used to feed the next block.
    public func forwardCapturingBlocks(_ input: MLXArray) -> (blocks: [MLXArray], output: MLXArray) {
        let dists = attentionDists()
        var x = inputLinear(input)
        var blocks: [MLXArray] = []
        for (i, layer) in layers.enumerated() {
            x = layer(x, attentionDists: dists)
            blocks.append(x)
            if (i + 1) == numLayers / 2 {
                let xMid = out(x)
                x = x + outMid(softmax(xMid, axis: -1))
            }
        }
        return (blocks, x)
    }
}
