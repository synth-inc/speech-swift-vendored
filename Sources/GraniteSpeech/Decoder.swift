//
//  Decoder.swift
//  GraniteSpeech
//
//  Created by Kévin Naudin on 2026-06-03.
//
//  Granite dense-transformer decoder (Phase 4e), ported 1:1 from
//  `mlx_lm.models.granite` + the mlx-audio `granite_speech.Model.__call__`
//  embeddings path. Standard GQA + RoPE + SwiGLU transformer with the four
//  Granite scalars (embedding/attention/residual multipliers, logits scaling).
//  Weight keys map under `model.*` / `lm_head` (the `language_model.` prefix is
//  stripped by the caller).
//

import Foundation
import MLX
import MLXFast
import MLXNN

/// Minimal incremental KV cache (concat-based). One per decoder layer.
public final class KVCache {
    private var keys: MLXArray?
    private var values: MLXArray?
    /// Number of cached timesteps (the RoPE offset for the next call).
    public private(set) var offset = 0

    public init() {}

    /// Append `(keys, values)` (shape `(B, nKVHeads, L, headDim)`) and return the
    /// full cached tensors.
    public func update(keys k: MLXArray, values v: MLXArray) -> (MLXArray, MLXArray) {
        let newK: MLXArray
        let newV: MLXArray
        if let keys, let values {
            newK = concatenated([keys, k], axis: 2)
            newV = concatenated([values, v], axis: 2)
        } else {
            newK = k
            newV = v
        }
        keys = newK
        values = newV
        offset += k.dim(2)
        return (newK, newV)
    }
}

final class GraniteAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPE

    init(_ c: GraniteTextConfig) {
        let dim = c.hiddenSize
        self.nHeads = c.numAttentionHeads
        self.nKVHeads = c.numKeyValueHeads
        self.headDim = dim / c.numAttentionHeads
        self.scale = c.attentionMultiplier
        let bias = c.attentionBias
        self._qProj = ModuleInfo(wrappedValue: Linear(dim, nHeads * headDim, bias: bias), key: "q_proj")
        self._kProj = ModuleInfo(wrappedValue: Linear(dim, nKVHeads * headDim, bias: bias), key: "k_proj")
        self._vProj = ModuleInfo(wrappedValue: Linear(dim, nKVHeads * headDim, bias: bias), key: "v_proj")
        self._oProj = ModuleInfo(wrappedValue: Linear(nHeads * headDim, dim, bias: bias), key: "o_proj")
        self.rope = RoPE(dimensions: headDim, traditional: false, base: c.ropeTheta)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: KVCache?) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var q = qProj(x).reshaped([B, L, nHeads, headDim]).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped([B, L, nKVHeads, headDim]).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped([B, L, nKVHeads, headDim]).transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0
        q = rope(q, offset: offset)
        k = rope(k, offset: offset)
        if let cache {
            (k, v) = cache.update(keys: k, values: v)
        }

        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask)
        return oProj(out.transposed(0, 2, 1, 3).reshaped([B, L, -1]))
    }
}

final class GraniteMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ c: GraniteTextConfig) {
        let dim = c.hiddenSize
        let hidden = c.intermediateSize
        let bias = c.mlpBias
        self._gateProj = ModuleInfo(wrappedValue: Linear(dim, hidden, bias: bias), key: "gate_proj")
        self._upProj = ModuleInfo(wrappedValue: Linear(dim, hidden, bias: bias), key: "up_proj")
        self._downProj = ModuleInfo(wrappedValue: Linear(hidden, dim, bias: bias), key: "down_proj")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

final class GraniteTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: GraniteAttention
    @ModuleInfo(key: "mlp") var mlp: GraniteMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    let residualMultiplier: Float

    init(_ c: GraniteTextConfig) {
        self._selfAttn = ModuleInfo(wrappedValue: GraniteAttention(c), key: "self_attn")
        self._mlp = ModuleInfo(wrappedValue: GraniteMLP(c), key: "mlp")
        self._inputLayernorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps), key: "input_layernorm")
        self._postAttentionLayernorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps), key: "post_attention_layernorm")
        self.residualMultiplier = c.residualMultiplier
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: KVCache?) -> MLXArray {
        let r = selfAttn(inputLayernorm(x), mask: mask, cache: cache)
        let h = x + r * residualMultiplier
        let r2 = mlp(postAttentionLayernorm(h))
        return h + r2 * residualMultiplier
    }
}

final class GraniteModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [GraniteTransformerBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ c: GraniteTextConfig) {
        self._embedTokens = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: c.vocabSize, dimensions: c.hiddenSize), key: "embed_tokens")
        self._layers = ModuleInfo(
            wrappedValue: (0 ..< c.numHiddenLayers).map { _ in GraniteTransformerBlock(c) }, key: "layers")
        self._norm = ModuleInfo(wrappedValue: RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps), key: "norm")
    }
}

/// Granite decoder for Granite-Speech: consumes pre-spliced `inputs_embeds`
/// (audio features already inserted at the `<|audio|>` positions).
public final class GraniteForCausalLM: Module {
    @ModuleInfo(key: "model") var model: GraniteModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    let embeddingMultiplier: Float
    let logitsScaling: Float

    public init(_ c: GraniteTextConfig) {
        self.embeddingMultiplier = c.embeddingMultiplier
        self.logitsScaling = c.logitsScaling
        self._model = ModuleInfo(wrappedValue: GraniteModel(c), key: "model")
        self._lmHead = ModuleInfo(
            wrappedValue: Linear(c.hiddenSize, c.vocabSize, bias: false), key: "lm_head")
    }

    /// Additive causal mask (L, L): 0 on/below the diagonal, -inf above.
    static func causalMask(_ length: Int) -> MLXArray {
        let idx = MLXArray(Int32(0) ..< Int32(length))
        let above = expandedDimensions(idx, axis: 1) .< expandedDimensions(idx, axis: 0) // row < col
        return which(above, MLXArray(-Float.greatestFiniteMagnitude), MLXArray(Float(0)))
    }

    /// Forward over pre-spliced embeddings -> logits `(B, L, vocab)`.
    /// Pass `cache` (one `KVCache` per layer) for incremental decoding; a
    /// single-token step (L==1) uses no mask (it attends to all cached keys),
    /// matching Python's `create_attention_mask` returning None when N==1.
    public func callAsFunction(inputsEmbeds: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = inputsEmbeds * embeddingMultiplier
        let L = h.dim(1)
        let mask: MLXArray? = L > 1 ? Self.causalMask(L).asType(h.dtype) : nil
        for (i, layer) in model.layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        h = model.norm(h)
        // Only the last position's logits are ever consumed (next-token
        // prediction), so run lm_head (vocab 100353) over that row only - a
        // prefill over L tokens would otherwise waste L× the projection.
        let last = h[0..., (h.dim(1) - 1) ..< h.dim(1)] // (B, 1, hidden)
        return lmHead(last) / logitsScaling
    }

    /// Embed token ids `(B, L)` -> `(B, L, hidden)` (pre-multiplier; the forward
    /// applies `embedding_multiplier`). Used to feed generated tokens back in.
    public func embed(_ ids: MLXArray) -> MLXArray {
        model.embedTokens(ids)
    }

    /// Greedy (temperature 0) decode from pre-spliced `inputsEmbeds`. Returns the
    /// generated token ids (excluding EOS), mirroring `Model.generate`.
    public func generate(inputsEmbeds: MLXArray, eosTokenId: Int, maxTokens: Int = 256) -> [Int] {
        let cache = (0 ..< model.layers.count).map { _ in KVCache() }

        var logits = self(inputsEmbeds: inputsEmbeds, cache: cache)
        var token = argMax(logits[0, -1]).item(Int.self)

        var tokens: [Int] = []
        while tokens.count < maxTokens {
            if token == eosTokenId { break }
            tokens.append(token)
            let emb = embed(MLXArray([Int32(token)]).reshaped([1, 1]))
            logits = self(inputsEmbeds: emb, cache: cache)
            token = argMax(logits[0, -1]).item(Int.self)
        }
        return tokens
    }
}
