//
//  TrieBoost.swift
//  Qwen3ASR
//
//  Decode-time contextual biasing ("shallow-fusion hotword boosting") for the
//  Qwen3-ASR greedy decoder.
//
//  Mechanism: dictionary terms are tokenized into a token-prefix trie. At each
//  decode step, tokens that would start or continue a registered term receive a
//  logit bonus before argmax. Because greedy decoding cannot "un-emit" a token,
//  boosted picks are speculative: when the boost flips a decision, the loop
//  snapshots (scores, KV cache, emitted count) at that position; if the trie
//  path later dies before completing a term, the loop rolls back to the
//  snapshot and retries with the offending entry token banned. The KV cache in
//  this package is functional (each step returns fresh arrays), so a rollback
//  is just holding the old references - no recompute.
//
//  A completed term ends the speculation and keeps the emitted tokens. A
//  bounded backtrack budget (`maxBacktracks`) guarantees the loop degenerates
//  to plain greedy rather than thrashing on pathological audio.
//

import Foundation
import MLX

// MARK: - Token trie

/// Token-prefix trie over the tokenizations of registered dictionary terms.
///
/// Each term is inserted in two surface variants - with a leading space
/// (mid-sentence, the common case for byte-level BPE) and bare (utterance
/// start). Multi-word terms form a single path spanning their word tokens.
public final class Qwen3TokenTrie: @unchecked Sendable {
    public struct Node {
        public var children: [Int32: Int] = [:]
        public var terminal = false
        /// Non-nil on alias terminals: the canonical-form tokens that replace
        /// the emitted alias span when this path completes (same leading-space
        /// variant as the alias surface). Nil = canonical path, keep as-is.
        public var rewrite: [Int32]? = nil
    }

    /// Node storage; index 0 is the root.
    public private(set) var nodes: [Node] = [Node()]

    public var isEmpty: Bool { nodes[0].children.isEmpty }

    public init() {}

    public func insert(_ tokens: [Int32], rewrite: [Int32]? = nil) {
        guard !tokens.isEmpty else { return }
        var cur = 0
        for t in tokens {
            if let next = nodes[cur].children[t] {
                cur = next
            } else {
                nodes.append(Node())
                nodes[cur].children[t] = nodes.count - 1
                cur = nodes.count - 1
            }
        }
        nodes[cur].terminal = true
        if let rewrite { nodes[cur].rewrite = rewrite }
    }

    /// Build a trie from dictionary terms using the model's tokenizer.
    /// `encode` must be the same text→ids mapping the decoder emits under.
    ///
    /// `aliases` maps a canonical term to known mis-spellings/mishearings
    /// (e.g. "Fernardo" → ["Leonardo", "Linardo"]). Alias paths are boosted
    /// like the canonical and, on completion, rewrite the emitted span to the
    /// canonical form - the multi-pronunciation-variant trie from the
    /// contextual-biasing literature. Register only surfaces the user would
    /// not dictate deliberately; the rewrite is unconditional.
    public static func build(
        terms: [String],
        aliases: [String: [String]] = [:],
        encode: (String) -> [Int]
    ) -> Qwen3TokenTrie {
        let trie = Qwen3TokenTrie()
        for raw in terms {
            let term = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !term.isEmpty else { continue }
            for prefix in [" ", ""] {
                trie.insert(encode(prefix + term).map(Int32.init))
            }
            for alias in aliases[raw] ?? aliases[term] ?? [] {
                let surface = alias.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !surface.isEmpty, surface != term else { continue }
                for prefix in [" ", ""] {
                    trie.insert(
                        encode(prefix + surface).map(Int32.init),
                        rewrite: encode(prefix + term).map(Int32.init))
                }
            }
        }
        return trie
    }
}

// MARK: - Boost controller

/// Pure-Swift decision engine for trie-boosted greedy decoding. Owns the trie
/// walk, speculation bookkeeping, per-position bans, and the backtrack budget;
/// the MLX decode loop owns tensors and the single rollback snapshot.
public final class Qwen3TrieBoostController {
    /// Alias-completion splice the loop must apply to its emitted tokens:
    /// the last `spanLength` tokens (including the token just emitted) are
    /// replaced by `replacement`. The KV cache keeps the alias tokens - the
    /// forms are near-homographs, so downstream conditioning drift is
    /// negligible for word-length spans.
    public struct Rewrite: Equatable {
        public let spanLength: Int
        public let replacement: [Int32]
    }

    public enum Decision: Equatable {
        /// Emit `token`. When `snapshot` is true the loop must save
        /// (scores, cache, emitted-count) for this position BEFORE emitting -
        /// a speculation is starting here. A non-nil `rewrite` means the token
        /// completed an alias path: splice after appending.
        case emit(token: Int32, snapshot: Bool, rewrite: Rewrite?)
        /// Speculation failed: restore the snapshot, then call `decide` again
        /// on the restored scores.
        case rollback
        /// Backtrack budget exhausted: restore the snapshot; boosting is now
        /// disabled and subsequent decisions are plain greedy.
        case abort
    }

    private let trie: Qwen3TokenTrie
    private let boost: Float
    private let maxBacktracks: Int
    private let eosToken: Int32

    /// Partial term matches for the current hypothesis: trie node + how many
    /// emitted tokens this path spans (needed for alias-rewrite splices).
    private struct PathState: Equatable { var node: Int; var depth: Int }
    private var active: [PathState] = []
    private var speculating = false
    private var checkpointActive: [PathState] = []
    private var checkpointCooldown = false
    private var entryToken: Int32 = -1
    /// Tokens banned at the checkpoint position (failed speculative entries).
    private var bans: Set<Int32> = []
    private var boostingEnabled = true
    /// True while the previously-accepted token completed a term. Suppresses
    /// boosted term *starts* for one token so a completed term cannot chain
    /// straight into the next one (glossary-echo cascade guard).
    private var cooldown = false
    /// Emit counter + the emit-position bans belong to. Bans veto only the
    /// re-decision at the checkpoint position they were created for; at any
    /// other position they are inert (a banned token must never override the
    /// raw argmax elsewhere - review finding: ban leak).
    private var emitCount = 0
    private var banPosition = -1
    /// Until armed, decisions are pure greedy with no trie tracking. Used to
    /// keep the boost away from Qwen's auto-language prefix
    /// ("language XX<asr_text>") when no language is forced.
    private var armed = true

    public private(set) var backtracks = 0
    public private(set) var completions = 0
    public var isSpeculating: Bool { speculating }

    /// Disarm at init via `armed: false`, then `arm()` once the transcript
    /// region begins (after the `<asr_text>` token).
    public func setArmed(_ value: Bool) { armed = value }

    public init(
        trie: Qwen3TokenTrie,
        boost: Float,
        maxBacktracks: Int = 16,
        eosToken: Int32 = Int32(Qwen3ASRTokens.eosTokenId)
    ) {
        self.trie = trie
        self.boost = boost
        self.maxBacktracks = maxBacktracks
        self.eosToken = eosToken
    }

    /// One decode-step decision over the CPU logits for the current position.
    public func decide(scores: [Float]) -> Decision {
        guard armed else {
            // Pure greedy, no trie tracking, until armed.
            let (raw, _) = Self.pick(scores: scores, boostable: [], banned: [], boost: 0)
            emitCount += 1
            return .emit(token: raw, snapshot: false, rewrite: nil)
        }
        let boostable = currentBoostable()
        // Bans are positional: only the checkpoint re-decision they were
        // created at may be vetoed by them.
        let effectiveBans = (emitCount == banPosition) ? bans : []
        var (raw, chosen) = Self.pick(scores: scores, boostable: boostable, banned: effectiveBans, boost: boost)
        // EOS guard: when the model wants to stop and no speculation is in
        // flight, never let the bonus start a term instead - that is how
        // boosted decoding regurgitates the glossary at end of transcript.
        if !speculating && raw == eosToken && chosen != raw {
            chosen = raw
        }
        let influenced = chosen != raw
        let newActive = advanced(with: chosen)
        // Completion = deepest terminal reached by this token (longest match
        // wins when an alias and a canonical end on the same token).
        let completedState = newActive
            .filter { trie.nodes[$0.node].terminal }
            .max { $0.depth < $1.depth }
        let completed = completedState != nil

        if speculating && (chosen == eosToken || newActive.isEmpty) {
            // The path died before a term completed - revoke.
            backtracks += 1
            speculating = false
            active = checkpointActive
            cooldown = checkpointCooldown
            emitCount = banPosition  // back at the checkpoint position
            if backtracks >= maxBacktracks {
                boostingEnabled = false
                bans.removeAll()
                banPosition = -1
                return .abort
            }
            bans.insert(entryToken)
            return .rollback
        }

        var snapshot = false
        if influenced && !speculating && boostingEnabled {
            speculating = true
            checkpointActive = active
            checkpointCooldown = cooldown
            entryToken = chosen
            // Bans created by earlier failed entries at THIS position stay
            // associated with it (positional gating in `decide`).
            banPosition = emitCount
            snapshot = true
        }
        active = newActive
        var rewrite: Rewrite? = nil
        if let done = completedState {
            completions += 1
            speculating = false
            bans.removeAll(); banPosition = -1
            if let replacement = trie.nodes[done.node].rewrite {
                rewrite = Rewrite(spanLength: done.depth, replacement: replacement)
                // The splice invalidates other partial paths over that span.
                active = []
            }
        } else if !speculating {
            // Moved past any contested position without speculation.
            bans.removeAll(); banPosition = -1
        }
        cooldown = completed
        emitCount += 1
        return .emit(token: chosen, snapshot: snapshot, rewrite: rewrite)
    }

    /// Tokens eligible for the bonus at this step. During speculation only
    /// continuations of the active path(s) are boosted - new term starts wait
    /// until the current one resolves, so a single snapshot always suffices.
    /// Directly after a completed term (`cooldown`) new starts are not
    /// boosted either, so completions cannot cascade term-to-term.
    private func currentBoostable() -> Set<Int32> {
        guard boostingEnabled else { return [] }
        var out = Set<Int32>()
        for s in active { out.formUnion(trie.nodes[s.node].children.keys) }
        if !speculating && !cooldown { out.formUnion(trie.nodes[0].children.keys) }
        return out
    }

    private func advanced(with token: Int32) -> [PathState] {
        var out: [PathState] = []
        for s in active {
            if let next = trie.nodes[s.node].children[token] {
                out.append(PathState(node: next, depth: s.depth + 1))
            }
        }
        // No NEW root starts while speculating: a speculation must live or die
        // by the path its flipped entry token began. Tracking unrelated starts
        // let a dead speculation be "rescued" by a different term completing,
        // permanently committing the wrong flipped token (review finding).
        if !speculating,
           let start = trie.nodes[0].children[token],
           !out.contains(where: { $0.node == start }) {
            out.append(PathState(node: start, depth: 1))
        }
        return out
    }

    /// Raw and boosted argmax over the vocab. Bans apply only to the boosted
    /// pick; a banned token can never be the raw winner (bans only ever hold
    /// tokens that won *because* of the bonus).
    static func pick(
        scores: [Float], boostable: Set<Int32>, banned: Set<Int32>, boost: Float
    ) -> (raw: Int32, chosen: Int32) {
        var rawBest = 0
        var rawScore = -Float.infinity
        var boostedBest = 0
        var boostedScore = -Float.infinity
        let checkBans = !banned.isEmpty
        let checkBoost = !boostable.isEmpty
        for i in 0..<scores.count {
            let s = scores[i]
            if s > rawScore { rawScore = s; rawBest = i }
            let t = Int32(i)
            if checkBans && banned.contains(t) { continue }
            let b = (checkBoost && boostable.contains(t)) ? s + boost : s
            if b > boostedScore { boostedScore = b; boostedBest = i }
        }
        return (Int32(rawBest), Int32(boostedBest))
    }
}

// MARK: - Decode loop

extension Qwen3ASRModel {
    /// Greedy decode with trie-boosted token selection and snapshot/rollback
    /// revocation. Structure mirrors `generateSlow` (per-token CPU sync); the
    /// extra cost is one 600 KB logits copy per step plus re-decoded tokens
    /// after a rollback (rare, bounded by `controller`'s backtrack budget).
    static func generateTrieBoosted(
        textDecoder: QuantizedTextModel,
        initialLogits: MLXArray,
        cache initialCache: [(MLXArray, MLXArray)],
        maxTokens: Int,
        controller: Qwen3TrieBoostController,
        armAfterToken: Int32? = nil
    ) -> [Int32] {
        var generated: [Int32] = []
        guard maxTokens > 0 else { return generated }

        var cache = initialCache
        var scores = cpuScores(initialLogits)

        // Single rollback snapshot - valid because at most one speculation is
        // in flight (the controller defers new term starts while speculating).
        var snapScores: [Float] = []
        var snapCache: [(MLXArray, MLXArray)] = []
        var snapCount = 0

        let eosToken = Int32(Qwen3ASRTokens.eosTokenId)

        // Bound DECODER ITERATIONS, not the spliced output length: an alias
        // whose replacement is shorter than its surface shrinks `generated`,
        // which must not extend the decode budget (review finding). Rollback
        // re-decodes also consume budget via their re-emits.
        var steps = 0
        // Waiting-to-arm: keep the boost away from the auto-language prefix.
        var awaitingArm = armAfterToken != nil
        if awaitingArm { controller.setArmed(false) }

        while steps < maxTokens {
            switch controller.decide(scores: scores) {
            case .emit(let token, let snapshot, let rewrite):
                steps += 1
                if snapshot {
                    snapScores = scores
                    snapCache = cache
                    snapCount = generated.count
                }
                generated.append(token)
                if let rewrite, rewrite.spanLength <= generated.count {
                    generated.replaceSubrange(
                        (generated.count - rewrite.spanLength)..<generated.count,
                        with: rewrite.replacement)
                }
                if token == eosToken { return generated }
                if awaitingArm, token == armAfterToken! || steps >= 8 {
                    // Prefix over (or clearly absent) - boost from here on.
                    controller.setArmed(true)
                    awaitingArm = false
                }
                if !controller.isSpeculating && !snapCache.isEmpty {
                    // Speculation resolved - release the checkpoint KV cache
                    // instead of pinning it for the rest of the utterance.
                    snapCache = []; snapScores = []
                }

                let embed = textDecoder.embedTokens(
                    MLXArray([token]).expandedDimensions(axis: 0))
                let (hidden, newCache) = textDecoder(inputsEmbeds: embed, cache: cache)
                let lastHidden = hidden[0..., (-1)..., .ellipsis]
                let logits = textDecoder.embedTokens.asLinear(lastHidden)
                cache = newCache
                scores = cpuScores(logits)

            case .rollback, .abort:
                generated.removeLast(generated.count - snapCount)
                cache = snapCache
                scores = snapScores
            }
        }

        // Budget exhausted mid-speculation: drop the dangling partial term.
        if controller.isSpeculating && snapCount <= generated.count {
            generated.removeLast(generated.count - snapCount)
        }
        return generated
    }

    private static func cpuScores(_ logits: MLXArray) -> [Float] {
        logits.squeezed().asType(.float32).asArray(Float.self)
    }
}
