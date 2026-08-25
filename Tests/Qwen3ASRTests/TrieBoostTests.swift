//
//  TrieBoostTests.swift
//  Qwen3ASRTests
//
//  Pure-CPU tests for the trie-boost decision engine: trie construction,
//  boosted argmax, speculation lifecycle (snapshot → complete / rollback),
//  ban bookkeeping, and the backtrack budget. No model, no GPU.
//
//  Toy vocabulary used throughout (token ids):
//    0 " the"   1 " C"    2 "er"    3 "ity"   4 " S"
//    5 " mood"  6 "."     7 EOS     8 " On"   9 "it"
//  Term "Cerity"  → [1, 2, 3] (leading-space variant)
//  Term "Onit"    → [8, 9]
//

import XCTest
@testable import Qwen3ASR

final class TrieBoostTests: XCTestCase {

    private let eos: Int32 = 7
    private let vocab = 10

    /// Scores with one clear winner and a given runner-up gap.
    private func scores(winner: Int, _ base: Float = -10, gap: [Int: Float] = [:]) -> [Float] {
        var s = [Float](repeating: base, count: vocab)
        s[winner] = -1
        for (tok, v) in gap { s[tok] = v }
        return s
    }

    private func makeTrie() -> Qwen3TokenTrie {
        let trie = Qwen3TokenTrie()
        trie.insert([1, 2, 3])  // " C" "er" "ity"
        trie.insert([8, 9])     // " On" "it"
        return trie
    }

    private func controller(boost: Float = 4, maxBacktracks: Int = 16) -> Qwen3TrieBoostController {
        Qwen3TrieBoostController(
            trie: makeTrie(), boost: boost, maxBacktracks: maxBacktracks, eosToken: eos)
    }

    // MARK: - Trie construction

    func testBuildInsertsSpaceAndBareVariants() {
        // Encoder stub: " X" → [1], "X" → [2], anything else char count sentinel.
        let trie = Qwen3TokenTrie.build(terms: ["X"]) { text in
            text == " X" ? [1] : [2]
        }
        XCTAssertEqual(Set(trie.nodes[0].children.keys), [1, 2])
        XCTAssertTrue(trie.nodes[trie.nodes[0].children[1]!].terminal)
        XCTAssertTrue(trie.nodes[trie.nodes[0].children[2]!].terminal)
    }

    func testBuildSkipsEmptyTerms() {
        let trie = Qwen3TokenTrie.build(terms: ["  ", ""]) { _ in [1] }
        XCTAssertTrue(trie.isEmpty)
    }

    // MARK: - pick()

    func testPickNoBoostIsArgmax() {
        let s = scores(winner: 5)
        let (raw, chosen) = Qwen3TrieBoostController.pick(
            scores: s, boostable: [], banned: [], boost: 4)
        XCTAssertEqual(raw, 5)
        XCTAssertEqual(chosen, 5)
    }

    func testPickBoostFlipsCloseDecision() {
        // Token 4 (" S") wins raw by 2; boost +4 on token 1 (" C") flips it.
        let s = scores(winner: 4, gap: [1: -3])
        let (raw, chosen) = Qwen3TrieBoostController.pick(
            scores: s, boostable: [1], banned: [], boost: 4)
        XCTAssertEqual(raw, 4)
        XCTAssertEqual(chosen, 1)
    }

    func testPickBoostCannotFlipLargeGap() {
        // Gap of 8 > boost of 4 — acoustics win.
        let s = scores(winner: 4, gap: [1: -9])
        let (_, chosen) = Qwen3TrieBoostController.pick(
            scores: s, boostable: [1], banned: [], boost: 4)
        XCTAssertEqual(chosen, 4)
    }

    func testPickBanExcludesToken() {
        let s = scores(winner: 4, gap: [1: -3])
        let (_, chosen) = Qwen3TrieBoostController.pick(
            scores: s, boostable: [1], banned: [1], boost: 4)
        XCTAssertEqual(chosen, 4)
    }

    // MARK: - Uninfluenced decoding

    func testPlainTokensEmitWithoutSnapshot() {
        let c = controller()
        let d = c.decide(scores: scores(winner: 0))
        XCTAssertEqual(d, .emit(token: 0, snapshot: false, rewrite: nil))
        XCTAssertFalse(c.isSpeculating)
    }

    // MARK: - Successful speculation

    func testBoostedTermCompletesAndClearsSpeculation() {
        let c = controller()
        // Step 1: " S" wins raw by 2, " C" flips via boost → speculation starts.
        var d = c.decide(scores: scores(winner: 4, gap: [1: -3]))
        XCTAssertEqual(d, .emit(token: 1, snapshot: true, rewrite: nil))
        XCTAssertTrue(c.isSpeculating)
        // Step 2: "er" continues (also boosted, close call).
        d = c.decide(scores: scores(winner: 2))
        XCTAssertEqual(d, .emit(token: 2, snapshot: false, rewrite: nil))
        XCTAssertTrue(c.isSpeculating)
        // Step 3: "ity" completes the term → speculation over.
        d = c.decide(scores: scores(winner: 3))
        XCTAssertEqual(d, .emit(token: 3, snapshot: false, rewrite: nil))
        XCTAssertFalse(c.isSpeculating)
        XCTAssertEqual(c.completions, 1)
        XCTAssertEqual(c.backtracks, 0)
    }

    func testBoostAppliesToContinuationMidPath() {
        let c = controller()
        _ = c.decide(scores: scores(winner: 4, gap: [1: -3]))  // enter " C"
        // "er" trails "." by 2 raw; continuation boost should flip it.
        let d = c.decide(scores: scores(winner: 6, gap: [2: -3]))
        XCTAssertEqual(d, .emit(token: 2, snapshot: false, rewrite: nil))
    }

    // MARK: - Failed speculation

    func testDeadPathRollsBackAndBansEntry() {
        let c = controller()
        var d = c.decide(scores: scores(winner: 4, gap: [1: -3]))  // speculative " C"
        XCTAssertEqual(d, .emit(token: 1, snapshot: true, rewrite: nil))
        // Next step: "." wins by a mile; no continuation → path dies.
        d = c.decide(scores: scores(winner: 6))
        XCTAssertEqual(d, .rollback)
        XCTAssertFalse(c.isSpeculating)
        XCTAssertEqual(c.backtracks, 1)
        // Re-decision at the restored position: entry token 1 is banned,
        // so the raw winner " S" goes through.
        d = c.decide(scores: scores(winner: 4, gap: [1: -3]))
        XCTAssertEqual(d, .emit(token: 4, snapshot: false, rewrite: nil))
        XCTAssertFalse(c.isSpeculating)
    }

    func testEosDuringSpeculationRollsBack() {
        let c = controller()
        _ = c.decide(scores: scores(winner: 4, gap: [1: -3]))  // speculative " C"
        let d = c.decide(scores: scores(winner: Int(eos)))
        XCTAssertEqual(d, .rollback)
    }

    func testBansClearAfterMovingPast() {
        let c = controller()
        _ = c.decide(scores: scores(winner: 4, gap: [1: -3]))   // speculate
        _ = c.decide(scores: scores(winner: 6))                  // rollback, ban 1
        _ = c.decide(scores: scores(winner: 4, gap: [1: -3]))   // emit raw " S"
        // New position: token 1 must be boostable again (ban was positional).
        let d = c.decide(scores: scores(winner: 4, gap: [1: -3]))
        XCTAssertEqual(d, .emit(token: 1, snapshot: true, rewrite: nil))
    }

    func testSecondEntryCanSpeculateAfterFirstBanned() {
        let c = controller()
        _ = c.decide(scores: scores(winner: 4, gap: [1: -3]))   // speculate " C"
        _ = c.decide(scores: scores(winner: 6))                  // rollback, ban 1
        // Restored position now has " On" (8) close behind: with 1 banned,
        // 8 should win via boost → a fresh speculation.
        let d = c.decide(scores: scores(winner: 4, gap: [1: -3, 8: -3]))
        XCTAssertEqual(d, .emit(token: 8, snapshot: true, rewrite: nil))
        XCTAssertTrue(c.isSpeculating)
    }

    // MARK: - Backtrack budget

    func testBudgetExhaustionAbortsAndDisablesBoosting() {
        let c = controller(maxBacktracks: 1)
        _ = c.decide(scores: scores(winner: 4, gap: [1: -3]))   // speculate
        let d = c.decide(scores: scores(winner: 6))              // dies → budget hit
        XCTAssertEqual(d, .abort)
        // Boosting disabled: the same close call now resolves raw.
        let d2 = c.decide(scores: scores(winner: 4, gap: [1: -3]))
        XCTAssertEqual(d2, .emit(token: 4, snapshot: false, rewrite: nil))
    }

    // MARK: - Alias rewrite

    /// Toy alias: "Leonardo" = [4, 2, 3] rewrites to "Lenardo" = [1, 2, 3].
    private func aliasController(boost: Float = 4) -> Qwen3TrieBoostController {
        let trie = Qwen3TokenTrie()
        trie.insert([1, 2, 3])                       // canonical
        trie.insert([4, 2, 3], rewrite: [1, 2, 3])   // alias → canonical
        return Qwen3TrieBoostController(
            trie: trie, boost: boost, maxBacktracks: 16, eosToken: eos)
    }

    func testNaturalAliasCompletionRewrites() {
        let c = aliasController()
        // Model freely produces the alias (raw wins each step — no boost).
        var d = c.decide(scores: scores(winner: 4))
        XCTAssertEqual(d, .emit(token: 4, snapshot: false, rewrite: nil))
        d = c.decide(scores: scores(winner: 2))
        XCTAssertEqual(d, .emit(token: 2, snapshot: false, rewrite: nil))
        d = c.decide(scores: scores(winner: 3))
        XCTAssertEqual(d, .emit(
            token: 3, snapshot: false,
            rewrite: .init(spanLength: 3, replacement: [1, 2, 3])))
        XCTAssertEqual(c.completions, 1)
    }

    func testBoostedAliasCompletionRewrites() {
        let c = aliasController()
        // Alias start trails by 2, flips via boost, completes → rewrite.
        var d = c.decide(scores: scores(winner: 5, gap: [4: -3]))
        XCTAssertEqual(d, .emit(token: 4, snapshot: true, rewrite: nil))
        _ = c.decide(scores: scores(winner: 2))
        d = c.decide(scores: scores(winner: 3))
        XCTAssertEqual(d, .emit(
            token: 3, snapshot: false,
            rewrite: .init(spanLength: 3, replacement: [1, 2, 3])))
        XCTAssertFalse(c.isSpeculating)
    }

    func testCanonicalCompletionHasNoRewrite() {
        let c = aliasController()
        _ = c.decide(scores: scores(winner: 1))
        _ = c.decide(scores: scores(winner: 2))
        let d = c.decide(scores: scores(winner: 3))
        XCTAssertEqual(d, .emit(token: 3, snapshot: false, rewrite: nil))
    }

    func testBuildRegistersAliasWithRewrite() {
        // encode stub: " Lenardo"→[1,2], "Lenardo"→[11,12], " Leonardo"→[4,2],
        // "Leonardo"→[14,12].
        let trie = Qwen3TokenTrie.build(
            terms: ["Lenardo"], aliases: ["Lenardo": ["Leonardo"]]
        ) { text in
            switch text {
            case " Lenardo": return [1, 2]
            case "Lenardo": return [11, 12]
            case " Leonardo": return [4, 2]
            default: return [14, 12]
            }
        }
        // Walk " Leonardo" and confirm the terminal rewrites to " Lenardo".
        let n4 = trie.nodes[0].children[4]!
        let n42 = trie.nodes[n4].children[2]!
        XCTAssertTrue(trie.nodes[n42].terminal)
        XCTAssertEqual(trie.nodes[n42].rewrite, [1, 2])
        // Canonical path has no rewrite.
        let n1 = trie.nodes[0].children[1]!
        let n12 = trie.nodes[n1].children[2]!
        XCTAssertTrue(trie.nodes[n12].terminal)
        XCTAssertNil(trie.nodes[n12].rewrite)
    }

    // MARK: - Review-finding regressions

    func testSpeculationNotRescuedByUnrelatedRootStart() {
        // Boost flips " C" (term A entry); next raw token " On" (8) starts a
        // DIFFERENT term. The speculation must die (rollback), not survive by
        // adopting the unrelated path — otherwise the wrong flipped token is
        // permanently committed when "Onit" completes.
        let c = controller()
        var d = c.decide(scores: scores(winner: 4, gap: [1: -3]))
        XCTAssertEqual(d, .emit(token: 1, snapshot: true, rewrite: nil))
        d = c.decide(scores: scores(winner: 8))
        XCTAssertEqual(d, .rollback)
    }

    func testBansArePositional() {
        let c = controller()
        // Spec-A on token 1 fails; ban(1) at checkpoint.
        _ = c.decide(scores: scores(winner: 4, gap: [1: -3]))
        _ = c.decide(scores: scores(winner: 6))
        // Re-decision: 1 banned here; 8 flips instead → spec-B.
        var d = c.decide(scores: scores(winner: 4, gap: [1: -3, 8: -3]))
        XCTAssertEqual(d, .emit(token: 8, snapshot: true, rewrite: nil))
        // Deeper position: raw winner is token 1. The old code vetoed it via
        // the leaked ban and emitted second-best (corruption). Correct: 1 is
        // pickable; it kills spec-B (not a continuation) → rollback.
        d = c.decide(scores: scores(winner: 1))
        XCTAssertEqual(d, .rollback)
        // Back at the checkpoint both entries are banned → raw goes through.
        d = c.decide(scores: scores(winner: 4, gap: [1: -3, 8: -3]))
        XCTAssertEqual(d, .emit(token: 4, snapshot: false, rewrite: nil))
    }

    func testDisarmedControllerIsPureGreedy() {
        let c = controller()
        c.setArmed(false)
        // A flippable close call must NOT flip while disarmed.
        var d = c.decide(scores: scores(winner: 4, gap: [1: -3]))
        XCTAssertEqual(d, .emit(token: 4, snapshot: false, rewrite: nil))
        XCTAssertFalse(c.isSpeculating)
        c.setArmed(true)
        d = c.decide(scores: scores(winner: 4, gap: [1: -3]))
        XCTAssertEqual(d, .emit(token: 1, snapshot: true, rewrite: nil))
    }

    // MARK: - Echo guards

    func testNoBoostedStartAgainstEos() {
        let c = controller()
        // Model wants EOS; " C" (1) is close behind. Boost must NOT flip it.
        let d = c.decide(scores: scores(winner: Int(eos), gap: [1: -3]))
        XCTAssertEqual(d, .emit(token: eos, snapshot: false, rewrite: nil))
        XCTAssertFalse(c.isSpeculating)
    }

    func testCooldownBlocksImmediateNextTermStart() {
        let c = controller()
        // Complete "Onit" via boost: enter " On" (8), finish "it" (9).
        _ = c.decide(scores: scores(winner: 4, gap: [8: -3]))
        var d = c.decide(scores: scores(winner: 9))
        XCTAssertEqual(d, .emit(token: 9, snapshot: false, rewrite: nil))
        XCTAssertEqual(c.completions, 1)
        // Immediately after the completion, " C" (1) trails by 2 — without the
        // cooldown the boost would flip it and chain a second term.
        d = c.decide(scores: scores(winner: 4, gap: [1: -3]))
        XCTAssertEqual(d, .emit(token: 4, snapshot: false, rewrite: nil))
        XCTAssertFalse(c.isSpeculating)
        // One token later the cooldown has cleared: starts boost again.
        d = c.decide(scores: scores(winner: 4, gap: [1: -3]))
        XCTAssertEqual(d, .emit(token: 1, snapshot: true, rewrite: nil))
    }

    // MARK: - Natural (unboosted) trie traversal

    func testNaturalPathDeathDoesNotRollBack() {
        let c = controller()
        // " C" wins RAW (no boost influence) → no speculation.
        var d = c.decide(scores: scores(winner: 1))
        XCTAssertEqual(d, .emit(token: 1, snapshot: false, rewrite: nil))
        XCTAssertFalse(c.isSpeculating)
        // Path dies naturally → plain emit, no rollback.
        d = c.decide(scores: scores(winner: 6))
        XCTAssertEqual(d, .emit(token: 6, snapshot: false, rewrite: nil))
        XCTAssertEqual(c.backtracks, 0)
    }
}
