//
//  GenerationParityTests.swift
//  GraniteSpeechTests
//
//  Created by Kévin Naudin on 2026-06-03.
//
//  Phase 4g parity: the Swift greedy decoding loop (KV cache) must reproduce
//  the EXACT generated token sequence `token_ids.npy` when fed the Python
//  `inputs_embeds.npy`. This isolates the autoregressive loop + KV cache from
//  the tokenizer (validated separately in 4f).
//

import Foundation
import MLX
import MLXNN
import XCTest
@testable import GraniteSpeech

final class GenerationParityTests: XCTestCase {

    private let cases = ["case_010", "case_011", "case_012", "case_013", "case_014", "case_015"]
    private let eosTokenId = 100257 // from config_dump.json

    func testGreedyTokenParity() throws {
        let fixtures = Numpy.fixturesRoot
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: fixtures.path),
            "Fixtures not found at \(fixtures.path)."
        )
        guard let modelDir = GraniteWeights.resolveModelDirectory() else {
            throw XCTSkip("Granite model dir not found. Set GRANITE_MODEL_DIR or download the model.")
        }

        let config = GraniteSpeechConfig.granite41_2b
        let decoder = GraniteForCausalLM(config.textConfig)
        let weights = try GraniteWeights.loadSanitized(from: modelDir)
        decoder.update(parameters: ModuleParameters.unflattened(
            GraniteWeights.subset(weights, prefix: "language_model.")))
        eval(decoder)

        var allExact = true
        for name in cases {
            let inputsEmbeds = try Numpy.load("\(name)/inputs_embeds.npy").mlx()
            let expected = try Numpy.load("\(name)/token_ids.npy").int32s().map { Int($0) }

            let got = decoder.generate(
                inputsEmbeds: inputsEmbeds, eosTokenId: eosTokenId, maxTokens: expected.count + 8)

            let exact = got == expected
            allExact = allExact && exact
            let prefixMatch = zip(got, expected).prefix { $0 == $1 }.count
            print("[\(name)] exact=\(exact) got=\(got.count) expected=\(expected.count) matchedPrefix=\(prefixMatch)/\(expected.count)")
            XCTAssertEqual(got, expected, "[\(name)] generated token sequence mismatch")
        }
        print("=== generation parity: all exact = \(allExact) ===")
    }
}
