//
//  DecoderParityTests.swift
//  GraniteSpeechTests
//
//  Created by Kévin Naudin on 2026-06-03.
//
//  Phase 4e parity: the Swift `GraniteForCausalLM` must reproduce
//  `lm_logits_step0.npy` (the distribution over the first generated token) when
//  fed the Python `inputs_embeds.npy`. The decisive check is argmax == the
//  first real generated id (token_ids[0]); logits are also compared relatively.
//

import Foundation
import MLX
import MLXNN
import XCTest
@testable import GraniteSpeech

final class DecoderParityTests: XCTestCase {

    private let cases = ["case_010", "case_011", "case_014"]
    private let relTolerance: Float = 0.05

    func testDecoderParity() throws {
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
        let lmWeights = GraniteWeights.subset(weights, prefix: "language_model.")
        decoder.update(parameters: ModuleParameters.unflattened(lmWeights))
        eval(decoder)

        var worstRel: Float = 0
        for name in cases {
            let inputsEmbeds = try Numpy.load("\(name)/inputs_embeds.npy").mlx()
            let logits = decoder(inputsEmbeds: inputsEmbeds) // (1, L, vocab)
            let last = logits[0, -1] // (vocab,)
            eval(last)

            let gotArgmax = argMax(last).item(Int32.self)
            let expectedFirst = try Numpy.load("\(name)/token_ids.npy").int32s().first ?? -1
            XCTAssertEqual(gotArgmax, expectedFirst, "[\(name)] first-token argmax mismatch")

            let expected = try Numpy.load("\(name)/lm_logits_step0.npy").mlx()
            let maxAbs = abs(expected).max().item(Float.self)
            let d = abs(last - expected).max().item(Float.self)
            let rel = d / Swift.max(maxAbs, 1e-6)
            worstRel = Swift.max(worstRel, rel)
            print("[\(name)] argmax got=\(gotArgmax) expected=\(expectedFirst) | logits rel=\(rel) (abs=\(d) scale=\(maxAbs))")
        }

        print("=== decoder parity: worst logits rel=\(worstRel) (tol \(relTolerance)) ===")
        XCTAssertLessThan(worstRel, relTolerance, "decoder logits relative parity exceeded tolerance")
    }
}
