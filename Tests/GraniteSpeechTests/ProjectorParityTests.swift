//
//  ProjectorParityTests.swift
//  GraniteSpeechTests
//
//  Created by Kévin Naudin on 2026-06-03.
//
//  Phase 4d parity: the Swift `EncoderProjector` must reproduce
//  `projector_out.npy`. Fed the Python `encoder_out.npy` fixture so the test
//  isolates the projector from the encoder.
//

import Foundation
import MLX
import MLXNN
import XCTest
@testable import GraniteSpeech

final class ProjectorParityTests: XCTestCase {

    private let cases = ["case_010", "case_011", "case_014"]
    private let relTolerance: Float = 0.02

    func testProjectorParity() throws {
        let fixtures = Numpy.fixturesRoot
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: fixtures.path),
            "Fixtures not found at \(fixtures.path)."
        )
        guard let modelDir = GraniteWeights.resolveModelDirectory() else {
            throw XCTSkip("Granite model dir not found. Set GRANITE_MODEL_DIR or download the model.")
        }

        let config = GraniteSpeechConfig.granite41_2b
        let projector = EncoderProjector(config)

        let weights = try GraniteWeights.loadSanitized(from: modelDir)
        let projWeights = GraniteWeights.subset(weights, prefix: "projector.")
        projector.update(parameters: ModuleParameters.unflattened(projWeights))
        eval(projector)

        var worstRel: Float = 0
        for name in cases {
            let encoderOut = try Numpy.load("\(name)/encoder_out.npy").mlx()
            let got = projector(encoderOut)
            eval(got)

            let expected = try Numpy.load("\(name)/projector_out.npy")
            XCTAssertEqual(got.shape, expected.shape, "[\(name)] projector_out shape")
            let exp = expected.mlx()
            let maxAbs = abs(exp).max().item(Float.self)
            let d = abs(got - exp).max().item(Float.self)
            let rel = d / Swift.max(maxAbs, 1e-6)
            worstRel = Swift.max(worstRel, rel)
            print("[\(name)] projector_out shape=\(got.shape) rel=\(rel) (abs=\(d) scale=\(maxAbs))")
        }

        print("=== projector parity: worst rel=\(worstRel) (tol \(relTolerance)) ===")
        XCTAssertLessThan(worstRel, relTolerance, "projector relative parity exceeded tolerance")
    }
}
