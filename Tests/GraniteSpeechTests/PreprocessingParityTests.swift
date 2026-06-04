//
//  PreprocessingParityTests.swift
//  GraniteSpeechTests
//
//  Created by Kévin Naudin on 2026-06-03.
//
//  Phase 4b parity: the Swift `GranitePreprocessor` must reproduce the Python
//  `mel_features.npy` fixtures within tolerance (numerical floor of fp32 STFT).
//

import Foundation
import MLX
import XCTest
@testable import GraniteSpeech

final class PreprocessingParityTests: XCTestCase {

    private let cases = ["case_010", "case_011", "case_012", "case_013", "case_014", "case_015"]
    private let tolerance: Float = 1e-4

    func testMelFeaturesParity() throws {
        let root = Numpy.fixturesRoot
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: root.path),
            "Fixtures not found at \(root.path). Run dump_granite_fixtures.py or set GRANITE_FIXTURES."
        )

        let pre = GranitePreprocessor()
        var maxDiffOverall: Float = 0

        for name in cases {
            let audio = try Numpy.load("\(name)/audio_input.npy").mlx()
            let expected = try Numpy.load("\(name)/mel_features.npy")

            let got = pre.melFeatures(audio)
            got.eval()

            XCTAssertEqual(got.shape, expected.shape, "[\(name)] shape mismatch")

            let exp = expected.mlx()
            let diff = abs(got - exp).max().item(Float.self)
            maxDiffOverall = Swift.max(maxDiffOverall, diff)
            print("[\(name)] shape=\(got.shape) maxAbsDiff=\(diff)")

            XCTAssertLessThan(diff, tolerance, "[\(name)] mel parity exceeded tolerance")
        }

        print("=== mel parity: worst maxAbsDiff over \(cases.count) cases = \(maxDiffOverall) ===")
    }
}
