#if canImport(CoreML)
import AudioCommon
import CoreML
import Foundation

extension SileroVADModel {

    /// Run CoreML inference for one 576-sample chunk (64 context + 512 new).
    ///
    /// Creates input MLMultiArrays in float16, runs prediction, and updates
    /// the internal LSTM h/c state for the next chunk.
    ///
    /// - Parameter fullSamples: 576 Float32 samples (context prepended)
    /// - Returns: speech probability in `[0, 1]`
    func processChunkCoreML(_ fullSamples: [Float]) throws -> Float {
        guard let model = coremlModel else {
            throw AudioModelError.inferenceFailed(
                operation: "VAD", reason: "CoreML model not loaded")
        }

        // Create audio input: [1, 1, 576] float16.
        // `Float16` is unavailable on x86_64 macOS, so the conversion goes
        // through Accelerate rather than a typed pointer - one vImage call
        // instead of the previous per-element loop.
        let audioArray = try MLMultiArray(shape: [1, 1, 576], dataType: .float16)
        fullSamples.withUnsafeBufferPointer {
            Float16Storage.write($0.baseAddress!, count: 576, intoPadded: audioArray)
        }

        // Initialize h/c to zeros on first call
        if coremlH == nil {
            coremlH = try MLMultiArray(shape: [1, 1, 128], dataType: .float16)
            coremlC = try MLMultiArray(shape: [1, 1, 128], dataType: .float16)
            Float16Storage.zeroFill(coremlH!)
            Float16Storage.zeroFill(coremlC!)
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "audio": MLFeatureValue(multiArray: audioArray),
            "h": MLFeatureValue(multiArray: coremlH!),
            "c": MLFeatureValue(multiArray: coremlC!),
        ])

        let result = try model.prediction(from: input)

        // Update LSTM state
        coremlH = result.featureValue(for: "h_out")!.multiArrayValue!
        coremlC = result.featureValue(for: "c_out")!.multiArrayValue!

        // Extract probability scalar
        let probArray = result.featureValue(for: "probability")!.multiArrayValue!
        return probArray[0].floatValue
    }
}
#endif
