#if canImport(CoreML)
import AudioCommon
import CoreML
import Foundation

extension WeSpeakerModel {

    /// Run CoreML inference to extract a 256-dim speaker embedding.
    ///
    /// Pads the mel spectrogram to the nearest enumerated shape (required by
    /// the CoreML model which uses EnumeratedShapes for the time dimension),
    /// runs prediction, and extracts the embedding.
    ///
    /// - Parameters:
    ///   - melSpec: flat Float array of log-mel features `[nFrames * 80]`
    ///   - nFrames: number of mel frames
    /// - Returns: 256-dim L2-normalized speaker embedding
    func embedCoreML(melSpec: [Float], nFrames: Int) throws -> [Float] {
        guard let model = coremlModel else {
            throw AudioModelError.inferenceFailed(
                operation: "SpeakerEmbedding", reason: "CoreML model not loaded")
        }

        // Find nearest enumerated length >= nFrames
        let targetLength = Self.enumeratedMelLengths.first { $0 >= nFrames }
            ?? Self.enumeratedMelLengths.last!

        // Create input: [1, targetLength, 80] float16
        // The CoreML model internally permutes (T,80) → (80,T) to match
        // the trained weight orientation (freq as height, time as width).
        let melArray = try MLMultiArray(
            shape: [1, targetLength as NSNumber, 80],
            dataType: .float16
        )
        // `Float16` is unavailable on x86_64 macOS - see CoreMLSileroInference.
        // Fill with mel data (row-major: frame-major, 80 mels per frame), then
        // zero-pad the remaining frames.
        //
        // `baseAddress` is nil for an empty array, so a degenerate segment
        // (`nFrames == 0`) takes the plain zero-fill instead of force-unwrapping
        // - the loops this replaced were a harmless no-op in that case.
        let copyCount = min(nFrames, targetLength) * 80
        if copyCount > 0 {
            melSpec.withUnsafeBufferPointer {
                Float16Storage.write($0.baseAddress!, count: copyCount, intoPadded: melArray)
            }
        } else {
            Float16Storage.zeroFill(melArray)
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "mel": MLFeatureValue(multiArray: melArray),
        ])

        let result = try model.prediction(from: input)

        // Extract "embedding" output: [1, 256]
        guard let embArray = result.featureValue(for: "embedding")?.multiArrayValue else {
            throw AudioModelError.inferenceFailed(
                operation: "SpeakerEmbedding", reason: "Missing 'embedding' output")
        }

        // Read 256 float16 values
        var embedding = Float16Storage.read(embArray, count: 256)

        // L2 normalize
        let norm = sqrt(embedding.reduce(Float(0)) { $0 + $1 * $1 })
        if norm > 1e-10 {
            for i in 0..<256 { embedding[i] /= norm }
        }

        return embedding
    }
}
#endif
