//
//  AudioPreprocessing.swift
//  GraniteSpeech
//
//  Created by Onit on 2026-06-03.
//
//  Port of mlx-audio `granite_speech.Model._extract_features` (Phase 4b).
//  Mirrors the reference 1:1 to stay numerically aligned with the Python
//  fixtures: STFT(n_fft=512, hop=160, Hann(400) periodic centered in 512,
//  center=reflect) -> |.|^2 -> 80 htk mels -> log10/clip/normalize ->
//  frame-stack x2 -> (1, T, 160). Validated against `mel_features.npy`.
//

import Foundation
import MLX
import MLXFFT

public struct GranitePreprocessor: Sendable {
    public let nFFT = 512
    public let winLength = 400
    public let hopLength = 160
    public let nMels = 80
    public let sampleRate = 16_000

    /// Hann(400) periodic, centered in a 512 buffer (pad 56 each side).
    private let windowPadded: MLXArray
    /// Transposed mel filterbank `mel_fb.T`, shape (nFreqs=257, nMels=80).
    private let melFilterbankT: MLXArray

    public init() {
        // hanning(400, periodic=True): denom = size (not size-1)
        var win = [Float](repeating: 0, count: nFFT)
        let padLeft = (nFFT - winLength) / 2 // 56
        for n in 0 ..< winLength {
            win[padLeft + n] = Float(0.5 * (1.0 - cos(2.0 * Double.pi * Double(n) / Double(winLength))))
        }
        windowPadded = MLXArray(win)

        melFilterbankT = Self.makeMelFilterbankT(
            sampleRate: sampleRate, nFFT: nFFT, nMels: nMels
        )
    }

    /// Build `mel_filters(htk).T` as a constant (nFreqs, nMels) matrix.
    /// Computed in Double for determinism, then cast to float32 (matches the
    /// Python float32 filterbank within ~1e-6).
    private static func makeMelFilterbankT(sampleRate: Int, nFFT: Int, nMels: Int) -> MLXArray {
        let nFreqs = nFFT / 2 + 1 // 257
        let fMax = Double(sampleRate / 2) // 8000 (sample_rate // 2)

        func hzToMel(_ f: Double) -> Double { 2595.0 * log10(1.0 + f / 700.0) }
        func melToHz(_ m: Double) -> Double { 700.0 * (pow(10.0, m / 2595.0) - 1.0) }

        func linspace(_ a: Double, _ b: Double, _ n: Int) -> [Double] {
            guard n > 1 else { return [a] }
            return (0 ..< n).map { a + Double($0) * (b - a) / Double(n - 1) }
        }

        let allFreqs = linspace(0, fMax, nFreqs)        // (257,)
        let mMin = hzToMel(0)
        let mMax = hzToMel(fMax)
        let mPts = linspace(mMin, mMax, nMels + 2)       // (82,)
        let fPts = mPts.map(melToHz)                     // (82,)
        var fDiff = [Double](repeating: 0, count: nMels + 1)
        for j in 0 ..< (nMels + 1) { fDiff[j] = fPts[j + 1] - fPts[j] }

        // fbT[i][j] = max(0, min(down, up)), shape (nFreqs, nMels)
        var flat = [Float](repeating: 0, count: nFreqs * nMels)
        for i in 0 ..< nFreqs {
            for j in 0 ..< nMels {
                let down = -(fPts[j] - allFreqs[i]) / fDiff[j]
                let up = (fPts[j + 2] - allFreqs[i]) / fDiff[j + 1]
                flat[i * nMels + j] = Float(max(0.0, min(down, up)))
            }
        }
        return MLXArray(flat).reshaped([nFreqs, nMels])
    }

    /// Reflect-pad a 1D signal by `p` (mirrors mlx-audio's hand-rolled reflect:
    /// prefix = x[1:p+1][::-1], suffix = x[-(p+1):-1][::-1]).
    private func reflectPad(_ x: MLXArray, _ p: Int) -> MLXArray {
        let l = x.dim(0)
        let prefixIdx = MLXArray((0 ..< p).map { Int32(p - $0) })       // [p, p-1, ..., 1]
        let suffixIdx = MLXArray((0 ..< p).map { Int32(l - 2 - $0) })   // [l-2, ..., l-p-1]
        let prefix = take(x, prefixIdx, axis: 0)
        let suffix = take(x, suffixIdx, axis: 0)
        return concatenated([prefix, x, suffix], axis: 0)
    }

    /// Compute log-mel encoder input features `(1, T, 160)` from a 16 kHz mono
    /// waveform. `T` = floor(numFrames / 2) (frame-stack x2).
    public func melFeatures(_ audio: MLXArray) -> MLXArray {
        let audio1d = audio.reshaped([-1]).asType(.float32)

        // STFT with center=True (reflect pad by n_fft//2) + explicit framing.
        let padded = reflectPad(audio1d, nFFT / 2)
        let numFrames = 1 + (padded.dim(0) - nFFT) / hopLength
        let frames = asStrided(
            padded, [numFrames, nFFT], strides: [hopLength, 1]
        )
        let spec = MLXFFT.rfft(frames * windowPadded) // (numFrames, 257) complex
        let power = square(abs(spec))                 // |.|^2

        var logmel = log10(clip(matmul(power, melFilterbankT), min: 1e-10)) // (numFrames, 80)
        let mxVal = logmel.max()
        logmel = maximum(logmel, mxVal - 8.0) / 4.0 + 1.0

        // drop trailing frame if odd, then frame-stack x2 -> 160 dims
        let evenFrames = numFrames - (numFrames % 2)
        if evenFrames != numFrames {
            logmel = logmel[0 ..< evenFrames]
        }
        let encoderInput = logmel.reshaped([-1, 2 * nMels]) // (T, 160)
        return expandedDimensions(encoderInput, axis: 0)    // (1, T, 160)
    }
}
