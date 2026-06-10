//
//  WeightLoading.swift
//  GraniteSpeech
//
//  Created by Onit on 2026-06-03.
//
//  Loads the 3 safetensors shards of ibm-granite/granite-speech-4.1-2b and
//  applies the sanitize that the upstream mlx-audio `Model.sanitize` gets wrong
//  for kernel-1 convs: ALL up_conv/down_conv/depth_conv 3D weights must be
//  transposed (out, in, kernel) -> (out, kernel, in) for MLX Conv1d, not only
//  those where shape[-1] > shape[-2]. See tasks/lessons.md.
//

import Foundation
import MLX
import MLXCommon

public enum GraniteWeights {

    /// Load every safetensors shard in `directory` and apply the corrected
    /// sanitize. Keys keep their full namespaced form (`encoder.*`,
    /// `projector.*`, `language_model.*`).
    public static func loadSanitized(from directory: URL) throws -> [String: MLXArray] {
        let raw = try CommonWeightLoader.loadAllSafetensors(from: directory)
        return sanitize(raw)
    }

    /// Subset of `weights` whose keys start with `prefix`, with the prefix
    /// stripped (e.g. `encoder.layers.0.…` -> `layers.0.…`).
    public static func subset(_ weights: [String: MLXArray], prefix: String) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (k, v) in weights where k.hasPrefix(prefix) {
            out[String(k.dropFirst(prefix.count))] = v
        }
        return out
    }

    static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        let alreadyConverted = weights.keys.contains { $0.contains("scales") }
        let convNames = ["up_conv", "down_conv", "depth_conv"]

        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)
        for (k, v) in weights {
            if k.contains("num_batches_tracked") { continue }
            if !alreadyConverted,
               k.contains("weight"),
               v.ndim == 3,
               convNames.contains(where: { k.contains($0) }) {
                // (out, in, kernel) -> (out, kernel, in)
                out[k] = v.transposed(0, 2, 1)
            } else {
                out[k] = v
            }
        }
        return out
    }

    /// Resolve the local model directory (the safetensors live here) for tests:
    /// `$GRANITE_MODEL_DIR`, else the most recent HuggingFace cache snapshot of
    /// `ibm-granite/granite-speech-4.1-2b`.
    public static func resolveModelDirectory() -> URL? {
        let fm = FileManager.default
        if let env = ProcessInfo.processInfo.environment["GRANITE_MODEL_DIR"] {
            return URL(fileURLWithPath: env, isDirectory: true)
        }
        let home = fm.homeDirectoryForCurrentUser
        let snapshots = home
            .appendingPathComponent(
                ".cache/huggingface/hub/models--ibm-granite--granite-speech-4.1-2b/snapshots",
                isDirectory: true)
        guard let entries = try? fm.contentsOfDirectory(
            at: snapshots, includingPropertiesForKeys: [.contentModificationDateKey]
        ) else { return nil }
        // newest snapshot dir that actually contains safetensors
        return entries
            .filter { dir in
                ((try? fm.contentsOfDirectory(atPath: dir.path)) ?? [])
                    .contains { $0.hasSuffix(".safetensors") }
            }
            .sorted { a, b in
                let da = (try? a.resourceValues(forKeys: [.contentModificationDateKey]))?
                    .contentModificationDate ?? .distantPast
                let db = (try? b.resourceValues(forKeys: [.contentModificationDateKey]))?
                    .contentModificationDate ?? .distantPast
                return da > db
            }
            .first
    }
}
