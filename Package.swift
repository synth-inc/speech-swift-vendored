// swift-tools-version: 5.10
//
//  Package.swift
//  speech-swift-vendored
//
//  Vendored subset of https://github.com/soniqo/speech-swift (Apache-2.0).
//  Only the targets required by Qwen3-ASR inference are kept:
//    Qwen3ASR, AudioCommon, MLXCommon, SpeechVAD
//  Platform requirements lowered from macOS 15 / iOS 18 to macOS 14 / iOS 18
//  (audit confirmed no macOS 15-only APIs are used in those targets).
//
//  GraniteSpeech is original work (a native MLX-Swift port of IBM
//  Granite-Speech 4.1-2b), not part of the vendored subset.
//
//  See NOTICE.md for upstream attribution and modification log.
//

import PackageDescription

let package = Package(
    name: "Qwen3Speech",
    platforms: [
        .macOS("14.0"),
        .iOS("18.0")
    ],
    products: [
        .library(
            name: "Qwen3ASR",
            targets: ["Qwen3ASR"]
        ),
        .library(
            name: "AudioCommon",
            targets: ["AudioCommon"]
        ),
        .library(
            name: "MLXCommon",
            targets: ["MLXCommon"]
        ),
        .library(
            name: "SpeechVAD",
            targets: ["SpeechVAD"]
        ),
        .library(
            name: "GraniteSpeech",
            targets: ["GraniteSpeech"]
        )
    ],
    dependencies: [
        // Pinned to the 0.30.x minor so a standalone build resolves against a
        // matching Metal kernel library — a 0.31.x Cmlx expects fused kernels
        // (e.g. layer_norm) absent from the 0.30.3 default.metallib used by the
        // downstream consumer's runtime.
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.30.3")),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6")
    ],
    targets: [
        .target(
            name: "AudioCommon",
            dependencies: [
                .product(name: "Hub", package: "swift-transformers")
            ]
        ),
        .target(
            name: "MLXCommon",
            dependencies: [
                "AudioCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift")
            ]
        ),
        .target(
            name: "SpeechVAD",
            dependencies: [
                "AudioCommon",
                "MLXCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift")
            ]
        ),
        .target(
            name: "Qwen3ASR",
            dependencies: [
                "AudioCommon",
                "MLXCommon",
                "SpeechVAD",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift")
            ]
        ),
        .target(
            name: "GraniteSpeech",
            dependencies: [
                "AudioCommon",
                "MLXCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers")
            ]
        )
    ]
)
