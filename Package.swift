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
        // Pinned to the same minor the Onit app resolves (0.30.x) so standalone
        // `swift test` runs against a matching Metal kernel library — a 0.31.x
        // Cmlx expects fused kernels (e.g. layer_norm) absent from the app's
        // 0.30.3 default.metallib used by the CLI test runner.
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
        ),
        .testTarget(
            name: "GraniteSpeechTests",
            dependencies: [
                "GraniteSpeech",
                .product(name: "MLX", package: "mlx-swift")
            ]
        ),
        .testTarget(
            name: "Qwen3ASRTests",
            dependencies: [
                "Qwen3ASR"
            ]
        )
    ]
)
