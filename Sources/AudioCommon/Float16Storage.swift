//
//  Float16Storage.swift
//  AudioCommon
//
//  Created by Kévin Naudin on 2026-08-14.
//

#if canImport(CoreML)
import Accelerate
import CoreML
import Foundation

/// Half-precision `MLMultiArray` access that never names Swift's `Float16`.
///
/// `Float16` does not exist on x86_64 macOS, so every `assumingMemoryBound(to:
/// Float16.self)` in this package blocks a universal build. Reading and writing
/// the storage as raw `UInt16` keeps the layout byte-identical (same size,
/// stride and alignment) and lets Accelerate do the conversion, which is also
/// faster than the per-element loops it replaces.
///
/// The bulk paths require a C-contiguous array. Arrays we allocate ourselves
/// always are; model *outputs* may carry ANE padding, so `isContiguous` is
/// checked and the per-element subscript (which honours `strides`) is used as
/// the fallback rather than silently reading padding slots.
public enum Float16Storage {

    /// Byte width of one half element. Stands in for `MemoryLayout<Float16>.size`.
    public static let elementBytes = MemoryLayout<UInt16>.size

    /// Whether `array`'s strides are exactly C-order for its shape, i.e. the
    /// linear subscript index and the raw storage offset coincide.
    ///
    /// Conservative by construction: shape-1 axes may carry an arbitrary stride
    /// and are reported non-contiguous, which costs speed on the subscript path
    /// but can never read the wrong offset.
    public static func isContiguous(_ array: MLMultiArray) -> Bool {
        // Hoisted: `shape` / `strides` bridge NSArray to [NSNumber] on every
        // access, which dominated this check when read inside the loop.
        let shape = array.shape
        let strides = array.strides
        var expected = 1
        for axis in (0..<shape.count).reversed() {
            if strides[axis].intValue != expected { return false }
            expected *= shape[axis].intValue
        }
        return true
    }

    /// Read `count` half elements from `array` into `dst` as `Float`.
    public static func read(
        _ array: MLMultiArray,
        count: Int,
        into dst: UnsafeMutablePointer<Float>
    ) {
        guard isContiguous(array) else {
            for i in 0..<count { dst[i] = array[i].floatValue }
            return
        }
        var src = vImage_Buffer(
            data: array.dataPointer,
            height: 1,
            width: vImagePixelCount(count),
            rowBytes: count * elementBytes
        )
        var out = vImage_Buffer(
            data: UnsafeMutableRawPointer(dst),
            height: 1,
            width: vImagePixelCount(count),
            rowBytes: count * MemoryLayout<Float>.size
        )
        vImageConvert_Planar16FtoPlanarF(&src, &out, 0)
    }

    /// Read `count` half elements from `array` as a fresh `[Float]`.
    public static func read(_ array: MLMultiArray, count: Int) -> [Float] {
        var out = [Float](repeating: 0, count: count)
        out.withUnsafeMutableBufferPointer { read(array, count: count, into: $0.baseAddress!) }
        return out
    }

    /// Write `count` `Float` values into `array`'s half storage, then zero the
    /// remaining `array.count - count` elements.
    public static func write(
        _ src: UnsafePointer<Float>,
        count: Int,
        intoPadded array: MLMultiArray
    ) {
        // The loops this replaced were bounds-checked by Swift; vImage is not,
        // so an over-long `count` would silently write past the buffer.
        precondition(
            count <= array.count,
            "Float16Storage.write: \(count) elements into a \(array.count)-element array"
        )
        guard isContiguous(array) else {
            for i in 0..<count { array[i] = NSNumber(value: src[i]) }
            for i in count..<array.count { array[i] = 0 }
            return
        }
        var input = vImage_Buffer(
            data: UnsafeMutableRawPointer(mutating: src),
            height: 1,
            width: vImagePixelCount(count),
            rowBytes: count * MemoryLayout<Float>.size
        )
        var output = vImage_Buffer(
            data: array.dataPointer,
            height: 1,
            width: vImagePixelCount(count),
            rowBytes: count * elementBytes
        )
        vImageConvert_PlanarFtoPlanar16F(&input, &output, 0)
        if count < array.count {
            // Half zero is all-zero bits, so the tail is a plain memset.
            memset(array.dataPointer + count * elementBytes, 0, (array.count - count) * elementBytes)
        }
    }

    /// Zero every element. Half zero is all-zero bits.
    public static func zeroFill(_ array: MLMultiArray) {
        guard isContiguous(array) else {
            for i in 0..<array.count { array[i] = 0 }
            return
        }
        memset(array.dataPointer, 0, array.count * elementBytes)
    }
}
#endif
