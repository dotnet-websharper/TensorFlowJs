namespace WebSharper.TensorFlowJs.Types

open WebSharper.JavaScript
open WebSharper.InterfaceGenerator

[<AutoOpen>]
module Types = 
    let FloatOrFloatArray = T<float> + T<float[]>
    let StringOrInt = T<string> + T<int>
    let NumberTypedArrayOrObjArray = T<obj[]> + T<Uint32Array> + T<Int32Array> + T<Float32Array>
    let IteratorResult = T<obj>
    let Iterator = T<obj>
    let BackendValues = T<Uint8Array> + T<Uint8Array[]> + T<Int32Array> + T<Float32Array>
    let DataValues = T<Float32Array> + T<Int32Array> + T<Uint8Array> + T<string[]>
    let FlagValue = T<float> + T<bool> + T<string> 
    let Flags = T<obj[]>
    let DataId = T<obj>
    let Kwargs = T<obj[]>
    let DataType = T<string>
    let NamedTensorMap = T<obj[]>
    let Shape = T<int[]>
    let ShapeOrArray = Shape + !|Shape
    let StringOrArray = T<string> + T<string[]>
    let IntOrIntArray = T<int> + T<int[]>
    let StringOrIntOrIntArray = T<string> + IntOrIntArray
    let TypedArray = T<Uint8Array> + T<Uint8ClampedArray> + T<Int32Array> + T<Float32Array>
    let FlattenType = T<int> + T<bool> + T<string> + T<Promise<int>> + TypedArray
    let TensorLike = TypedArray + T<int> + T<bool> + T<string> + T<int[]> + T<float[]>
    let ScalarLike = T<int> + T<bool> + T<string> + T<Uint8Array>
    let TensorLike1D = TypedArray + T<int[]> + T<bool[]> + T<string[]> + T<Uint8Array[]>
    let TensorLike2D = TypedArray + T<int[]> + T<int[][]> + T<bool[]> + T<bool[][]> + T<string[]> + T<string[][]> + T<Uint8Array[]> + T<Uint8Array[][]>
    let TensorLike3D = TypedArray + T<int[]> + T<int[][][]> + T<bool[]> + T<bool[][][]> + T<string[]>  + T<string[][][]> + T<Uint8Array[]> + T<Uint8Array[][][]>
    let TensorLike4D = TypedArray + T<int[]> + T<int[][][][]> + T<bool[]> + T<bool[][][][]> + T<string[]> + T<string[][][][]> + T<Uint8Array[]> + T<Uint8Array[][][][]>
    let TensorLike5D = TypedArray + T<int[]> + T<int[][][][][]> + T<bool[]> + T<bool[][][][][]> + T<string[]> + T<string[][][][][]> + T<Uint8Array[]> + T<Uint8Array[][][][][]>
    let TensorLike6D = TypedArray + T<int[]> + T<int[][][][][][]> + T<bool[]> + T<bool[][][][][][]> + T<string[]> + T<string[][][][][][]> + T<Uint8Array[]> + T<Uint8Array[][][][][][]>
    
    let InitializerIdentifier = T<string>
    let ConstraintIdentifier = T<string>
    let RegularizerIdentifier = T<string>

