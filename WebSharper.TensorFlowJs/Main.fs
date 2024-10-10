namespace WebSharper.TensorFlowJs

open WebSharper
open WebSharper.JavaScript
open WebSharper.InterfaceGenerator

module Definition =
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

    [<AutoOpen>]
    module Enumuration = 
        let Reduction = 
            Pattern.EnumStrings "Reduction" [
                "NONE"
                "MEAN"
                "SUM"
                "SUM_BY_NONZERO_WEIGHTS"
            ]

        let ActivationIdentifier =
            Pattern.EnumStrings "ActivationIdentifier" [
                "elu"
                "hardSigmoid"
                "linear"
                "relu"
                "relu6"
                "selu"
                "sigmoid"
                "softmax"
                "softplus"
                "softsign"
                "tanh"
                "swish"
                "mish"
                "gelu"
                "gelu_new"
            ]

        let WebGLChannels = Pattern.EnumStrings "WebGLChannels" [
            "A"; "B"; "G"; "R"; "AB"; "AG"; "AR"; "BA"; "BG"; "BR"; "GA"; "GB"; "GR"; 
            "RA"; "RB"; "RG"; "ABG"; "ABR"; "AGB"; "AGR"; "ARB"; "ARG"; "BAG"; "BAR"; 
            "BGA"; "BGR"; "BRA"; "BRG"; "GAB"; "GAR"; "GBA"; "GBR"; "GRA"; "GRB"; 
            "RAB"; "RAG"; "RBA"; "RBG"; "RGA"; "RGB"; "ABGR"; "ABRG"; "AGBR"; "AGRB"; 
            "ARBG"; "ARGB"; "BAGR"; "BARG"; "BGAR"; "BGRA"; "BRAG"; "BRGA"; "GABR"; 
            "GARB"; "GBAR"; "GBRA"; "GRAB"; "GRBA"; "RABG"; "RAGB"; "RBAG"; "RBGA"; 
            "RGAB"; "RGBA"
        ]

        let Rank =
            Pattern.EnumStrings "Rank" [
                "R0"; "R1"; "R2"; "R3"; "R4"; "R5"; "R6"
            ]

        let WeightGroup = 
            Pattern.EnumStrings "WeightGroup" [
                "model"; "optimizer"
            ] 

        let Category = 
            Pattern.EnumStrings "Category" [
                "arithmetic"
                "basic_math"
                "control"
                "convolution"
                "creation"
                "custom"
                "dynamic"
                "evaluation"
                "graph"
                "hash_table"
                "image"
                "logical"
                "matrices"
                "normalization"
                "ragged"
                "reduction"
                "slice_join"
                "sparse"
                "spectral"
                "string"
                "transformation"
            ]

        let ParamType = 
            Pattern.EnumStrings "ParamType" [
                "number"
                "string"
                "string[]"
                "number[]"
                "bool"
                "bool[]"
                "shape"
                "shape[]"
                "tensor"
                "tensors"
                "dtype"
                "dtype[]"
                "func"
            ]  

    [<AutoOpen>]
    module Interface = 
        let MemoryInfo = 
            Pattern.Config "MemoryInfo" {
                Required = [
                    "numTensors", T<int>
                    "numDataBuffers", T<int>
                    "numBytes", T<int>
                    "reasons", T<string[]>
                ]
                Optional = [
                    "unreliable", T<bool>
                ]
            }

        let BackendTimingInfo = 
            Pattern.Config "BackendTimingInfo" {
                Required = [
                    "kernelMs", T<int> + T<obj>
                ]
                Optional = [
                    "getExtraProfileInfo", T<string>
                ]
            }

        let KernelInfo = 
            Pattern.Config "KernelInfo" {
                Required = [
                    "name", T<string>
                    "bytesAdded", T<int>
                    "totalBytesSnapshot", T<int>
                    "tensorsAdded", T<int>
                    "totalTensorsSnapshot", T<int>
                    "inputShapes", T<int[][]>
                    "outputShapes", T<int[][]>
                    "kernelTimeMs", T<int> + T<obj> + T<Promise<_>>[T<int> + T<obj>]
                    "extraInfo", T<string> + T<Promise<string>>
                ]
                Optional = []
            }

        let TimingInfo = 
            Pattern.Config "TimingInfo" {
                Required = [
                    "wallMs", T<int>
                ]
                Optional = []
            }
            |=> Inherits BackendTimingInfo

        let Memory = 
            Pattern.Config "Memory" {
                Required = [
                    "unreliable", T<bool>
                ]
                Optional = [
                    "reasons", T<string []>
                ]
            }

        let SingleValueMap =
            Pattern.Config "SingleValueMap" {
                Required = []
                Optional = [
                    "bool", T<bool>
                    "int32", T<int>
                    "float32", T<float>
                    "complex64", T<float>
                    "string", T<string>
                ]
            }

        let TensorInfo =
            Pattern.Config "TensorInfo" {
                Required = [
                    "dataId", DataId
                    "shape", Shape
                    "dtype", T<string>
                ]
                Optional = []
            }

        let DataTypeMap =
            Pattern.Config "DataTypeMap" {
                Required = [
                    "float32", T<Float32Array>
                    "int32", T<Int32Array>
                    "bool", T<Uint8Array>
                    "complex64", T<Float32Array>
                    "string", !| T<string>
                ]
                Optional = []
            }

        let DataToGPUWebGLOption =
            Pattern.Config "DataToGPUWebGLOption" {
                Required = []
                Optional = [
                    "customTexShape", !|T<int>
                ]
            }

        let WebGLData = Pattern.Config "WebGLData" {
            Required = [
                "texture", T<WebGL.Texture>
                "height", T<int>
                "width", T<int>
                "channels", WebGLChannels.Type
            ]
            Optional = []
        }

        let WebGPUData = Pattern.Config "WebGPUData" {
            Required = [
                "buffer", T<WebGPU.GPUBuffer>
            ]
            Optional = [
                "zeroCopy", T<bool>
            ]
        }   

        let DataToGPUOptions = DataToGPUWebGLOption

        let TensorValuesType = TensorLike + WebGLData + WebGPUData
        let LogitsType = TensorLike1D + TensorLike2D + TypedArray + T<float[]>

        let ModelTensorInfo = 
            Pattern.Config "ModelTensorInfo" {
                Required = [
                    "name", T<string>
                    "dtype", T<string>
                ]
                Optional = [
                    "shape", !| T<int>
                    "tfDtype", T<string>
                ]
            }

        let WeightData = T<ArrayBuffer> + !| T<ArrayBuffer>

        let TrainingConfig = 
            Pattern.Config "TrainingConfig" {
                Required = [
                    "optimizer_config", T<obj> 
                    "loss", T<string> + !| T<string> + T<obj>
                ]
                Optional = [
                    "metrics", !| T<string> + T<obj>
                    "weighted_metrics", !| T<string>
                    "sample_weight_mode", T<string>
                    "loss_weights", !| T<float> + T<obj>
                ]
            }

        let Quantization = 
            Pattern.Config "Quantization" {
                Required = [
                    "dtype", T<string> 
                ]
                Optional = [
                    "scale", T<float>
                    "min", T<float>
                ]
            }

        let WeightsManifestEntry = 
            Pattern.Config "WeightsManifestEntry" {
                Required = [
                    "name", T<string>
                    "shape", !| T<int>
                    "dtype", T<string> 
                ]
                Optional = [
                    "group", T<string> 
                    "quantization", Quantization.Type
                ]
            }

        let GetWeightStream = T<unit> ^-> T<ReadableStream>

        let SaveConfig = 
            Pattern.Config "SaveConfig" {
                Required = []
                Optional = [
                    "trainableOnly", T<bool>
                    "includeOptimizer", T<bool>
                ]
            }

        let ModelArtifactsInfo = 
            Pattern.Config "ModelArtifactsInfo" {
                Required = [
                    "dateSaved", T<Date>
                    "modelTopologyType", T<string> 
                ]
                Optional = [
                    "modelTopologyBytes", T<int>
                    "weightSpecsBytes", T<int>
                    "weightDataBytes", T<int>
                ]
            }

        let ModelArtifacts = 
            Pattern.Config "ModelArtifacts" {
                Required = []
                Optional = [
                    "modelTopology", T<obj> + T<ArrayBuffer>
                    "trainingConfig", TrainingConfig.Type
                    "weightSpecs", !| WeightsManifestEntry.Type
                    "weightData", WeightData
                    "getWeightStream", T<unit> ^-> T<ReadableStream>
                    "format", T<string>
                    "generatedBy", T<string>
                    "convertedBy", T<string> + T<unit>
                    "signature", T<obj>
                    "userDefinedMetadata", T<obj>
                    "modelInitializer", T<obj>
                    "initializerSignature", T<obj>
                ]
        }

        let SaveResult = 
            Pattern.Config "SaveResult" {
                Required = [
                    "modelArtifactsInfo", ModelArtifactsInfo.Type
                ]
                Optional = [
                    "responses", !| T<Response>
                    "errors", !| (T<string> + T<obj>)
                ]
            }

        let ModelPredictConfig = 
            Pattern.Config "ModelPredictConfig" {
                Required = []
                Optional = [
                    "batchSize", T<int>
                    "verbose", T<bool>
                ]
            }

        let SaveHandler = ModelArtifacts?modelArtifact ^-> T<Promise<_>>[SaveResult]
        let LoadHandler = T<unit> ^-> T<Promise<_>>[ModelArtifacts]

        let IOHandler = 
            Pattern.Config "IOHandler" {
                Required = []
                Optional = [
                    "save", SaveHandler
                    "load", LoadHandler
                ]
            }

        let SaveHandlerSync = ModelArtifacts?modelArtifact ^-> SaveResult
        let LoadHandlerSync = T<unit> ^-> ModelArtifacts

        let IOHandlerSync = 
            Pattern.Config "IOHandlerSync" {
                Required = []
                Optional = [
                    "save", SaveHandlerSync
                    "load", LoadHandlerSync
                ]
            }

        let HandleOrString = IOHandler + T<string>

        let ModelEvaluateDatasetArgs = 
            Pattern.Config "ModelEvaluateDatasetArgs" {
                Required = []
                Optional = [
                    "batchSize", T<int>
                    "verbose", T<int>
                ]
            }

        let ModelPredictArgs = 
            Pattern.Config "ModelPredictArgs" {
                Required = []
                Optional = [
                    "batchSize", T<int>
                    "verbose", T<bool>
                ]
            }

        let CustomCallbackArgs = 
            Pattern.Config "CustomCallbackArgs" {
                Required = []
                Optional = [
                    "onTrainBegin", (T<obj>?log ^-> (T<unit> + T<Promise<unit>>)) 
                    "onTrainEnd", (T<obj>?log ^-> (T<unit> + T<Promise<unit>>))
                    "onEpochBegin", (T<int>?epoch * !?T<obj>?log) ^-> (T<unit> + T<Promise<unit>>)
                    "onEpochEnd", (T<int>?epoch * !?T<obj>?log) ^-> (T<unit> + T<Promise<unit>>)
                    "onBatchBegin", (T<int>?batch * !?T<obj>?log) ^-> (T<unit> + T<Promise<unit>>)
                    "onBatchEnd", (T<int>?batch * !?T<obj>?log) ^-> (T<unit> + T<Promise<unit>>)
                    "onYield", (T<int>?epoch * T<int>?batch * T<obj>?log) ^-> (T<unit> + T<Promise<unit>>)
                ]
            }

        let PrintFnType = 
            Pattern.Config "PrintFnType" {
                Required = [
                    "message", T<obj>
                ]
                Optional = [
                    "optionalParams", T<obj[]>
                ]
            }

        let RequestDetails = 
            Pattern.Config "RequestDetails" {
                Required = []
                Optional = [
                    "isBinary", T<bool>
                ]
            }

        let fetch = T<string>?path * !? T<Request>?init * !? RequestDetails?options ^-> T<Promise<Response>>
        let OnProgressCallback = T<float>?fraction ^-> T<unit>

        let LoadOptions = 
            Pattern.Config "LoadOptions" {
                Required = []
                Optional = [
                    "requestInit", T<Request>
                    "onProgress", OnProgressCallback
                    "fetchFunc", fetch
                    "strict", T<bool>
                    "weightPathPrefix", T<string>
                    "fromTFHub", T<bool>
                    "weightUrlConverter", T<string>?weightFileName ^-> T<Promise<string>>
                    "streamWeights", T<bool>
                ]
            }

        let InputConfig =
            Pattern.Config "InputConfig" {
                Required = []
                Optional = [
                    "shape", T<int[]>
                    "batchShape", Shape
                    "name", T<string>
                    "dtype", T<string>
                    "sparse", T<bool>
                ]
            }

        let copyModel = T<string>?sourceURL * T<string>?destURL ^-> T<Promise<_>>[ModelArtifactsInfo]
        let listModels = T<unit> ^-> T<Promise<obj>>
        let moveModel = T<string>?sourceURL * T<string>?destURL ^-> T<Promise<_>>[ModelArtifactsInfo]
        let removeModel = T<string>?url ^-> T<Promise<_>>[ModelArtifactsInfo]

        let browserFiles = T<File[]>?files ^-> IOHandler
        let browserHTTPRequest = T<string>?path * !?LoadOptions?loadOptions ^-> IOHandler

        let ArrayBufferOrArray = T<ArrayBuffer> + T<ArrayBuffer[]>

        let CompositeArrayBuffer = 
            Class "CompositeArrayBuffer" 
            |+> Static [
                Constructor !?(ArrayBufferOrArray + TypedArray + !| TypedArray)?buffers
                "join" => !?ArrayBufferOrArray?buffers ^-> T<unit>
            ]
            |+> Instance [
                "slice" => T<unit> ^-> T<ArrayBuffer>
            ] 

        let EncodeWeightsResult = 
            Pattern.Config "EncodeWeightsResult" {
                Required = [
                    "data", T<ArrayBuffer>
                    "specs", !|WeightsManifestEntry
                ]
                Optional = []
            }

        let WeightsManifestGroupConfig =
            Pattern.Config "WeightsManifestGroupConfig" {
                Required = [
                    "paths", !| T<string> 
                    "weights", !| WeightsManifestEntry.Type 
                ]
                Optional = []
            }

        let WeightsManifestConfig = !|WeightsManifestGroupConfig

        let ModelJSON =
            Pattern.Config "ModelJSON" {
                Required = [
                    "modelTopology", T<obj> 
                    "weightsManifest", WeightsManifestConfig
                ]
                Optional = [
                    "trainingConfig", TrainingConfig.Type
                    "format", T<string>
                    "generatedBy", T<string>
                    "convertedBy", (T<string> + T<unit>)
                    "signature", T<obj>
                    "userDefinedMetadata", !| T<obj>
                    "modelInitializer", T<obj>
                    "initializerSignature", T<obj>
                ]
            }

        let ModelStoreManager = 
            Pattern.Config "ModelStoreManager" {
                Required = []
                Optional = [
                    "listModels", T<unit> ^-> T<Promise<obj>>
                    "removeModel", T<string>?path ^-> T<Promise<_>>[ModelArtifactsInfo]
                ]
            }

        let IORouter = StringOrArray?url * !?LoadOptions?loadOptions ^-> IOHandler

        let IORouterRegistry  =     
            Class "IORouterRegistry"
            |+> Static [
                "registerSaveRouter" => IORouter ^-> T<unit>
                "registerLoadRouter" => IORouter ^-> T<unit>
                "getSaveHandlers" => StringOrArray?url ^-> !|IOHandler
                "getLoadHandlers" => StringOrArray?url * !?LoadOptions?loadOptions ^-> !|IOHandler
            ]

        let LoadWeightAsyncResult = !| (!| WeightsManifestEntry + WeightData)

        let LoadWeightAsync = WeightsManifestConfig?weightsManifest ^-> T<Promise<_>>[LoadWeightAsyncResult]

        let concatenateArrayBuffers = ArrayBufferOrArray?buffers ^-> T<ArrayBuffer>
        let decodeWeights = WeightData?weightData * (!|WeightsManifestEntry)?specs ^-> NamedTensorMap
        let decodeWeightsStream = T<ReadableStream>?weightStream * (!|WeightsManifestEntry)?specs ^-> T<Promise<_>>[NamedTensorMap]

        let fromMemory = (T<obj> + ModelArtifacts)?modelArtifacts * !? (!|WeightsManifestEntry)?weightSpecs * !?WeightData?weightData * !?TrainingConfig?trainingConfig ^-> IOHandler
        let fromMemorySync = (T<obj> + ModelArtifacts)?modelArtifacts * (!? (!|WeightsManifestEntry))?weightSpecs * !?WeightData?weightData * !?TrainingConfig?trainingConfig ^-> IOHandlerSync

        let getLoadHandlers = StringOrArray?url * !?LoadOptions?loadOptions ^-> IORouterRegistry
        let getModelArtifactsForJSON = ModelJSON?modelJSON * LoadWeightAsync?loadWeights ^-> T<Promise<_>>[ModelArtifacts]
        let getModelArtifactsForJSONSync = ModelJSON?modelJSON * !? (!|WeightsManifestEntry)?weightSpecs * !?WeightData?weightData ^-> ModelArtifacts
        let getModelArtifactsInfoForJSON = ModelArtifacts?modelArtifacts ^-> ModelArtifactsInfo
        let getSaveHandlers = StringOrArray?url ^-> IORouterRegistry
        let getWeightSpecs = WeightsManifestConfig?modelArtifacts ^-> !| WeightsManifestEntry

        let http = T<string>?path * !?LoadOptions?loadOptions ^-> IOHandler 
        let isHTTPScheme = T<string>?url^-> T<bool>

        let loadWeights = WeightsManifestConfig?manifest * !?T<string>?filePathPrefix * !?(!|T<string>)?weightNames * !?T<Request>?requestInit ^-> T<Promise<_>>[NamedTensorMap] 

        let registerLoadRouter = IORouter?loudROuter ^-> IORouterRegistry
        let registerSaveRouter  = IORouter?loudROuter ^-> IORouterRegistry

        let FetchWeightsFunction = T<string[]>?fetchUrls ^-> T<Promise<ArrayBuffer[]>>
        let weightsLoaderFactory = FetchWeightsFunction?fetchWeightsFunction ^-> T<Promise<_>>[NamedTensorMap]
        let SaveHandlerFunction = ModelArtifacts?artifacts ^-> T<Promise<_>>[SaveResult]
        let SaveHandlerFunctionSync = ModelArtifacts?artifacts ^-> SaveResult
        let withSaveHandler = SaveHandlerFunction?saveHandler ^-> IOHandler
        let withSaveHandlerSync = SaveHandlerFunctionSync?saveHandler ^-> IOHandlerSync

        let ConfigDict = T<obj[]>     
        
        let SerializableConstructor =
            Generic - fun t ->
            Class "SerializableConstructor"
            |+> Static [            
                Constructor (T<obj[]>?args ^-> t)
            ]

        let Serializable = 
            Class "Serializable" 
            |+> Instance [
                "getClassName" => T<unit> ^-> T<string>
                "getConfig" => T<unit> ^-> ConfigDict
            ]    

        Serializable 
        |+> Static [
            Generic - fun t ->
                "fromConfig" => SerializableConstructor[t]?cls * ConfigDict?config ^-> t
        ] |> ignore

        let InputSpecArgs =
            Pattern.Config "InputSpecArgs" {
                Required = []
                Optional = [
                    "dtype", !?T<string>
                    "shape", !?Shape
                    "ndim", !?T<int>
                    "maxNDim", !?T<int>
                    "minNDim", !?T<int>
                    "axes", !?T<obj[]>
                ]
            }

        let DisposeResult =
            Pattern.Config "DisposeResult" {
                Required = [
                    "refCountAfterDispose", T<int>
                    "numDisposedVariables", T<int>
                ]
                Optional = []
            }

        let MaxNormArgs = 
            Pattern.Config "MaxNormArgs" {
                Required = []
                Optional = [
                    "maxValue", T<int>
                    "axis", T<int>
                ]
            }

        let MinMaxNormArgs = 
            Pattern.Config "MinMaxNormArgs" {
                Required = []
                Optional = [
                    "minValue", T<int>
                    "maxValue", T<int>
                    "axis", T<int>
                ]
            }

        let UnitNormArgs = 
            Pattern.Config "UnitNormArgs" {
                Required = []
                Optional = [
                    "axis", T<int>
                ]
            }

        let ConstantArgs = 
            Pattern.Config "ConstantArgs" {
                Required = []
                Optional = [
                    "value", T<int>
                ]
            }

        let GlorotNormalArgs = 
            Pattern.Config "GlorotNormalArgs" {
                Required = []
                Optional = [
                    "seed", T<int>
                ]
            }

    [<AutoOpen>]
    module TensorFlow = 
        let Tensor =
            Class "Tf.Tensor"
            |=> Inherits TensorInfo
            |+> Static [
                Constructor (Shape?shape * DataType?dtype * DataId?dataId * T<int>?id)
            ]
            |+> Instance [
                "id" =? T<int> 
                "dataId" =? DataId 
                "shape" =? Shape
                "size" =? T<int> 
                "dtype" =? DataType
                "rankType" =? Rank 
                "kept" =@ T<bool>
                "scopeId" =@ T<int> 
                "kerasMask" =@ !? TSelf 
                "strides" =? T<int[]> 
            ] 

        let LossOrMetricFn = Tensor?yTrue * Tensor?yPred ^-> Tensor

        let GPUData =
            Pattern.Config "GPUData" {
                Required = [
                    "tensorRef", Tensor.Type
                ]
                Optional = [
                    "texture", T<WebGL.Texture>
                    "buffer", T<WebGPU.GPUBuffer>
                    "texShape", !|T<int>
                ]
            }

        let TensorBuffer = 
            Class "Tf.TensorBuffer" 
            |+> Static [
                Constructor (Shape?shape * DataType?dtype * !?DataTypeMap?values)
            ]
            |+> Instance [       
                "size" =@ T<int>
                "shape" =@ Shape
                "strides" =@ T<int[]>
                "values" =@ DataTypeMap

                "set" => SingleValueMap?value * T<float[]>?locs ^-> T<unit>
                "get" => T<float[]>?locs ^-> SingleValueMap
                "locToIndex" => T<float[]>?locs ^-> T<float>
                "indexToLoc" => T<float>?index ^-> T<float[]>
                "toTensor" => T<unit> ^-> Tensor
            ]

        let Variable = 
            Class "Tf.Variable"
            |=> Inherits Tensor
            |+> Static [
                Constructor (Tensor?initialValue * T<bool>?trainable * T<string>?name * T<int>?tensorId)
            ]
            |+> Instance [
                "name" =@ T<string>

                "assign" => Tensor?newValue ^-> T<unit>
            ]

        Tensor 
        |+> Instance [
            "buffer" => T<unit> ^-> T<Promise<_>>[TensorBuffer]
            "bufferSync" => T<unit> ^-> TensorBuffer
            "array" => T<unit> ^-> T<Promise<_>>[!|T<float>]
            "arraySync" => T<unit> ^-> !|T<float>
            "data" => T<unit> ^-> T<Promise<_>>[DataTypeMap]
            "dataToGPU" => !?DataToGPUOptions?options ^-> GPUData
            "dataSync" => T<unit> ^-> DataTypeMap
            "bytes" => T<unit> ^-> T<Promise<_>>[T<Uint8Array[]> + T<Uint8Array>]
            "dispose" => T<unit> ^-> T<unit>
            "throwIfDisposed" => T<unit> ^-> T<unit>
            "print" => !?T<bool>?verbose ^-> T<unit>
            "clone" => TSelf?this ^-> TSelf
            "toString" => !?T<bool>?verbose ^-> T<string>
            "cast" => DataType?dtype ^-> TSelf
            "variable" => T<bool>?trainable * !?T<string>?name * !?DataType?dtype ^-> Variable
        ] |> ignore

        let TensorOrTensorArray = Tensor + !| Tensor
        let TensorOrArrayOrMap = Tensor + !| Tensor + T<obj>
        let UnitToTensorFunc = T<unit> ^-> Tensor

        let NamedTensor = 
            Pattern.Config "NamedTensor" {
                Required = [
                    "name", T<string>
                    "tensor", Tensor.Type
                ]
                Optional = []
            } 

        let ComputeGradientsResult = 
            Pattern.Config "ComputeGradientsResult" {
                Required = [
                    "value", Tensor.Type
                    "grads", NamedTensorMap
                ]
                Optional = []
            }

        let Optimizer =
            Class "tf.Train.Optimizer"
            |=> Inherits Serializable
            |+> Instance [
                "minimize" => UnitToTensorFunc?f * !?T<bool>?returnCost * !?(!|Variable)?varList ^-> Tensor + T<unit>
                "computeGradients" => UnitToTensorFunc?f * !?(!|Variable)?varList ^-> ComputeGradientsResult
                "applyGradients" => (NamedTensorMap + !|NamedTensor)?variableGradients ^-> T<unit>
                "dispose" => T<unit> ^-> T<unit>
                "saveIterations" => T<unit> ^-> T<Promise<_>>[NamedTensor]
                "getWeights" => T<unit> ^-> T<Promise<_>>[!|NamedTensor]
                "setWeights" => (!|NamedTensor)?weightValues ^-> T<Promise<unit>>
                "extractIterations" => (!|NamedTensor)?weightValues ^-> T<Promise<_>>[NamedTensor]
            ]

        let TensorContainer = T<unit> + Tensor + T<string> + T<float> + T<bool> + T<obj[]> + !|TSelf + T<Float32Array> + T<Int32Array> + T<Uint8Array>

        let Dataset =
            Generic - fun t ->
                let tToBoolFunc = t?value ^-> T<bool>
                let tToUnitFunc = t?input ^-> T<unit>

                Class "tf.Data.Dataset"
                |+> Instance [
                    "batch" => T<int>?batchSize * !?T<bool>?smallLastBatch ^-> TSelf[t]
                    "concatenate" => TSelf[t]?dataset ^-> TSelf[t]
                    "filter" => tToBoolFunc?predicate ^-> TSelf[t]
                    "forEachAsync" => tToUnitFunc?f ^-> T<Promise<unit>>
                    Generic - fun o -> "map" => (t?value ^-> o)?transform ^-> TSelf[o]
                    Generic - fun o -> "mapAsync" => (t?value ^-> T<Promise<_>>[o])?transform ^-> TSelf[o]
                    "prefetch" => T<int>?bufferSize ^-> TSelf[t]
                    "repeat" => !?T<int>?count ^-> TSelf[t]
                    "skip" => T<int>?count ^-> TSelf[t]
                    "shuffle" => T<int>?bufferSize * !?T<string>?seed * !? T<bool>?reshuffleEachIteration ^-> TSelf[t]
                    "take" => T<int>?count ^-> TSelf[t]
                    "toArray" => T<unit> ^-> T<Promise<_>>[!|t]
                ]

        let DeepMapResult = 
            Pattern.Config "DeepMapResult" {
                Required = [
                    "value", T<obj>
                    "recurse", T<bool>
                ]
                Optional = []
            }

        let ZipFn = T<obj[]>?xs ^-> DeepMapResult

        let LazyIterator = 
            Generic - fun t ->
                Class "LazyIterator"
                |+> Instance [
                    "summary" => T<unit> ^-> T<string>
                    "next" => T<unit> ^-> T<Promise<_>>[IteratorResult]
                    "toArray" => T<unit> ^-> T<Promise<_>>[!|t]
                    "toArrayForTest" => T<unit> ^-> T<Promise<_>>[!|t]
                    "resolveFully" => T<unit> ^-> T<Promise<unit>>
                    "resolveWhile" => (t?r ^-> T<bool>)?predicate ^-> T<Promise<unit>>
                    "handleErrors" => (T<Error>?error ^-> T<bool>)?handler ^-> TSelf[t]
                    "filter" => (t?value ^-> T<bool>)?predicate ^-> TSelf[t]

                    Generic - fun o -> "map" => (t?value ^-> o)?transform ^-> TSelf[o]
                    Generic - fun o -> "mapAsync" => (t?value ^-> T<Promise<_>>[o])?transform ^-> TSelf[o]
                    Generic - fun o -> "serialMapAsync" => (t?value ^-> T<Promise<_>>[o])?transform ^-> TSelf[o]
                    Generic - fun o -> "flatmap" => (t?value ^-> !|o)?transform ^-> TSelf[o]

                    "forEachAsync" => (t?value ^-> T<unit>)?f ^-> T<Promise<unit>>
                    "serialForEach" => (t?value ^-> T<Promise<bool>>)?f ^-> T<Promise<unit>>
                    "rowMajorBatch" => T<int>?batchSize * T<bool>?smallLastBatch ^-> TSelf[!|t]
                    "columnMajorBatch" => T<int>?batchSize * T<bool>?smallLastBatch * ZipFn?zipFn ^-> TSelf[TensorContainer]
                    "concatenate" => TSelf[t]?iterator * !?(T<Error>?e ^-> T<bool>)?baseErrorHandler ^-> TSelf[!|t]
                    "take" => T<int>?count ^-> TSelf[t]
                    "skip" => T<int>?count ^-> TSelf[t]
                    "prefetch" => T<int>?bufferSize ^-> TSelf[t]
                    "shuffle" => T<int>?windowSize * !?T<string>?seed ^-> TSelf[t]
                    "serial" => T<unit> ^-> TSelf[t]
            ]

        let StringIterator = 
            Class "StringIterator"
            |=> Inherits LazyIterator[T<string>]
            |+> Instance [
                "split" => T<string>?seperator ^-> TSelf
            ]

        let ByteChunkIterator = 
            Class "ByteChunkIterator"
            |=> Inherits LazyIterator[T<Uint8Array>]
            |+> Instance [
                "decodeUTF8" => T<unit> ^-> StringIterator
            ]

        let DataSource = 
            Class "DataSource"
            |+> Instance [
                "iterator" => T<unit> ^-> T<Promise<_>>[ByteChunkIterator]
            ]

        let CSVConfig = 
            Pattern.Config "CSVConfig" {
                Required = []
                Optional = [
                    "hasHeader", T<bool>
                    "columnNames", !|T<string>
                    "columnConfigs", T<obj[]>
                    "configuredColumnsOnly", T<bool>
                    "delimiter", T<string>
                    "delimWhitespace", T<bool>
                ]
            }

        let MicrophoneConfig = 
            Pattern.Config "MicrophoneConfig" {
                Required = []
                Optional = [
                    "sampleRateHz", T<int>
                    "fftSize", T<int>
                    "columnTruncateLength", T<int>
                    "numFramesPerSpectrogram", T<int>
                    "audioTrackConstraints", T<MediaTrackConstraints>
                    "smoothingTimeConstant", T<float>
                    "includeSpectrogram", T<bool>
                    "includeWaveform", T<bool>
                ]
            }

        let WebcamConfig = 
            Pattern.Config "WebcamConfig" {
                Required = []
                Optional = [
                    "facingMode", T<string>
                    "deviceId", T<string>
                    "resizeWidth", T<int>
                    "resizeHeight", T<int>
                    "centerCrop", T<bool>
                ]
            }

        let CaptureResult =
            Pattern.Config "CaptureResult" {
                Required = []
                Optional = [
                    "spectrogram", Tensor.Type
                    "waveform", Tensor.Type
                ]
            }

        let MicrophoneIterator = 
            Class "MicrophoneIterator"
            |=> Inherits LazyIterator[TensorContainer]
            |+> Static [
                "create" => MicrophoneConfig?microphoneConfig ^-> T<unit>
            ]
            |+> Instance [
                "summary" => T<unit> ^-> T<unit>
                "start" => T<unit> ^-> T<Promise<unit>>
                "next" => T<unit> ^-> T<Promise<_>>[IteratorResult[TensorContainer]]
                "capture" => T<unit> ^-> T<Promise<_>>[CaptureResult]
                "stop" => T<unit> ^-> T<unit>
                "getSampleRate" => T<unit> ^-> T<int>
            ]

        let WebcamIterator = 
            Class "WebcamIterator"
            |=> Inherits LazyIterator[Tensor]
            |+> Static [
                "create" => T<HTMLVideoElement>?webcamVideoElement * WebcamConfig?webcamConfig ^-> T<unit>
            ]
            |+> Instance [
                "summary" => T<unit> ^-> T<unit>
                "start" => T<unit> ^-> T<Promise<unit>>
                "next" => T<unit> ^-> T<Promise<_>>[IteratorResult]
                "cropAndResizeFrame" => Tensor?img ^-> Tensor
                "stop" => T<unit> ^-> T<unit>
            ]

        let CSVDataset = 
            Class "tf.Data.CSVDataset"
            |=> Inherits Dataset[TensorContainer]
            |+> Instance [
                "columnNames" => T<unit> ^-> T<unit>
                "constructor" => DataSource?input * CSVConfig?csvConfig ^-> T<unit>
                "iterator" => T<unit> ^-> T<Promise<_>>[LazyIterator[TensorContainer]]
                "makeDataElement" => T<string>?line ^-> TensorContainer
                "makeDataElement" => T<string>?line ^-> TensorContainer
            ]

        let LayerArgs =
            Pattern.Config "LayerArgs" {
                Required = []
                Optional = [
                    "inputShape", !?Shape
                    "batchInputShape", !?Shape
                    "batchSize", !?T<int>
                    "dtype", !?DataType
                    "name", !?T<string>
                    "trainable", !?T<bool>
                    "weights", !|Tensor
                    "inputDType", !?DataType
                ]
            }

        let InputSpec =
            Class "InputSpec"
            |+> Static [
                Constructor InputSpecArgs?args
            ]
            |+> Instance [
                "dtype" =@ !?T<string>
                "shape" =@ !?Shape
                "ndim" =@ !?T<int>
                "maxNDim" =@ !?T<int>
                "minNDim" =@ !?T<int>
                "axes" =@ !?T<obj[]>
            ]

        let Layer = 
            Class "tf.Layers.Layer"

        let SymbolicTensor = 
            Class "SymbolicTensor"
            |+> Instance [
                "id" =? T<int>
                "name" =? T<string>
                "originalName" =? !?T<string> 
                "rank" =? T<int> 
                "nodeIndex" =@ T<int> 
                "tensorIndex" =@ T<int>
            ]

        let NodeArgs =
            Pattern.Config "NodeArgs" {
                Required = [
                    "outboundLayer", Layer.Type
                    "inboundLayers", !|Layer
                    "nodeIndices", T<int[]>
                    "tensorIndices", T<int[]>
                    "inputTensors", !|SymbolicTensor
                    "outputTensors", !|SymbolicTensor
                    "inputMasks", !|Tensor
                    "outputMasks", !|Tensor
                    "inputShapes", ShapeOrArray
                    "outputShapes", ShapeOrArray
                ]
                Optional = []
            }

        SymbolicTensor
        |+> Static [
                 Constructor (DataType?dtype * Shape?shape * Layer?sourceLayer * 
                 !|SymbolicTensor * Kwargs?callArgs * !?T<string>?name * T<int>?outputTensorIndex)
            ]|> ignore

        let Node = 
            Class "Node"
            |=> Inherits NodeArgs
            |+> Instance [
                "id" =? T<int>

                "getConfig" => T<unit> ^-> ConfigDict
            ]
            |+> Static [
                Constructor (NodeArgs?args * Kwargs?callargs)
            ]

        let Regularizer = 
            Class "Regularizer"
            |=> Inherits Serializable
            |+> Instance [
                "apply" => Tensor?x ^-> Tensor
            ]

        let SymbolicTensorOrArray = SymbolicTensor + !|SymbolicTensor

        let Initializer =
            Class "Initializer"
            |=> Inherits Serializable
            |+> Instance [
                "fromConfigUsesCustomObjects" => T<unit> ^-> T<bool> 
                "apply" => Shape?shape * !?DataType?dtype ^-> Tensor 
                "getConfig" => T<unit> ^-> ConfigDict 
            ]

        let Constraint =
            Class "tf.Constraints.Constraint"
            |=> Inherits Serializable
            |+> Instance [
                "apply" => Tensor?w ^-> Tensor 
                "getConfig" => T<unit> ^-> ConfigDict 
            ]

        let LayerVariable =
            Class "LayerVariable"
            |+> Static [
                Constructor (Tensor?``val`` * !?DataType?dtype * !?T<string>?name * !?T<bool>?trainable * !?Constraint?``constraint``)
            ]
            |+> Instance [
                "read" => T<unit> ^-> Tensor 
                "write" => Tensor?newVal ^-> TSelf 
                "dispose" => T<unit> ^-> T<unit> 
            ]

        let RegularizerFn = T<unit> ^-> Tensor
        let RegularizerFnOrArray = RegularizerFn + !|RegularizerFn

        Layer
        |=> Inherits Serializable
        |+> Static [
            Constructor (LayerArgs?args ^-> T<obj>)
            "nodeKey" => Layer?layer * T<int>?nodeIndex ^-> T<unit>
        ]
        |+> Instance [
            "apply" => (TensorOrTensorArray + SymbolicTensorOrArray)?inputs * !?Kwargs?kwargs ^-> TensorOrTensorArray + SymbolicTensorOrArray
            "countParams" => T<unit> ^-> T<int> 
            "build" => ShapeOrArray?inputShape ^-> T<unit> 
            "getWeights" => T<bool>?trainableOnly ^-> !|Tensor 
            "setWeights" => (!|Tensor)?weights ^-> T<unit> 
            "addWeight" => T<string>?name * Shape?shape * !?DataType?dtype * !?Initializer?initializer * !?Regularizer?regularizer * !?T<bool>?trainable * !?Constraint?``constraint`` * !?T<Function>?getInitializerFunc ^-> LayerVariable 
            "addLoss" => RegularizerFnOrArray?losses ^-> T<unit> 
            "computeOutputShape" => ShapeOrArray?inputShape ^-> ShapeOrArray
            "getConfig" => T<unit> ^-> ConfigDict 
            "dispose" => T<unit> ^-> DisposeResult 
        ] |> ignore

        let GraphNode =
            Pattern.Config "GraphNode" {
                Required = [
                    "inputs", !|Tensor
                    "attrs", T<obj[]>
                ]
                Optional = []
            }


        let OpExecutor = GraphNode?node ^-> TensorOrTensorArray + T<Promise<_>>[TensorOrTensorArray]      
        
        let ValueType = 
            Pattern.Config "ValueType" {
                Required = []
                Optional = [
                    "string", T<string>
                    "string[]", T<string[]>
                    "number", T<float>
                    "number[]", T<float[]>
                    "number[][]", T<float[][]>
                    "boolean", T<bool>
                    "boolean[]", T<bool[]>
                    "tensor", Tensor.Type
                    "tensors", !|Tensor
                ]
            }

        let ParamMapper =
            Pattern.Config "ParamMapper" {
                Required = [
                    "name", T<string>
                    "type", ParamType.Type
                ]
                Optional = [
                    "defaultValue", ValueType.Type
                    "notSupported", T<bool>
                ]
            }

        let InputParamMapper =
            Pattern.Config "InputParamMapper" {
                Required = [
                    "start", T<int>
                ]
                Optional = [
                    "end", T<int>
                ]
            }
            |=> Inherits ParamMapper

        let AttrParamMapper =
            Pattern.Config "AttrParamMapper" {
                Required = []
                Optional = [
                    "tfName", T<string>
                    "tfDeprecatedName", T<string>
                ]
            }
            |=> Inherits ParamMapper

        let OpMapper =
            Pattern.Config "OpMapper" {
                Required = [
                    "tfOpName", T<string>
                ]
                Optional = [
                    "category", Category.Type
                    "inputs", !|InputParamMapper
                    "attrs", !|AttrParamMapper
                    "outputs", T<string[]>
                    "custom", OpExecutor
                ]
            }

        let BaseCallback =  
            Class "BaseCallback" 
            |+> Instance [
                "validationData" =@ TensorOrTensorArray
                "params" =@ T<obj>

                "setParams" => T<obj>?``params`` ^-> T<unit>
                "onEpochBegin" => T<int>?epoch * !?T<obj>?log ^-> T<unit>
                "onEpochEnd" => T<int>?epoch * !?T<obj>?log ^-> T<unit>
                "onBatchBegin" => T<int>?epoch * !?T<obj>?log ^-> T<unit>
                "onBatchEnd" => T<int>?epoch * !?T<obj>?log ^-> T<unit>
                "onTrainBegin" => !?T<obj>?log ^-> T<unit>
                "onTrainEnd" => !?T<obj>?log ^-> T<unit>
                "setModel" => T<obj>?model ^-> T<unit>
            ]

        let Callback = CustomCallbackArgs + !| CustomCallbackArgs + !| BaseCallback

        let History = 
            Class "History" 
            |=> Inherits BaseCallback
            |+> Instance [
                "epoch" =@ T<int[]>
                "history" =@ T<obj> //property read-write

                "syncData" => T<unit> ^-> T<unit>
            ]

        let ModelCompileArgs = 
            Pattern.Config "ModelCompileArgs" {
                Required = [
                    "optimizer", T<string> + Optimizer
                    "loss", T<string> + !| T<string> + T<obj> + LossOrMetricFn + !| LossOrMetricFn
                    "metrics", T<string> + LossOrMetricFn
                ]
                Optional = []
            }

        let ModelEvaluateArgs = 
            Pattern.Config "ModelEvaluateArgs" {
                Required = []
                Optional = [
                    "batchSize", T<int>
                    "verbose", T<int>
                    "sampleWeight", Tensor.Type
                    "steps", T<int>
                ]
            }

        let ModelFitArgs = 
            Pattern.Config "ModelFitArgs" {
                Required = [
                    "batchSize", T<int>
                    "epochs", T<int>
                    "verbose", T<int>
                ]
                Optional = [
                    "calback", Callback
                    "validationSplit", T<float>
                    "validationData", !| (Tensor + !|Tensor) 
                    "shuffle", T<bool>
                    "classWeight", T<obj>
                    "sampleWeight", Tensor.Type
                    "initialEpoch", T<int>
                    "stepsPerEpoch", T<int>
                    "validationSteps", T<int>
                    "yieldEvery", T<string> + T<int>
                ]
            }

        let ModelFitDatasetArgs = 
            Generic - fun t ->
                Pattern.Config "ModelFitDatasetArgs" {
                    Required = []
                    Optional = [
                        "batchesPerEpoch", T<int>
                        "epochs", T<int>
                        "verbose", T<int>
                        "callbacks", Callback
                        "validationData", !| TensorOrArrayOrMap + Dataset[t]
                        "validationBatchSize", T<int>
                        "validationBatches", T<int>
                        "yieldEvery", T<string> + T<int>
                        "initialEpoch", T<int>
                        "classWeight", T<obj> + T<obj[]>
                    ]
                }

        let SequentialArgs = 
            Pattern.Config "SequentialArgs" {
                Required = []
                Optional = [
                    "layers", !| Layer
                    "name", T<string>
                ]
            }

        let SymbolicTensorArray = !| SymbolicTensor

        SymbolicTensor
            |+> Static [
                Constructor (DataType?dtype * Shape?shape * Layer?sourceLayer * SymbolicTensorArray?inputs * T<obj>?callArgs * !? T<string>?name * !? T<int>?outputTensorIndex)
                
            ] |> ignore

        let ContainerArgs =
            Pattern.Config "ContainerArgs" {
                Required = [
                    "inputs", SymbolicTensor + !|SymbolicTensor
                    "outputs", SymbolicTensor + !|SymbolicTensor
                ]
                Optional = [
                    "name", T<string>
                ]
            }          

        let EncodeWeightsTensors = NamedTensorMap + !| NamedTensor

        let encodeWeights = EncodeWeightsTensors?tensors * !?WeightGroup?group ^-> T<Promise<_>>[EncodeWeightsResult]

        let IO = 
            copyModel + listModels + moveModel + removeModel +
            browserFiles + browserHTTPRequest +
            CompositeArrayBuffer + concatenateArrayBuffers +
            decodeWeights + decodeWeightsStream +
            encodeWeights + fromMemory + fromMemorySync +
            getLoadHandlers + getModelArtifactsForJSON + getModelArtifactsForJSONSync + 
            getModelArtifactsInfoForJSON + getSaveHandlers + getWeightSpecs +
            http + IOHandler + IOHandlerSync +
            isHTTPScheme + LoadHandler + LoadOptions + loadWeights +
            ModelArtifacts + ModelArtifactsInfo + ModelJSON + ModelStoreManager +
            OnProgressCallback +
            registerLoadRouter + registerSaveRouter + RequestDetails +
            SaveConfig + SaveHandler + SaveResult + TrainingConfig +
            WeightData + WeightGroup + weightsLoaderFactory +
            WeightsManifestConfig + WeightsManifestEntry + withSaveHandler + withSaveHandlerSync

        let InferenceModel =
            Class "InferenceModel"
            |+> Instance [
                "inputs" =? !| ModelTensorInfo
                "outputs" =? !| ModelTensorInfo

                "predict" => TensorOrArrayOrMap?inputs * ModelPredictConfig?config ^-> TensorOrArrayOrMap
                "execute" => TensorOrArrayOrMap?inputs * (T<string> + !| T<string>)?outputs ^-> TensorOrTensorArray
            ]

        let GraphModel =
            Class "Tf.GraphModel"
            |+> Instance [ 
                "load" => T<unit> ^-> T<Promise<bool>>
                "loadSync" => ModelArtifacts?artifacts ^-> T<bool>
                "save" => HandleOrString?handlerOrURL * !? SaveConfig?config ^-> T<Promise<_>>[SaveResult]
                "predict" => TensorOrArrayOrMap * !? ModelPredictConfig?config ^-> TensorOrArrayOrMap
                "predictAsync" => TensorOrArrayOrMap?input * !? ModelPredictConfig?config ^-> T<Promise<_>>[TensorOrArrayOrMap]
                "execute" => TensorOrArrayOrMap?input * !? (T<string> + !| T<string>)?outputs ^-> (Tensor + !| Tensor)
                "executeAsync" => TensorOrArrayOrMap?input * !? (T<string> + !| T<string>)?outputs ^-> T<Promise<_>>[(Tensor + !| Tensor)]
                "getIntermediateTensors" => T<unit> ^-> T<obj>
                "disposeIntermediateTensors" => T<unit> ^-> T<unit>
                "dispose" => T<unit> ^-> T<unit>
            ]

        let LayersModel =
            Class "tf.LayersModel"
            |+> Static [
                Constructor ContainerArgs?args
            ]
            |+> Instance [
                "optimizer_" =@ Optimizer 
                "isOptimizerOwned" =@ T<bool>
                "loss" =@ StringOrArray + !|LossOrMetricFn + LossOrMetricFn + T<obj[]>
                "lossFunctions" =@ !|LossOrMetricFn 
                "history" =@ History 
                "stopTraining_" =@ T<bool>
                "isTraining" =@ T<bool>
                "metrics" =@ StringOrArray + !|LossOrMetricFn + LossOrMetricFn + T<obj[]>
                "metricsNames" =@ T<string[]>
                "metricsTensors" =@ T<int[]> + !|LossOrMetricFn

                "summary" => !?T<int>?lineLength * T<int[]>?positions * !?PrintFnType?printFn ^-> T<unit>
                "compile" => ModelCompileArgs?args ^-> T<unit>
                "checkTrainableWeightsConsistency" => T<unit> ^-> T<unit>
                "evaluate" => TensorOrTensorArray?x * TensorOrTensorArray?y * !?ModelEvaluateArgs?args ^-> TensorOrTensorArray
                "evaluateDataset" => Dataset[T<obj>]?dataset * !?ModelEvaluateDatasetArgs?args ^-> T<Promise<_>>[TensorOrTensorArray]
                "execute" => (TensorOrTensorArray + NamedTensorMap)?inputs * StringOrArray?outputs ^-> TensorOrTensorArray
                "predict" => TensorOrTensorArray?x * !?ModelPredictArgs?args ^-> TensorOrTensorArray
                "predictOnBatch" => TensorOrTensorArray?x ^-> TensorOrTensorArray
                "standardizeUserDataXY" => (TensorOrTensorArray + T<obj[]>)?y * (TensorOrTensorArray + T<obj[]>)?y * !?T<bool>?checkBatchAxis * !?T<int>?batchSize ^-> !| (!| Tensor)
                "standardizeUserData" => (TensorOrTensorArray + T<obj[]>)?y * (TensorOrTensorArray + T<obj[]>)?y * !?(TensorOrTensorArray + T<obj[]>)?sampleWeight * !?(T<obj[][]> + T<obj[]>)?classWeight * !?T<bool>?checkBatchAxis * !?T<int>?batchSize ^-> T<Promise<_>>[!| (!| Tensor)]
                "getDedupedMetricsNames" => T<unit> ^-> T<string[]>
                "makeTrainFunction" => T<unit> ^-> ((!|Tensor)?data ^-> !|Tensor)
                "fit" => TensorOrArrayOrMap?x * TensorOrArrayOrMap?y * !?ModelFitArgs?args ^-> T<Promise<_>>[History]
                Generic - fun t ->
                    "fitDataset" => Dataset[t]?dataset * !?ModelFitDatasetArgs[t]?args ^-> T<Promise<_>>[History]
                "trainOnBatch" => TensorOrArrayOrMap?x * TensorOrArrayOrMap?y ^-> T<Promise<_>>[T<int> + T<int[]>]
                "getNamedWeights" => !?SaveConfig?config ^-> !|NamedTensor
                "getTrainingConfig" => T<unit> ^-> TrainingConfig
                "loadTrainingConfig" => TrainingConfig?trainingConfig ^-> T<unit>
                "save" => HandleOrString?handlerOrURL * !?SaveConfig?config ^-> T<Promise<_>>[SaveResult]
                "setUserDefinedMetadata" => T<obj>?userDefinedMetadata ^-> T<unit>
                "getUserDefinedMetadata" => T<unit> ^-> T<obj>
                "getLayer" => !?(T<string> + T<int>)?nameOrIndex * !?T<int>?index ^-> Layer
            ]

        let Sequential = 
            Class "tf.Sequential"
            |=> Inherits LayersModel
            |+> Static [
                Constructor !? SequentialArgs?args
            ]
            |+> Instance [
                "add" => Layer?layer ^-> T<unit>
                "pop" => T<unit> ^-> T<unit>
            ]

        let Function = 
            Class "TFFunction"
            |=> Inherits LayersModel

        let RNNCell =
            Class "tf.RNNCell"
            |=> Inherits Layer
            |+> Instance [
                "stateSize" =@ (T<int> + T<int[]>)
                "dropoutMask" =@ TensorOrTensorArray
                "recurrentDropoutMask" =@ TensorOrTensorArray
            ]

        let ELULayerArgs = 
            Pattern.Config "ELULayerArgs" {
                Required = []
                Optional = [
                    "alpha", T<float>
                ]
            }
            |=> Inherits LayerArgs

        let LeakyReLULayerArgs = 
            Class "LeakyReLULayerArgs"
            |=> Inherits ELULayerArgs

        let ReLULayerArgs =
            Pattern.Config "ReLULayerArgs" {
                Required = []
                Optional = [
                    "maxValue", T<int>
                ]
            }
            |=> Inherits LayerArgs

        let SoftmaxLayerArgs =
            Pattern.Config "SoftmaxLayerArgs" {
                Required = []
                Optional = [
                    "axis", IntOrIntArray
                ]
            }
            |=> Inherits LayerArgs

        let ThresholdedReLULayerArgs =
            Pattern.Config "ThresholdedReLULayerArgs" {
                Required = []
                Optional = [
                    "theta", T<int>
                ]
            }
            |=> Inherits LayerArgs

        let PReLULayerArgs =
            Pattern.Config "PReLULayerArgs" {
                Required = []
                Optional = [
                    "alphaInitializer", !|Initializer + Initializer
                    "alphaRegularizer", Regularizer.Type
                    "alphaConstraint", Constraint.Type
                    "sharedAxes", IntOrIntArray
                ]
            }
            |=> Inherits LayerArgs

        let ActivationLayerArgs =
            Pattern.Config "ActivationLayerArgs" {
                Required = []
                Optional = [
                    "activation", ActivationIdentifier.Type
                ]
            }
            |=> Inherits LayerArgs

        let DenseLayerArgs = 
            Pattern.Config "DenseLayerArgs" {
                Required = [
                    "units", T<int>
                ]
                Optional = [
                    "activation", ActivationIdentifier.Type
                    "useBias", T<bool>
                    "kernelInitializer", InitializerIdentifier + Initializer
                    "biasInitializer", InitializerIdentifier + Initializer
                    "inputDim", T<int>
                    "kernelConstraint", ConstraintIdentifier + Constraint
                    "biasConstraint", ConstraintIdentifier + Constraint
                    "kernelRegularizer", RegularizerIdentifier + Regularizer
                    "biasRegularizer", RegularizerIdentifier + Regularizer
                    "activityRegularizer", RegularizerIdentifier + Regularizer
                ]
            }
            |=> Inherits LayerArgs

        let DropoutLayerArgs =
            Pattern.Config "DropoutLayerArgs" {
                Required = [
                    "rate", T<float>  
                ]
                Optional = [
                    "noiseShape", !| T<int>  
                    "seed", T<int> 
                ]
            }
            |=> Inherits LayerArgs

        let EmbeddingLayerArgs =
            Pattern.Config "EmbeddingLayerArgs" {
                Required = [
                    "inputDim", T<int>
                    "outputDim", T<int>
                ]
                Optional = [
                    "embeddingsInitializer", InitializerIdentifier + Initializer  
                    "embeddingsRegularizer", RegularizerIdentifier + Regularizer
                    "activityRegularizer", RegularizerIdentifier + Regularizer
                    "embeddingsConstraint", ConstraintIdentifier + Constraint
                    "maskZero", T<bool> 
                    "inputLength", T<int> + !|T<int>
                ]
            }
            |=> Inherits LayerArgs

        let FlattenLayerArgs =
            Pattern.Config "FlattenLayerArgs" {
                Required = []
                Optional = [
                    "dataFormat", T<string>
                ]
            }
            |=> Inherits LayerArgs

        let PermuteLayerArgs =
            Pattern.Config "PermuteLayerArgs" {
                Required = [
                    "dims", T<int[]>
                ]
                Optional = []
            }
            |=> Inherits LayerArgs

        let ReshapeLayerArgs =
            Pattern.Config "ReshapeLayerArgs" {
                Required = [
                    "targetShape", Shape
                ]
                Optional = []
            }
            |=> Inherits LayerArgs

        let RepeatVectorLayerArgs =
            Pattern.Config "RepeatVectorLayerArgs" {
                Required = [
                    "n", T<int>
                ]
                Optional = []
            }
            |=> Inherits LayerArgs

        let SpatialDropout1DLayerConfig =
            Pattern.Config "SpatialDropout1DLayerConfig" {
                Required = [
                    "rate", T<int>
                ]
                Optional = [
                    "seed", T<int>
                ]
            }
            |=> Inherits LayerArgs

        let BaseConvLayerArgs =
            Pattern.Config "BaseConvLayerArgs" {
                Required = [
                    "kernelSize", T<int> + !|T<int>  
                ]
                Optional = [
                    "strides", T<int> + !|T<int>  
                    "padding", T<string>  
                    "dataFormat", T<string> 
                    "dilationRate", T<int> + !|T<int>  
                    "activation", T<string>  
                    "useBias", T<bool> 
                    "kernelInitializer", InitializerIdentifier + Initializer 
                    "biasInitializer", InitializerIdentifier + Initializer 
                    "kernelConstraint", ConstraintIdentifier + Constraint  
                    "biasConstraint", ConstraintIdentifier + Constraint  
                    "kernelRegularizer", RegularizerIdentifier + Regularizer
                    "biasRegularizer", RegularizerIdentifier + Regularizer
                    "activityRegularizer", RegularizerIdentifier + Regularizer
                ]
            }
            |=> Inherits LayerArgs

        let ConvLayerArgs = 
            Pattern.Config "ConvLayerArgs" {
                Required = [
                    "filters", T<int>
                ]
                Optional = []
            }
            |=> Inherits BaseConvLayerArgs

        let Cropping2DLayerArgs = 
            Pattern.Config "Cropping2DLayerArgs" {
                Required = [
                    "cropping", T<int> + T<int[]>
                ]
                Optional = [
                    "dataFormat", T<string>
                ]
            }
            |=> Inherits LayerArgs

        let DepthwiseConv2DLayerArgs =
            Pattern.Config "DepthwiseConv2DLayerArgs" {
                Required = [
                    "kernelSize", T<int> + !|T<int>  
                ]
                Optional = [
                    "depthMultiplier", T<int>  
                    "depthwiseInitializer", InitializerIdentifier + Initializer  
                    "depthwiseConstraint", ConstraintIdentifier + Constraint  
                    "depthwiseRegularizer", RegularizerIdentifier + Regularizer  
                ]
            }
            |=> Inherits BaseConvLayerArgs

        let SeparableConvLayerArgs =
            Pattern.Config "SeparableConvLayerArgs" {
                Required = [
                    "kernelSize", T<int> + !|T<int> 
                ]
                Optional = [
                    "depthMultiplier", T<int>  
                    "depthwiseInitializer", InitializerIdentifier + Initializer  
                    "pointwiseInitializer", InitializerIdentifier + Initializer  
                    "depthwiseRegularizer", RegularizerIdentifier + Regularizer 
                    "pointwiseRegularizer", RegularizerIdentifier + Regularizer 
                    "depthwiseConstraint", ConstraintIdentifier + Constraint  
                    "pointwiseConstraint", ConstraintIdentifier + Constraint  
                ]
            }
            |=> Inherits ConvLayerArgs

        let UpSampling2DLayerArgs =
            Pattern.Config "UpSampling2DLayerArgs" {
                Required = []
                Optional = [
                    "size", !|T<int>  
                    "dataFormat", T<string>  
                    "interpolation", T<string>
                ]
            }
            |=> Inherits LayerArgs

        let ConcatenateLayerArgs = 
            Pattern.Config "ConcatenateLayerArgs" {
                Required = []
                Optional = [
                    "axis", T<int>  
                ]
            }
            |=> Inherits LayerArgs

        let DotLayerArgs = 
            Pattern.Config "DotLayerArgs" {
                Required = [
                    "axis", T<int> + T<int[]>
                ]
                Optional = [
                     "normalize", T<bool>
                ]
            }
            |=> Inherits LayerArgs

        let BatchNormalizationLayerArgs =
            Pattern.Config "BatchNormalizationLayerArgs" {
                Required = []
                Optional = [
                    "axis", T<int>  
                    "momentum", T<float>  
                    "epsilon", T<float>  
                    "center", T<bool>  
                    "scale", T<bool>  
                    "betaInitializer", InitializerIdentifier + Initializer  
                    "gammaInitializer", InitializerIdentifier + Initializer  
                    "movingMeanInitializer", InitializerIdentifier + Initializer  
                    "movingVarianceInitializer", InitializerIdentifier + Initializer 
                    "betaConstraint", ConstraintIdentifier + Constraint  
                    "gammaConstraint", ConstraintIdentifier + Constraint  
                    "betaRegularizer", RegularizerIdentifier + Regularizer  
                    "gammaRegularizer", RegularizerIdentifier + Regularizer  
                ]
            }
            |=> Inherits LayerArgs

        let LayerNormalizationLayerArgs =
            Pattern.Config "LayerNormalizationLayerArgs" {
                Required = []
                Optional = [
                    "axis", T<int> + !|T<int>  
                    "epsilon", T<float>  
                    "center", T<bool>  
                    "scale", T<bool>  
                    "betaInitializer", InitializerIdentifier + Initializer  
                    "gammaInitializer", InitializerIdentifier + Initializer  
                    "betaRegularizer", RegularizerIdentifier + Regularizer  
                    "gammaRegularizer", RegularizerIdentifier + Regularizer  
                ]
            }
            |=> Inherits LayerArgs

        let Pooling1DLayerArgs = 
            Pattern.Config "Pooling1DLayerArgs" {
                Required = []
                Optional = [
                    "poolSize", T<int> + T<int[]>
                    "strides", T<int> + T<int[]>
                    "padding", T<string>
                ]
            }
            |=> Inherits LayerArgs

        let Pooling2DLayerArgs = 
            Pattern.Config "Pooling2DLayerArgs" {
                Required = []
                Optional = [
                    "poolSize", T<int> + T<int[]>
                    "strides", T<int> + T<int[]>
                    "padding", T<string>
                    "dataFormat", T<string>
                ]
            }
            |=> Inherits LayerArgs

        let Pooling3DLayerArgs = 
            Pattern.Config "Pooling3DLayerArgs" {
                Required = []
                Optional = [
                    "poolSize", T<int> + T<int[]>
                    "strides", T<int> + T<int[]>
                    "padding", T<string>
                    "dataFormat", T<string>
                ]
            }
            |=> Inherits LayerArgs

        let GlobalPooling2DLayerArgs = 
            Pattern.Config "GlobalPooling2DLayerArgs" {
                Required = []
                Optional = [
                    "dataFormat", T<string>
                ]
            }
            |=> Inherits LayerArgs

        let BaseRNNLayerArgs =
            Pattern.Config "BaseRNNLayerArgs" {
                Required = []
                Optional = [
                    "cell", RNNCell + !|RNNCell
                    "returnSequences", T<bool>
                    "returnState", T<bool>
                    "goBackwards", T<bool>
                    "stateful", T<bool>
                    "unroll", T<bool>
                    "inputDim", T<int>
                    "inputLength", T<int>
                ]
            }
            |=> Inherits LayerArgs

        let ConvRNN2DCellArgs =
            Pattern.Config "ConvRNN2DCellArgs" {
                Required = [
                    "filters", T<int>
                    "kernelSize", T<int> + T<int[]>
                ]
                Optional = [
                    "strides", !|T<int> + T<int[]>
                    "padding", T<string>
                    "dataFormat", T<string>
                    "dilationRate", T<int> + T<int[]> 
                    "activation", T<string>
                    "useBias", T<bool>
                    "kernelInitializer", InitializerIdentifier + Initializer
                    "recurrentInitializer", InitializerIdentifier + Initializer
                    "biasInitializer", InitializerIdentifier + Initializer
                    "kernelRegularizer", RegularizerIdentifier + Regularizer
                    "recurrentRegularizer", RegularizerIdentifier + Regularizer
                    "biasRegularizer", RegularizerIdentifier + Regularizer
                    "kernelConstraint", ConstraintIdentifier + Constraint
                    "recurrentConstraint", ConstraintIdentifier + Constraint
                    "biasConstraint", ConstraintIdentifier + Constraint
                    "dropout", T<float>
                    "recurrentDropout", T<float>
                    "dropoutFunc", T<Function>
                ]
            }
            |=> Inherits LayerArgs

        let SimpleRNNCellLayerArgs =
            Pattern.Config "SimpleRNNCellLayerArgs" {
                Required = []
                Optional = [
                    "units", T<int>
                    "activation", T<string>
                    "useBias", T<bool>
                    "kernelInitializer", InitializerIdentifier + Initializer
                    "recurrentInitializer", InitializerIdentifier + Initializer
                    "biasInitializer", InitializerIdentifier + Initializer
                    "kernelRegularizer", RegularizerIdentifier + Regularizer
                    "recurrentRegularizer", RegularizerIdentifier + Regularizer
                    "biasRegularizer", RegularizerIdentifier + Regularizer
                    "kernelConstraint", ConstraintIdentifier + Constraint
                    "recurrentConstraint", ConstraintIdentifier + Constraint
                    "biasConstraint", ConstraintIdentifier + Constraint
                    "dropout", T<float>
                    "recurrentDropout", T<float>
                    "dropoutFunc", T<Function>
                ]
            }
            |=> Inherits LayerArgs

        let StackedRNNCellsArgs =
            Pattern.Config "StackedRNNCellsArgs" {
                Required = [
                    "cells", !| RNNCell
                ]
                Optional = []
            }
            |=> Inherits LayerArgs

        let SimpleRNNLayerArgs =
            Pattern.Config "SimpleRNNLayerArgs" {
                Required = []
                Optional = [
                    "units", T<int>
                    "activation", T<string> 
                    "useBias", T<bool> 
                    "kernelInitializer", InitializerIdentifier + Initializer 
                    "recurrentInitializer", InitializerIdentifier + Initializer 
                    "biasInitializer", InitializerIdentifier + Initializer 
                    "kernelRegularizer", RegularizerIdentifier + Regularizer 
                    "recurrentRegularizer", RegularizerIdentifier + Regularizer 
                    "biasRegularizer", RegularizerIdentifier + Regularizer 
                    "kernelConstraint", ConstraintIdentifier + Constraint 
                    "recurrentConstraint", ConstraintIdentifier + Constraint 
                    "biasConstraint", ConstraintIdentifier + Constraint
                    "dropout", T<float> 
                    "recurrentDropout", T<float>
                    "dropoutFunc", T<Function> 
                ]
            }

        let LSTMLayerArgs =
            Pattern.Config "LSTMLayerArgs" {
                Required = []
                Optional = [
                    "recurrentActivation", T<string>
                    "unitForgetBias", T<bool>
                    "implementation", T<int>
                ]
            }
            |=> Inherits SimpleRNNLayerArgs

        let ConvRNN2DLayerArgs = 
            Class "ConvRNN2DLayerArgs"
            |=> Inherits BaseRNNLayerArgs
            |=> Inherits ConvRNN2DCellArgs

        let ConvLSTM2DArgs = 
            Class "ConvLSTM2DArgs"
            |=> Inherits LSTMLayerArgs
            |=> Inherits ConvRNN2DLayerArgs

        let LSTMCellLayerArgs =
            Pattern.Config "LSTMCellLayerArgs" {
                Required = []
                Optional = [
                    "recurrentActivation", T<string> 
                    "unitForgetBias", T<bool> 
                    "implementation", T<int> 
                ]
            }
            |=> Inherits SimpleRNNCellLayerArgs

        let GRULayerArgs =
            Pattern.Config "GRULayerArgs" {
                Required = []
                Optional = [
                    "recurrentActivation", T<string> 
                    "implementation", T<int> 
                ]
            }
            |=> Inherits SimpleRNNLayerArgs

        let GRUCellLayerArgs =
            Pattern.Config "GRUCellLayerArgs" {
                Required = []
                Optional = [
                    "recurrentActivation", T<string> 
                    "implementation", T<int> 
                    "resetAfter", T<bool>
                ]
            }
            |=> Inherits SimpleRNNCellLayerArgs

        let WrapperLayerArgs =
            Pattern.Config "WrapperLayerArgs" {
                Required = ["layer", Layer.Type]
                Optional = []
            }
            |=> Inherits LayerArgs

        let RNNLayerArgs = 
            Pattern.Config "RNNLayerArgs" {
                Required = [
                    "cell", RNNCell + !| RNNCell
                ]
                Optional = []
            }
            |=> Inherits BaseRNNLayerArgs

        let InputLayerArgs =
            Pattern.Config "InputLayerArgs" {
                Required = []
                Optional = [
                    "inputShape", Shape
                    "batchSize", T<int>
                    "batchInputShape", Shape
                    "dtype", DataType
                    "sparse", T<bool>
                    "name", T<string>
                ]
            }

        let ZeroPadding2DLayerArgs = 
            Pattern.Config "ZeroPadding2DLayerArgs" {
                Required = []
                Optional = [
                    "padding", T<int> + T<int[]>
                    "dataFormat", T<string>
                ]
            }
            |=> Inherits LayerArgs

        let AlphaDropoutArgs = 
            Pattern.Config "AlphaDropoutArgs" {
                Required = ["rate", T<int>]
                Optional = [
                    "noiseShape", Shape
                ]
            }
            |=> Inherits LayerArgs

        let GaussianDropoutArgs = 
            Pattern.Config "GaussianDropoutArgs" {
                Required = ["rate", T<int>]
                Optional = []
            }
            |=> Inherits LayerArgs

        let GaussianNoiseArgs = 
            Pattern.Config "GaussianNoiseArgs" {
                Required = ["stddev", T<int>]
                Optional = []
            }
            |=> Inherits LayerArgs

        let MaskingArgs = 
            Pattern.Config "MaskingArgs" {
                Required = []
                Optional = ["maskValue", T<int>]
            }
            |=> Inherits LayerArgs

        let RescalingArgs = 
            Pattern.Config "RescalingArgs" {
                Required = ["scale", T<int>]
                Optional = ["offset", T<int>]
            }
            |=> Inherits LayerArgs

        let CenterCropArgs = 
            Pattern.Config "CenterCropArgs" {
                Required = []
                Optional = [
                    "height", T<int>
                    "width", T<int>
                ]
            }
            |=> Inherits LayerArgs

        let ResizingArgs = 
            Pattern.Config "ResizingArgs" {
                Required = [
                    "height", T<int>
                    "width", T<int>
                ]
                Optional = [
                    "interpolation", T<string>
                    "cropToAspectRatio", T<bool>
                ]
            }
            |=> Inherits LayerArgs

        let CategoryEncodingArgs = 
            Pattern.Config "CategoryEncodingArgs" {
                Required = [
                    "numTokens", T<int>
                ]
                Optional = [
                    "outputMode", T<string>
                ]
            }
            |=> Inherits LayerArgs

        let BaseRandomLayerArgs = 
            Pattern.Config "BaseRandomLayerArgs" {
                Required = []
                Optional = [
                    "seed", T<int>
                ]
            }
            |=> Inherits LayerArgs

        let RandomWidthArgs = 
            Pattern.Config "RandomWidthArgs" {
                Required = [
                    "factor", T<int> + T<int[]>
                ]
                Optional = [
                    "interpolation", T<string>
                    "seed", T<int>
                    "autoVectorize", T<bool>
                ]
            }
            |=> Inherits BaseRandomLayerArgs

        let RNN = 
            Class "RNN"
            |=> Inherits Layer
            |+> Static [
                Constructor RNNLayerArgs?args
            ]
            |+> Instance [
                "cell" =? RNNCell.Type
                "returnSequences" =? T<bool>
                "returnState" =? T<bool>
                "goBackwards" =? T<bool>
                "unroll" =? T<bool>
                "stateSpec" =? !|InputSpec

                "getState" => T<unit> ^-> !| Tensor
                "setSate" => Tensor?states ^-> T<unit>
            ]

        let BidirectionalLayerArgs =
            Pattern.Config "BidirectionalLayerArgs" {
                Required = ["layer", RNN.Type]
                Optional = [
                    "mergeMode", T<string>
                ]
            }
            |=> Inherits WrapperLayerArgs

        let ConvLSTM2DCellArgs = 
            Class "ConvLSTM2DCellArgs"
            |=> Inherits LSTMCellLayerArgs
            |=> Inherits ConvRNN2DCellArgs 

        let BaseConv = 
            Class "BaseConv" 
            |=> Inherits Layer
            |+> Static [
                Constructor (T<int>?rank * BaseConvLayerArgs?args)
            ]

        let Conv = 
            Class "Conv" 
            |=> Inherits BaseConv
            |+> Static [
                Constructor (T<int>?rank * ConvLayerArgs?args)
            ]

        let Conv1D = 
            Class "Conv1D" 
            |=> Inherits Conv
            |+> Static [
                Constructor (ConvLayerArgs?args)
            ] 

        let Conv2D = 
            Class "Conv2D" 
            |=> Inherits Conv
            |+> Static [
                Constructor (ConvLayerArgs?args)
            ] 

        let Conv2DTranspose = 
            Class "Conv2DTranspose" 
            |=> Inherits Conv2D
            |+> Static [
                Constructor (ConvLayerArgs?args)
            ] 

        let Conv3D = 
            Class "Conv3D" 
            |=> Inherits Conv
            |+> Static [
                Constructor (ConvLayerArgs?args)
            ]

        let Cropping2D = 
            Class "Cropping2D" 
            |=> Inherits Layer
            |+> Static [
                Constructor (Cropping2DLayerArgs?args)
            ] 

        let DepthwiseConv2D = 
            Class "DepthwiseConv2D" 
            |=> Inherits BaseConv
            |+> Static [
                Constructor (DepthwiseConv2DLayerArgs?args)
            ] 

        let SeparableConv = 
            Class "SeparableConv" 
            |=> Inherits Conv
            |+> Static [
                Constructor (T<int>?rank * !?SeparableConvLayerArgs?config)
            ]  

        let SeparableConv2D = 
            Class "SeparableConv2D" 
            |=> Inherits SeparableConv
            |+> Static [
                Constructor (!?SeparableConvLayerArgs?args)
            ] 

        let UpSampling2D = 
            Class "UpSampling2D" 
            |=> Inherits Layer
            |+> Static [
                Constructor (UpSampling2DLayerArgs?args)
            ] 

        let ELU = 
            Class "ELU"
            |=> Inherits Layer
            |+> Static [
                Constructor !?ELULayerArgs?args
            ]

        let LeakyReLU = 
            Class "LeakyReLU"
            |=> Inherits Layer
            |+> Static [
                Constructor !?LeakyReLULayerArgs?args
            ]

        let PReLU = 
            Class "PReLU"
            |=> Inherits Layer
            |+> Static [
                Constructor !?PReLULayerArgs?args
            ]

        let ReLU = 
            Class "ReLU"
            |=> Inherits Layer
            |+> Static [
                Constructor !?ReLULayerArgs?args
            ]

        let Softmax = 
            Class "Softmax"
            |=> Inherits Layer
            |+> Instance [
                Constructor !?SoftmaxLayerArgs?args
            ]

        let ThresholdedReLU = 
            Class "ThresholdedReLU"
            |=> Inherits Layer
            |+> Instance [
                Constructor !?ThresholdedReLULayerArgs?args
            ]

        let Activation = 
            Class "Activation"
            |=> Inherits Layer
            |+> Instance [
                Constructor ActivationLayerArgs?args
            ]

        let Dense = 
            Class "Dense"
            |=> Inherits Layer
            |+> Instance [
                Constructor DenseLayerArgs?args
            ]

        let Dropout = 
            Class "Dropout"
            |=> Inherits Layer
            |+> Instance [
                Constructor DropoutLayerArgs?args
            ]

        let Embedding = 
            Class "Embedding"
            |=> Inherits Layer
            |+> Instance [
                Constructor EmbeddingLayerArgs?args
            ]

        let Flatten = 
            Class "Flatten"
            |=> Inherits Layer
            |+> Instance [
                Constructor !?FlattenLayerArgs?args
            ]

        let Permute = 
            Class "Permute"
            |=> Inherits Layer
            |+> Instance [
                Constructor PermuteLayerArgs?args
            ]

        let RepeatVector = 
            Class "RepeatVector"
            |=> Inherits Layer
            |+> Instance [
                Constructor RepeatVectorLayerArgs?args
            ]

        let Reshape = 
            Class "Reshape"
            |=> Inherits Layer
            |+> Instance [
                Constructor ReshapeLayerArgs?args
            ]

        let SpatialDropout1D = 
            Class "SpatialDropout1D"
            |=> Inherits Layer
            |+> Instance [
                Constructor SpatialDropout1DLayerConfig?args
            ]

        let Merge = 
            Class "Merge" 
            |=> Inherits Layer
            |+> Static [
                Constructor !?LayerArgs?args
            ]
            |+> Instance [
                "mergeFunction" => (!|Tensor)?inputs ^-> Tensor
            ]

        let Add = 
            Class "Add" 
            |=> Inherits Merge
            |+> Static [
                Constructor !?LayerArgs?args
            ]

        let Average = 
            Class "Average" 
            |=> Inherits Merge
            |+> Static [
                Constructor !?LayerArgs?args
            ]

        let Concatenate = 
            Class "Concatenate" 
            |=> Inherits Merge
            |+> Static [
                Constructor !?ConcatenateLayerArgs?args
            ]

        let Dot = 
            Class "Dot" 
            |=> Inherits Merge
            |+> Static [
                Constructor DotLayerArgs?args
            ]

        let Maximum = 
            Class "Maximum" 
            |=> Inherits Merge
            |+> Static [
                Constructor !?LayerArgs?args
            ]

        let Minimum = 
            Class "Minimum" 
            |=> Inherits Merge
            |+> Static [
                Constructor !?LayerArgs?args
            ]

        let Multiply = 
            Class "Multiply" 
            |=> Inherits Merge
            |+> Static [
                Constructor !?LayerArgs?args
            ]

        let BatchNormalization = 
            Class "BatchNormalization" 
            |=> Inherits Layer
            |+> Static [
                Constructor !?BatchNormalizationLayerArgs?args
            ]

        let LayerNormalization = 
            Class "LayerNormalization" 
            |=> Inherits Layer
            |+> Static [
                Constructor !?LayerNormalizationLayerArgs?args
            ]

        let Pooling1D = 
            Class "Pooling1D" 
            |=> Inherits Layer
            |+> Static [
                Constructor Pooling1DLayerArgs?args
            ]

        let AveragePooling1D = 
            Class "AveragePooling1D" 
            |=> Inherits Pooling1D
            |+> Static [
                Constructor Pooling1DLayerArgs?args
            ]
            |+> Instance [
                "poolingFunction" => Tensor?input * T<int[]>?poolSize * T<int[]>?strides * T<string>?padding * T<string>?dataFormat ^-> Tensor 
            ]

        let Pooling2D = 
            Class "Pooling2D" 
            |=> Inherits Layer
            |+> Static [
                Constructor Pooling2DLayerArgs?args
            ]

        let AveragePooling2D = 
            Class "AveragePooling2D" 
            |=> Inherits Pooling2D
            |+> Static [
                Constructor Pooling2DLayerArgs?args
            ]
            |+> Instance [
                "poolingFunction" => Tensor?input * T<int[]>?poolSize * T<int[]>?strides * T<string>?padding * T<string>?dataFormat ^-> Tensor 
            ]

        let Pooling3D = 
            Class "Pooling3D" 
            |=> Inherits Layer
            |+> Static [
                Constructor Pooling3DLayerArgs?args
            ]

        let AveragePooling3D = 
            Class "AveragePooling3D" 
            |=> Inherits Pooling2D
            |+> Static [
                Constructor Pooling3DLayerArgs?args
            ]
            |+> Instance [
                "poolingFunction" => Tensor?input * T<int[]>?poolSize * T<int[]>?strides * T<string>?padding * T<string>?dataFormat ^-> Tensor 
            ]

        let GlobalPooling1D = 
            Class "GlobalPooling1D" 
            |=> Inherits Layer
            |+> Static [
                Constructor LayerArgs?args
            ]

        let GlobalAveragePooling1D = 
            Class "GlobalAveragePooling1D" 
            |=> Inherits GlobalPooling1D
            |+> Static [
                Constructor !?LayerArgs?args
            ]

        let GlobalPooling2D = 
            Class "GlobalPooling2D" 
            |=> Inherits Layer
            |+> Static [
                Constructor GlobalPooling2DLayerArgs?args
            ]

        let GlobalAveragePooling2D = 
            Class "GlobalAveragePooling2D" 
            |=> Inherits GlobalPooling2D

        let GlobalMaxPooling1D = 
            Class "GlobalMaxPooling1D" 
            |=> Inherits GlobalPooling1D
            |+> Static [
                Constructor LayerArgs?args
            ]

        let GlobalMaxPooling2D = 
            Class "GlobalMaxPooling2D" 
            |=> Inherits GlobalPooling2D

        let MaxPooling1D = 
            Class "MaxPooling1D" 
            |=> Inherits Pooling1D
            |+> Static [
                Constructor Pooling1DLayerArgs?args
            ]
            |+> Instance [
                "poolingFunction" => Tensor?input * T<int[]>?poolSize * T<int[]>?strides * T<string>?padding * T<string>?dataFormat ^-> Tensor 
            ]

        let MaxPooling2D = 
            Class "MaxPooling2D" 
            |=> Inherits Pooling2D
            |+> Static [
                Constructor Pooling2DLayerArgs?args
            ]
            |+> Instance [
                "poolingFunction" => Tensor?input * T<int[]>?poolSize * T<int[]>?strides * T<string>?padding * T<string>?dataFormat ^-> Tensor 
            ]

        let MaxPooling3D = 
            Class "MaxPooling3D" 
            |=> Inherits Pooling3D
            |+> Static [
                Constructor Pooling3DLayerArgs?args
            ]
            |+> Instance [
                "poolingFunction" => Tensor?input * T<int[]>?poolSize * T<int[]>?strides * T<string>?padding * T<string>?dataFormat ^-> Tensor 
            ]

        let ConvRNN2D = 
            Class "ConvRNN2D" 
            |=> Inherits RNN
            |+> Static [
                Constructor ConvRNN2DLayerArgs?args
            ]

        let ConvLSTM2D = 
            Class "ConvLSTM2D" 
            |=> Inherits ConvRNN2D
            |+> Static [
                Constructor ConvLSTM2DArgs?args
            ] 

        let GRU =
            Class "GRU"
            |=> Inherits RNN
            |+> Static [
                Constructor (LSTMCellLayerArgs?args)
            ]

        let GRUCell =
            Class "GRUCell"
            |=> Inherits RNNCell
            |+> Static [
                Constructor (GRUCellLayerArgs?args)
            ]

        let LSTM =
            Class "LSTM"
            |=> Inherits RNN
            |+> Static [
                Constructor (LSTMLayerArgs?args)
            ]

        let LSTMCell =
            Class "LSTMCell"
            |=> Inherits RNNCell
            |+> Static [
                Constructor (LSTMCellLayerArgs?args)
            ]

        let ConvLSTM2DCell =
            Class "ConvLSTM2DCell"
            |=> Inherits RNNCell
            |+> Static [
                Constructor (ConvLSTM2DCellArgs?args)
            ]

        let SimpleRNN =
            Class "SimpleRNN"
            |=> Inherits RNN
            |+> Static [
                Constructor (SimpleRNNLayerArgs?args)
            ]

        let SimpleRNNCell =
            Class "SimpleRNNCell"
            |=> Inherits RNNCell
            |+> Static [
                Constructor (SimpleRNNCellLayerArgs?args)
            ]

        let StackedRNNCells =
            Class "StackedRNNCells"
            |=> Inherits RNNCell
            |+> Static [
                Constructor (StackedRNNCellsArgs?args)
            ]

        let Wrapper =
            Class "Wrapper"
            |=> Inherits Layer
            |+> Static [
                Constructor (WrapperLayerArgs?args)
            ]

        let Bidirectional =
            Class "Bidirectional"
            |=> Inherits Wrapper
            |+> Static [
                Constructor (WrapperLayerArgs?args)
            ]

        let TimeDistributed =
            Class "TimeDistributed"
            |=> Inherits Wrapper
            |+> Static [
                Constructor (BidirectionalLayerArgs?args)
            ]

        let InputLayer =
            Class "InputLayer"
            |=> Inherits Layer
            |+> Static [
                Constructor (InputLayerArgs?args)
            ]

        let ZeroPadding2D =
            Class "ZeroPadding2D"
            |=> Inherits Layer
            |+> Static [
                Constructor (!?ZeroPadding2DLayerArgs?args)
            ]

        let AlphaDropout =
            Class "AlphaDropout"
            |=> Inherits Layer
            |+> Static [
                Constructor (AlphaDropoutArgs?args)
            ]
            |+> Instance [
                "_getNoiseShape" => (Tensor + !|Tensor)?inputs ^-> T<unit> 
            ]

        let GaussianDropout =
            Class "GaussianDropout"
            |=> Inherits Layer
            |+> Static [
                Constructor (GaussianDropoutArgs?args)
            ]

        let GaussianNoise =
            Class "GaussianNoise"
            |=> Inherits Layer
            |+> Static [
                Constructor (GaussianNoiseArgs?args)
            ]

        let Masking =
            Class "Masking"
            |=> Inherits Layer
            |+> Static [
                Constructor (!?MaskingArgs?args)
            ]

        let Rescaling =
            Class "Rescaling"
            |=> Inherits Layer
            |+> Static [
                Constructor (RescalingArgs?args)
            ]

        let CenterCrop =
            Class "CenterCrop"
            |=> Inherits Layer
            |+> Static [
                Constructor (CenterCropArgs?args)
            ]
            |+> Instance [
                "centerCrop" => Tensor?inputs * T<int>?hBuffer * T<int>?wBuffer * T<int>?height * T<int>?width * T<int>?inputHeight * T<int>?inputWidth * DataType?dtype ^-> Tensor + !|Tensor
                "upsize" => Tensor?inputs * T<int>?height * T<int>?width * DataType?dtype ^-> Tensor + !|Tensor
            ]

        let Resizing =
            Class "Resizing"
            |=> Inherits Layer
            |+> Static [
                Constructor (ResizingArgs?args)
            ]

        let CategoryEncoding =
            Class "CategoryEncoding"
            |=> Inherits Layer
            |+> Static [
                Constructor (CategoryEncodingArgs?args)
            ]

        let RandomWidth =
            Class "RandomWidth"
            |=> Inherits Layer
            |+> Static [
                Constructor (RandomWidthArgs?args)
            ]

        let SGDOptimizer =  
            Class "SGDOptimizer"
            |=> Inherits Optimizer
            |+> Static [
                Constructor T<float>?learningRate
            ]
            |+> Instance [
                "setLearningRate" => T<float>?learningRate ^-> T<unit>
                "getConfig" => T<unit> ^-> ConfigDict
            ]

        let MomentumOptimizer =  
            Class "MomentumOptimizer"
            |=> Inherits SGDOptimizer
            |+> Static [
                Constructor (T<float>?learningRate)
            ]

        let AdagradOptimizer =  
            Class "AdagradOptimizer"
            |=> Inherits Optimizer
            |+> Static [
                Constructor (T<float>?learningRate)
            ]

        let AdamOptimizer =  
            Class "AdamOptimizer"
            |=> Inherits Optimizer
            |+> Static [
                Constructor (T<float>?learningRate * T<float>?beta1 * T<float>?beta2 * T<float>?epsilon)
            ]

        let AdadeltaOptimizer =  
            Class "AdadeltaOptimizer"
            |=> Inherits Optimizer
            |+> Static [
                Constructor (T<float>?learningRate * T<float>?rho * T<float>?epsilon)
            ]

        let RMSPropOptimizer =  
            Class "RMSPropOptimizer"
            |=> Inherits Optimizer
            |+> Static [
                Constructor (T<float>?learningRate * T<float>?decay * T<float>?momentum * T<float>?epsilon * T<bool>?centered)
            ]

        let AdamaxOptimizer =  
            Class "AdamaxOptimizer"
            |=> Inherits Optimizer
            |+> Static [
                Constructor (T<float>?learningRate * T<float>?rho * T<float>?epsilon * T<float>?decay)
            ]        

        let ProfileInfo = 
            Pattern.Config "ProfileInfo" {
                Required = [
                    "newBytes", T<int> 
                    "newTensors", T<int>
                    "peakBytes", T<int>
                    "kernels", !|KernelInfo
                    "result", TensorContainer
                    "kernelNames", T<string[]>
                ]
                Optional = []
            }

        let Platform =
            Class "Platform"
            |+> Instance [
                "fetch" => T<string>?path * !?T<Request>?requestInits * !?RequestDetails?optioins ^-> T<Promise<Response>>
                "now" => T<unit> ^-> T<int>
                "encode" => T<string>?text * T<string>?encoding ^-> T<Uint8Array>
                "decode" => T<Uint8Array>?bytes * T<string>?encoding ^-> T<string>
                "setTimeoutCustom" => !?T<Function>?functionRef * T<int>?delay ^-> T<unit>
                "isTypedArray" => TypedArray?a ^-> T<bool>
            ]

        let FlagEvaluationFn = (T<unit> ^-> FlagValue) + (T<unit> ^-> T<Promise<_>>[FlagValue])

        let Environment =
            Class "Environment"
            |+> Static [
                Constructor T<obj>?``global``
            ]
            |+> Instance [
                "setPlatform" => T<string>?platformName * Platform?platform ^-> T<unit>
                "registerFlag" => T<string>?flagName * FlagEvaluationFn?evaluationFn * (FlagValue?value ^-> T<unit>)?setHook ^-> T<unit>
                "getAsync" => T<string>?flagName ^-> T<Promise<_>>[FlagValue]
                "get" => T<string>?flagName ^-> FlagValue
                "getNumber" => T<string>?flagName ^-> T<float>
                "getBool" => T<string>?flagName ^-> T<bool>
                "getString" => T<string>?flagName ^-> T<string>
                "getFlags" => T<unit> ^-> Flags
                "set" => T<string>?flagName * FlagValue?value ^-> T<unit>
                "setFlag" => Flags?flags ^-> T<unit>
                "reset" => T<unit> ^-> T<unit>
            ]

        let KernelBackend = 
            Class "KernelBackend"
            |+> Instance [
                "refCount" => DataId?dataId ^-> T<int>
                "incRef" => DataId?dataId ^-> T<unit>
                "timerAvailable" => T<unit> ^-> T<bool>
                "time" => (T<unit> ^-> T<unit>)?f ^-> T<Promise<_>>[BackendTimingInfo]
                "read" => T<obj>?dataId ^-> T<Promise<_>>[BackendValues]
                "readSync" => T<obj>?dataId ^-> BackendValues
                "readToGPU" => DataId?dataId * DataToGPUOptions?options ^-> GPUData
                "numDataIds" => T<unit> ^-> T<int>
                "disposeData" => T<obj>?dataId * !?T<bool>?force ^-> T<bool>
                "write" => BackendValues?values * Shape?shape * DataType?dtype ^-> DataId
                "move" => DataId?dataId * BackendValues?values * Shape?shape * DataType?dtype * T<int>?refCount ^-> T<unit>
                "createTensorFromGPUData" => (WebGLData + WebGPUData)?values * Shape?shape * DataType?dtype ^-> Tensor
                "memory" => T<unit> ^-> Memory
                "floatPrecision" => T<unit> ^-> T<float>
                "epsilon" => T<unit> ^-> T<float>
                "dispose" => T<unit> ^-> T<unit>
            ]

        let UnitToKernelFunc = T<unit> ^-> KernelBackend + T<Promise<_>>[KernelBackend]
        let ScopeFn = T<unit> ^-> TensorContainer

        let ValueAndGradsResultReturn = 
            Pattern.Config "ValueAndGradsResultReturn" {
                Required = [
                    "value", Tensor.Type
                    "grads", !|Tensor
                ]
                Optional = []
            }

        let CustomGradientFuncResult = 
            Pattern.Config "CustomGradientFuncResult" {
                Required = [
                    "value", Tensor.Type
                    "gradFunc", Tensor?d * (!|Tensor)?saved ^-> Tensor + !|Tensor
                ]
                Optional = []
            }

        let IntOrTensor = T<int> + Tensor
        let WindowFn = T<int>?length ^-> Tensor
        let GradFunc = Tensor?x ^-> Tensor
        let GradsFunc = Tensor?x1 * !?(!|Tensor)?x2 ^-> Tensor
        let GradSaveFunc = (!|Tensor)?save ^-> T<unit>
        let CustomGradientFunc = (!|GradSaveFunc + !|Tensor)?inputs ^-> CustomGradientFuncResult

        let Engine =
            Class "Engine"
            |+> Static [
                Constructor Environment?ENV
            ]
            |+> Instance [
                "ready" => T<unit> ^-> T<Promise<unit>>
                "backendNames" => T<unit> ^-> T<string[]>
                "findBackend" => T<string>?backendName ^-> KernelBackend
                "findBackendFactory" => T<string>?backendName ^-> UnitToKernelFunc
                "registerBackend" => T<string>?backendName * UnitToKernelFunc?factory * T<int>?priority ^-> T<bool>
                "setBackend" => T<string>?backendName ^-> T<Promise<bool>>
                "removeBackend" => T<string>?backendName ^-> T<unit>
                "moveData" => KernelBackend?backend * DataId?dataId ^-> T<unit>
                "tidy" => (T<string> + ScopeFn)?nameOrFn * !?ScopeFn?fn ^-> TensorContainer
                "runKernel" => T<string>?kernelName * NamedTensorMap?inputs * T<obj[]>?attrs ^-> TensorOrTensorArray
                "makeTensor" => DataValues?values * Shape?shape * DataType?dtype * !?KernelBackend?backend ^-> Tensor
                "makeTensorFromDataId" => DataId?dataId * Shape?shape * DataType?dtype * !?KernelBackend?backend ^-> Tensor
                "makeTensorFromDataId" => TensorInfo?tensorInfo * !?KernelBackend?backend ^-> Tensor
                "makeVariable" => Tensor?initialValue * T<bool>?trainable * !?T<string>?name * DataType?dtype ^-> Variable
                "trackTensor" => Tensor?a * KernelBackend?backend ^-> T<unit>
                "incRef" => Tensor?a * KernelBackend?backend ^-> T<unit>
                "removeDataId" => DataId?dataId * KernelBackend?backend ^-> T<unit>
                "disposeTensor" => Tensor?a ^-> T<unit>
                "disposeVariables" => T<unit> ^-> T<unit>
                "disposeVariable" => Variable?v ^-> T<unit>
                "memory" => T<unit> ^-> MemoryInfo
                "profile" => (T<unit> ^-> TensorContainer + T<Promise<_>>[TensorContainer])?query ^-> T<Promise<_>>[ProfileInfo]
                "isTapeOn" => T<unit> ^-> T<bool>
                "keep" => Tensor?result ^-> Tensor
                "startScope" => !?T<string>?name ^-> T<unit>
                "endScope" => !?TensorContainer?result ^-> T<unit>
                "gradients" => (T<unit> ^-> Tensor)?f * (!|Tensor)?xs * !?Tensor?dy * T<bool>?allowNoGradients ^-> ValueAndGradsResultReturn
                "customGrad" => !?CustomGradientFunc?f ^-> (!|Tensor + !|GradSaveFunc) ^-> Tensor
                "readSync" => DataId?dataId ^-> BackendValues
                "read" => DataId?dataId ^-> T<Promise<_>>[BackendValues]
                "readToGPU" => DataId?dataId * !?DataToGPUOptions?options ^-> GPUData
                "time" => (T<unit> ^-> T<unit>)?query ^-> T<Promise<_>>[TimingInfo]
                "reset" => T<unit> ^-> T<unit>
            ]

        let GlorotUniformArgs = 
            Class "GlorotUniformArgs" 
            |=> Inherits GlorotNormalArgs

        let HeNormalArgs = 
            Class "HeNormalArgs" 
            |=> Inherits GlorotNormalArgs

        let HeUniformArgs = 
            Class "HeUniformArgs" 
            |=> Inherits GlorotNormalArgs

        let IdentityArgs = 
            Pattern.Config "IdentityArgs" {
                Required = []
                Optional = [
                    "gain", T<float>
                ]
            }

        let LeCunNormalArgs = 
            Class "LeCunNormalArgs" 
            |=> Inherits GlorotNormalArgs

        let LeCunUniformArgs = 
            Class "LeCunUniformArgs" 
            |=> Inherits GlorotNormalArgs

        let OrthogonalArgs = 
            Pattern.Config "OrthogonalArgs" {
                Required = []
                Optional = [
                    "gain", T<float>
                    "seed", T<int>
                ]
            }

        let RandomNormalArgs = 
            Pattern.Config "RandomNormalArgs" {
                Required = []
                Optional = [
                    "mean", T<float>
                    "stddev", T<float>
                    "seed", T<int>
                ]
            }

        let RandomUniformArgs = 
            Pattern.Config "RandomUniformArgs" {
                Required = []
                Optional = [
                    "minval", T<float>
                    "maxval", T<float>
                    "seed", T<int>
                ]
            }

        let TruncatedNormalArgs = 
            Class "TruncatedNormalArgs" 
            |=> Inherits RandomNormalArgs

        let VarianceScalingArgs = 
            Pattern.Config "VarianceScalingArgs" {
                Required = []
                Optional = [
                    "scale", T<float>
                    "mode", T<string>
                    "distribution", T<string>
                    "seed", T<int>
                ]
            }

        let Zeros = 
            Class "Zeros"
            |+> Instance [
                "apply" => Shape?shape * DataType?dtype ^-> Tensor
            ]

        let L1Config = 
            Pattern.Config "L1Config" {
                Required = []
                Optional = [
                    "l1", T<float>
                ]
            }

        let L1L2Config = 
            Pattern.Config "L1L2Config" {
                Required = []
                Optional = [
                    "l1", T<float>
                    "l2", T<float>
                ]
            }

        let L2Config = 
            Pattern.Config "L2Config" {
                Required = []
                Optional = [
                    "l2", T<float>
                ]
            }

        let ImageOptions = 
            Pattern.Config "ImageOptions" {
                Required = []
                Optional = [
                    "alpha", T<float>
                ]
            }

        let WebGLContextAttributes = 
            Pattern.Config "WebGLContextAttributes" {
                Required = []
                Optional = [
                    "alpha", T<bool>
                    "antialias", T<bool>
                    "premultipliedAlpha", T<bool>
                    "preserveDrawingBuffer", T<bool>
                    "depth", T<bool>
                    "stencil", T<bool>
                    "failIfMajorPerformanceCaveat", T<bool>
                ]
            }

        let ContextOptions = 
            Pattern.Config "ContextOptions" {
                Required = []
                Optional = [
                    "contextType", T<string>
                    "contextAttributes", WebGLContextAttributes.Type
                ]
            }

        let DrawOptions = 
            Pattern.Config "DrawOptions" {
                Required = []
                Optional = [
                    "imageOptions", ImageOptions.Type
                    "contextOptions", ContextOptions.Type
                ]
            }

        let PixelData = 
            Pattern.Config "PixelData" {
                Required = [
                    "width", T<int>
                    "height", T<int>
                    "data", T<Uint8Array>
                ]
                Optional = []
            }

        let PixelDataOrImageDataOrHTMLElement = PixelData + T<ImageData> + T<HTMLImageElement> + T<HTMLCanvasElement> + T<HTMLVideoElement>

        let EarlyStoppingCallbackArgs = 
            Pattern.Config "EarlyStoppingCallbackArgs" {
                Required = []
                Optional = [
                    "monitor", T<string>
                    "minDelta", T<float>
                    "patience", T<int>
                    "verbose", T<int>
                    "mode", T<string>
                    "baseline", T<float>
                    "restoreBestWeights", T<bool>
                ]
            }

        let EarlyStopping = 
            Class "EarlyStopping"
            |=> Inherits Callback
            |+> Static [
                Constructor !?EarlyStoppingCallbackArgs?args
            ]
            |+> Instance [
            
            ]

    [<AutoOpen>]
    module TFCore = 
        let TensorOrTensorLike = Tensor + TensorLike

        let MomentsResult = 
            Pattern.Config "MomentsResult" {
                Required = [
                    "mean", Tensor.Type
                    "variance", Tensor.Type
                ]
                Optional = []
            }

        let LSTMCellFunc = Tensor?data * Tensor?c * Tensor?h ^-> !| Tensor

        let TopkResult = 
            Pattern.Config "TopkResult" {
                Required = [
                    "values", Tensor.Type
                    "indices", Tensor.Type
                ]
                Optional = []
            }

        let UniqueResult = 
            Class "UniqueResult"
            |=> Inherits TopkResult

        let MeshgridThridParameter = 
            Pattern.Config "MeshgridThridParameter" {
                Required = []
                Optional = [
                    "indexing", T<string>
                ]
            }

        let GradResult = Tensor?x * !?Tensor?dy ^-> Tensor
        let GradsResult = (!|TensorOrTensorLike)?args * !?TensorOrTensorLike?dy ^-> !|Tensor

        let ValueAndGradResultReturn = 
            Pattern.Config "ValueAndGradResultReturn" {
                Required = [
                    "value", Tensor.Type
                    "grad", Tensor.Type
                ]
                Optional = []
            }
            
        let ValueAndGradResult = Tensor?x * !?Tensor?dy ^-> ValueAndGradResultReturn
        let ValueAndGradsResult = (!|Tensor)?args * !?Tensor?dy ^-> ValueAndGradsResultReturn

        let VariableGradsResult = 
            Pattern.Config "VariableGradsResult" {
                Required = [
                    "value", Tensor.Type
                    "grads", NamedTensorMap
                ]
                Optional = []
            }

        let ProfileFunc = T<unit> ^-> TensorContainer + T<Promise<_>>[TensorContainer]
        
        let TF = 
            Class "Tf"
            |+> Static [
                // Tensors / Creation
                "tensor" => TensorValuesType?values * !?Shape?shape * !?DataType?dtype ^-> Tensor
                "scalar" => ScalarLike?value * !?DataType?dtype ^-> Tensor
                "tensor1d" => TensorLike1D?values * !?DataType?dtype ^-> Tensor
                "tensor2d" => TensorLike2D?values * !?Shape?shape * !?DataType?dtype ^-> Tensor
                "tensor3d" => TensorLike3D?values * !?Shape?shape * !?DataType?dtype ^-> Tensor
                "tensor4d" => TensorLike4D?values * !?Shape?shape * !?DataType?dtype ^-> Tensor
                "tensor5d" => TensorLike5D?values * !?Shape?shape * !?DataType?dtype ^-> Tensor
                "tensor6d" => TensorLike6D?values * !?Shape?shape * !?DataType?dtype ^-> Tensor
                "buffer" => Shape?shape * !?DataType?dtype * !?DataTypeMap?values ^-> TensorBuffer
                "clone" => TensorOrTensorLike?x  ^-> Tensor
                "complex" => TensorOrTensorLike?real * TensorOrTensorLike?imag ^-> Tensor
                "diag" => Tensor?x ^-> Tensor
                "eye" => T<float>?numRows * !?T<float>?numColumns * !?T<float[]>?batchShape * !?DataType?dtype ^-> Tensor
                "fill" => Shape?shape * (T<int> + T<string>)?value * !?DataType?dtype ^-> Tensor
                "imag" => TensorOrTensorLike?input ^-> Tensor
                "linspace" => T<float>?start * T<float>?stop * T<float>?num ^-> Tensor
                "oneHot" => TensorOrTensorLike?indices  * T<float>?depth * !?T<float>?onValue * !?T<float>?offValue * !?DataType?dtype ^-> Tensor
                "ones" => Shape?shape * !?DataType?dtype ^-> Tensor
                "onesLike" => TensorOrTensorLike?x ^-> Tensor
                "print" => Tensor?x * !?T<bool>?verbose ^-> T<unit>
                "range" => T<int>?start * T<int>?stop * !?T<int>?step * !?DataType?dtype ^-> Tensor
                "real" => TensorOrTensorLike?input ^-> Tensor
                "truncatedNormal" => Shape?shape * !?T<float>?mean * !?T<float>?stdDev * !?DataType?dtype * !?T<float>?seed ^-> Tensor
                "variable" => Tensor?initialValue  * !?T<bool>?trainable * !?T<string>?name * !?DataType?dtype ^-> Variable
                "zeros" => Shape?shape * !?DataType?dtype ^-> Tensor
                "zerosLike" => TensorOrTensorLike?x ^-> Tensor

                // Tensors / Transformations
                "batchToSpaceND" => TensorOrTensorLike * T<uint[]>?blockShape * T<uint[][]>?crops ^-> Tensor
                "broadcastArgs" => TensorOrTensorLike?s0 * TensorOrTensorLike?s1 ^-> Tensor
                "broadcastTo" => TensorOrTensorLike?x * Shape?shape ^-> Tensor
                "cast" => TensorOrTensorLike?x * DataType?dtype ^-> Tensor
                "depthToSpace" => TensorOrTensorLike?x * T<int>?blockSize * !?T<string>?dataFormat ^-> Tensor
                "ensureShape" => Tensor?x * Shape?shape ^-> Tensor
                "expandDims" => TensorOrTensorLike?x * !?T<int>?axis ^-> Tensor
                "mirrorPad" => TensorOrTensorLike?x * T<int[][]>?paddings * T<string>?mode ^-> Tensor
                "pad" => TensorOrTensorLike?x * T<int[][]>?paddings * !?T<int>?constantValue ^-> Tensor
                "reshape" => TensorOrTensorLike?x * Shape?shape ^-> Tensor
                "setdiff1dAsync" => TensorOrTensorLike?x * TensorOrTensorLike?y ^-> T<Promise<_>>[!| !| Tensor]
                "spaceToBatchND" => TensorOrTensorLike?x * T<int[]>?blockShape * T<int[][]>?paddings ^-> Tensor
                "squeeze" => TensorOrTensorLike?x * !?T<int[]>?axis ^-> Tensor

                // Tensors / Slicing and Joining
                "booleanMaskAsync" => TensorOrTensorLike?tensor * TensorOrTensorLike?mask * !?T<int>?axis ^-> T<Promise<_>>[Tensor]
                "concat" => (!|Tensor)?tensors * !?T<int>?axis ^-> Tensor
                "gather" => TensorOrTensorLike?x * TensorOrTensorLike?indices * !?T<int>?axis * !?T<int>?batchDims ^-> Tensor
                "reverse" => TensorOrTensorLike?x * !?IntOrIntArray?axis ^-> Tensor
                "slice" => TensorOrTensorLike?x * IntOrIntArray?``begin`` * !?IntOrIntArray?size ^-> Tensor
                "split" => TensorOrTensorLike?x * IntOrIntArray?numOrSizeSplits * !?T<int>?axis ^-> !|Tensor
                "stack" => (!|Tensor)?tensors * !?T<int>?axis ^-> Tensor
                "tile" => TensorOrTensorLike?x * T<int[]>?reps ^-> Tensor
                "unstack" => TensorOrTensorLike?x * !?T<int>?axis ^-> !|Tensor

                // Tensors / Matrices
                "einsum" => T<string>?equation * (!|Tensor)?tensors ^-> Tensor

                // Tensors / Random
                "multinomial" => LogitsType?logits * T<int>?numSamples * !?T<int>?seed * !?T<bool>?normalized ^-> (TensorLike1D + TensorLike2D)
                "rand" => Shape?shape * (T<unit> ^-> T<float>)?randFunction * !?DataType?dtype ^-> Tensor
                "randomGamma" => Shape?shape * T<float>?alpha * !?T<float>?beta * !?DataType?dtype * !?T<int>?seed ^-> Tensor
                "randomNormal" => Shape?shape * !?T<float>?mean * !?T<float>?stdDev * !?DataType?dtype * !?T<int>?seed ^-> Tensor
                "randomStandardNormal" => Shape?shape * !?DataType?dtype * !?T<int>?seed ^-> Tensor
                "randomUniform" => Shape?shape * !?T<float>?minval * !?T<float>?maxval * !?DataType?dtype * !?(T<int> + T<string>)?seed ^-> Tensor
                "randomUniformInt" => Shape?shape * T<int>?minval * T<int>?maxval * !?(T<int> + T<string>)?seed ^-> Tensor

                // Models / Creation
                "sequential" => !?SequentialArgs?config ^-> Sequential
                "model" => ContainerArgs?args ^-> LayersModel

                // Models / Inputs
                "input" => InputConfig?config ^-> SymbolicTensor

                // Models / Loading
                "loadGraphModel" => HandleOrString?modelUrl * LoadOptions?options * IO?tfio ^-> T<Promise<_>>[GraphModel]
                "browserDownloads" => !?T<string>?fileNamePrefix  ^-> IOHandler
                "browserFiles" => T<File[]>?files  ^-> IOHandler
                "http" => T<string>?path * !?LoadOptions?loadOptions ^-> IOHandler
                "loadGraphModelSync" => (IOHandlerSync + ModelArtifacts + (ModelJSON * T<ArrayBuffer>))?modelSource ^-> GraphModel

                // Models / Management
                "copyModel" => T<string>?sourceURL * T<string>?destURL ^-> T<Promise<_>>[ModelArtifactsInfo]
                "listModels" => T<unit> ^-> T<Promise<obj>>
                "moveModel" => T<string>?sourceURL * T<string>?destURL ^-> T<Promise<_>>[ModelArtifactsInfo]
                "removeModel" => T<string>?url ^-> T<Promise<_>>[ModelArtifactsInfo]

                // Models / Serialization
                Generic - fun t ->
                    "registerClass" => SerializableConstructor[t]?cls * !?T<string>?pkg * !?T<string>?name ^-> SerializableConstructor

                // Models / Op Registry
                "deregisterOp" => T<string>?name  ^-> T<unit>
                "getRegisteredOp" => T<string>?name  ^-> OpMapper
                "registerOp" => T<string>?name  * OpExecutor?opFunc ^-> T<unit>

                // Layers / Advanced Activation
                "layers.Elu" => !?ELULayerArgs?args ^-> ELU
                "layers.LeakyReLU" => !?LeakyReLULayerArgs?args ^-> LeakyReLU
                "layers.Prelu" => !?PReLULayerArgs?args ^-> PReLU
                "layers.ReLU" => !?ReLULayerArgs?args ^-> ReLU
                "layers.Softmax" => !?SoftmaxLayerArgs?args ^-> Softmax
                "layers.ThresholdedReLU" => !?ThresholdedReLULayerArgs?args ^-> ThresholdedReLU

                // Layers / Basic
                "layers.Activation" => ActivationLayerArgs?args ^-> Activation
                "layers.Dense" => DenseLayerArgs?args ^-> Dense
                "layers.Dropout" => DropoutLayerArgs?args ^-> Dropout
                "layers.Embedding" => EmbeddingLayerArgs?args ^-> Embedding
                "layers.Flatten" => !?FlattenLayerArgs?args ^-> Flatten
                "layers.Permute" => PermuteLayerArgs?args ^-> Permute
                "layers.RepeatVector" => RepeatVectorLayerArgs?args ^-> RepeatVector
                "layers.Reshape" => ReshapeLayerArgs?args ^-> Reshape
                "layers.SpatialDropout1d" => SpatialDropout1DLayerConfig?args ^-> SpatialDropout1D
    
                // Layers / Convolutional
                "layers.Conv1d" => ConvLayerArgs?args ^-> Conv1D
                "layers.Conv2d" => ConvLayerArgs?args ^-> Conv2D
                "layers.Conv2dTranspose" => ConvLayerArgs?args ^-> Conv2DTranspose
                "layers.Conv3d" => ConvLayerArgs?args ^-> Conv3D
                "layers.Cropping2D" => Cropping2DLayerArgs?args ^-> Cropping2D
                "layers.DepthwiseConv2d" => DepthwiseConv2DLayerArgs?args ^-> DepthwiseConv2D
                "layers.SeparableConv2d" => SeparableConvLayerArgs?args ^-> SeparableConv2D
                "layers.UpSampling2d" => UpSampling2DLayerArgs?args ^-> UpSampling2D

                // Layers / Merge
                "layers.Add" => !?LayerArgs?args ^-> Add
                "layers.Average" => !?LayerArgs?args ^-> Average
                "layers.Concatenate" => !?ConcatenateLayerArgs?args ^-> Concatenate
                "layers.Maximum" => !?LayerArgs?args ^-> Maximum
                "layers.Minimum" => !?LayerArgs?args ^-> Minimum
                "layers.Multiply" => !?LayerArgs?args ^-> Multiply

                // Layers / Normalization
                "layers.BatchNormalization" => !?BatchNormalizationLayerArgs?args ^-> BatchNormalization
                "layers.LayerNormalization" => !?LayerNormalizationLayerArgs?args ^-> LayerNormalization

                // Layers / Pooling
                "layers.AveragePooling1d" => Pooling1DLayerArgs?args ^-> AveragePooling1D 
                "layers.AveragePooling2d" =>  Pooling2DLayerArgs?args ^-> AveragePooling2D 
                "layers.AveragePooling3d" =>  Pooling3DLayerArgs?args ^-> AveragePooling3D 
                "layers.GlobalAveragePooling1d" => !?LayerArgs?args ^-> GlobalAveragePooling1D 
                "layers.GlobalAveragePooling2d" => GlobalPooling2DLayerArgs?args ^-> GlobalAveragePooling2D 
                "layers.GlobalMaxPooling1d" =>  !?LayerArgs?args ^-> GlobalMaxPooling1D 
                "layers.GlobalMaxPooling2d" =>  GlobalPooling2DLayerArgs?args ^-> GlobalMaxPooling2D 
                "layers.MaxPooling1d" =>  Pooling1DLayerArgs?args ^-> MaxPooling1D 
                "layers.MaxPooling2d" =>  Pooling2DLayerArgs?args ^-> MaxPooling2D 
                "layers.MaxPooling3d" =>  Pooling3DLayerArgs?args ^-> MaxPooling3D 
                
                // Layers / Recurrent
                "layers.ConvLstm2d" => ConvLSTM2DArgs?args ^-> ConvLSTM2D 
                "layers.ConvLstm2dCell" => ConvLSTM2DCellArgs?args ^-> ConvLSTM2DCell
                "layers.Gru" => GRULayerArgs?args ^-> GRU
                "layers.GruCell" => GRUCellLayerArgs?args ^-> GRUCell
                "layers.Lstm" => LSTMLayerArgs?args ^-> LSTM
                "layers.LstmCell" => LSTMCellLayerArgs?args ^-> LSTMCell
                "layers.Rnn" => RNNLayerArgs?args ^-> RNN
                "layers.SimpleRNN" => SimpleRNNLayerArgs?args ^-> SimpleRNN
                "layers.SimpleRNNCell" => SimpleRNNCellLayerArgs?args ^-> SimpleRNNCell
                "layers.StackedRNNCells" => StackedRNNCellsArgs?args ^-> StackedRNNCells

                // Layers / Wrapper
                "layers.Bidirectional" => BidirectionalLayerArgs?args ^-> Bidirectional
                "layers.TimeDistributed" => WrapperLayerArgs?args ^-> TimeDistributed

                // Layers / Inputs
                "layers.InputLayer" => InputLayerArgs?args ^-> InputLayer

                // Layers / Padding
                "layers.ZeroPadding2d" => !?ZeroPadding2DLayerArgs?args ^-> ZeroPadding2D

                // Layers / Noise
                "layers.AlphaDropout" => AlphaDropoutArgs?args ^-> AlphaDropout
                "layers.GaussianDropout" => GaussianDropoutArgs?args ^-> GaussianDropout
                "layers.GaussianNoise" => GaussianNoiseArgs?args ^-> GaussianNoise

                // Layers / Mask
                "layers.Masking" => MaskingArgs?args ^-> Masking

                // Layers / Rescaling
                "layers.Rescaling" => !?RescalingArgs?args ^-> Rescaling

                // Layers / CenterCrop
                "layers.Rescaling" => !?CenterCropArgs?args ^-> CenterCrop

                // Layers / Resizing
                "layers.Resizing" => !?ResizingArgs?args ^-> Resizing

                // Layers / CategoryEncoding
                "layers.CategoryEncoding" => CategoryEncodingArgs?args ^-> CategoryEncoding

                //Layers / RandomWidth
                "layers.RandomWidth" => RandomWidthArgs?args ^-> RandomWidth

                // Operations / Arithmetic
                "add" => TensorOrTensorLike?a * TensorOrTensorLike?b ^-> Tensor
                "sub" => TensorOrTensorLike?a * TensorOrTensorLike?b ^-> Tensor
                "mul" => TensorOrTensorLike?a * TensorOrTensorLike?b ^-> Tensor
                "div" => TensorOrTensorLike?a * TensorOrTensorLike?b ^-> Tensor
                "addN" => (!|TensorOrTensorLike)?tensor ^-> Tensor
                "divNoNan" => TensorOrTensorLike?a * TensorOrTensorLike?b ^-> Tensor
                "floorDiv" => TensorOrTensorLike?a * TensorOrTensorLike?b ^-> Tensor
                "maximum" => TensorOrTensorLike?a * TensorOrTensorLike?b ^-> Tensor
                "minimum" => TensorOrTensorLike?a * TensorOrTensorLike?b ^-> Tensor
                "mod" => TensorOrTensorLike?a * TensorOrTensorLike?b ^-> Tensor
                "pow" => TensorOrTensorLike?a * TensorOrTensorLike?b ^-> Tensor
                "squaredDifference" => TensorOrTensorLike?a * TensorOrTensorLike?b ^-> Tensor

                // Operations / Basic math
                "abs" => TensorOrTensorLike?x ^-> Tensor 
                "acos" => TensorOrTensorLike?x ^-> Tensor 
                "acosh" => TensorOrTensorLike?x ^-> Tensor 
                "asin" => TensorOrTensorLike?x ^-> Tensor 
                "asinh" => TensorOrTensorLike?x ^-> Tensor 
                "atan" => TensorOrTensorLike?x ^-> Tensor 
                "atan2" => TensorOrTensorLike?a * TensorOrTensorLike?b ^-> Tensor 
                "atanh" => TensorOrTensorLike?x ^-> Tensor 
                "ceil" => TensorOrTensorLike?x ^-> Tensor 
                "clipByValue" => TensorOrTensorLike?x * T<float>?clipValueMin * T<float>?clipValueMax ^-> Tensor 
                "cos" => TensorOrTensorLike?x ^-> Tensor 
                "cosh" => TensorOrTensorLike?x ^-> Tensor 
                "elu" => TensorOrTensorLike?x ^-> Tensor 
                "erf" => TensorOrTensorLike?x ^-> Tensor 
                "exp" => TensorOrTensorLike?x ^-> Tensor 
                "expm1" => TensorOrTensorLike?x ^-> Tensor 
                "floor" => TensorOrTensorLike?x ^-> Tensor 
                "isFinite" => TensorOrTensorLike?x ^-> Tensor 
                "isInf" => TensorOrTensorLike?x ^-> Tensor 
                "isNaN" => TensorOrTensorLike?x ^-> Tensor 
                "leakyRelu" => TensorOrTensorLike?x * !? T<float>?alpha ^-> Tensor 
                "log" => TensorOrTensorLike?x ^-> Tensor 
                "log1p" => TensorOrTensorLike?x ^-> Tensor 
                "logSigmoid" => TensorOrTensorLike?x ^-> Tensor 
                "neg" => TensorOrTensorLike?x ^-> Tensor 
                "prelu" => TensorOrTensorLike?x * TensorOrTensorLike?alpha  ^-> Tensor 
                "reciprocal" => TensorOrTensorLike?x ^-> Tensor 
                "relu" => TensorOrTensorLike?x ^-> Tensor 
                "relu6" => TensorOrTensorLike?x ^-> Tensor 
                "round" => TensorOrTensorLike?x ^-> Tensor 
                "rsqrt" => TensorOrTensorLike?x ^-> Tensor 
                "selu" => TensorOrTensorLike?x ^-> Tensor 
                "sigmoid" => TensorOrTensorLike?x ^-> Tensor 
                "sign" => TensorOrTensorLike?x ^-> Tensor 
                "sin" => TensorOrTensorLike?x ^-> Tensor 
                "sinh" => TensorOrTensorLike?x ^-> Tensor 
                "softplus" => TensorOrTensorLike?x ^-> Tensor 
                "sqrt" => TensorOrTensorLike?x ^-> Tensor 
                "square" => TensorOrTensorLike?x ^-> Tensor 
                "step" => TensorOrTensorLike?x * !? T<float>?alpha ^-> Tensor 
                "tan" => TensorOrTensorLike?x ^-> Tensor 
                "tanh" => TensorOrTensorLike?x ^-> Tensor 

                // Operations / Matrices
                "dot" => TensorOrTensorLike?t1 * TensorOrTensorLike?t2 ^-> Tensor 
                "euclideanNorm" => TensorOrTensorLike?x * !? IntOrIntArray?axis * !? T<bool>?keepDims  ^-> Tensor 
                "matMul" => TensorOrTensorLike?a * TensorOrTensorLike?b * !? T<bool>?transposeA * !? T<bool>?transposeB ^-> Tensor 
                "norm" => TensorOrTensorLike?x * !? (T<int> + T<string>)?ord * !? IntOrIntArray?axis * !? T<bool>?keepDims ^-> Tensor 
                "outerProduct" => TensorOrTensorLike?v1 * TensorOrTensorLike?v2 ^-> Tensor 
                "transpose" => TensorOrTensorLike?x * !? T<int[]>?perm * !? T<bool>?conjugate  ^-> Tensor 

                // Operations / Convolution
                "avgPool" => TensorOrTensorLike?x * IntOrIntArray?filterSize * IntOrIntArray?strides * StringOrIntOrIntArray?pad * !? T<string>?dimRoundingMode  ^-> Tensor
                "avgPool3d" => TensorOrTensorLike?x * IntOrIntArray?filterSize * IntOrIntArray?strides * StringOrIntOrIntArray?pad * !? T<string>?dimRoundingMode * !? T<string>?dataFormat ^-> Tensor
                "conv1d" => TensorOrTensorLike?x * TensorOrTensorLike?fiter * T<int>?stride * StringOrIntOrIntArray?pad * !? T<string>?dimRoundingMode * T<int>?dilation * !? T<string>?dataFormat ^-> Tensor
                "conv2d" => TensorOrTensorLike?x * TensorOrTensorLike?fiter * IntOrIntArray?strides * StringOrIntOrIntArray?pad * !? T<string>?dimRoundingMode * IntOrIntArray?dilations * !? T<string>?dataFormat ^-> Tensor
                "conv2dTranspose" => TensorOrTensorLike?x * TensorOrTensorLike?fiter * T<int[]>?outputShape * IntOrIntArray?strides * StringOrIntOrIntArray?pad * !? T<string>?dimRoundingMode ^-> Tensor
                "conv3d" => TensorOrTensorLike?x * TensorOrTensorLike?filter  * IntOrIntArray?strides  * T<string>?pad * !? T<string>?dataFormat * !? IntOrIntArray?dilations ^-> Tensor
                "conv3dTranspose" => TensorOrTensorLike?x * TensorOrTensorLike?filter * T<int[]>?outputShape * IntOrIntArray?strides * T<string>?pad ^-> Tensor
                "depthwiseConv2d" => TensorOrTensorLike?x * TensorOrTensorLike?filter * IntOrIntArray?strides * StringOrIntOrIntArray?pad * !? T<string>?dataFormat * !? IntOrIntArray?dilations * !? T<string>?dimRoundingMode ^-> Tensor
                "dilation2d" => TensorOrTensorLike?x * TensorOrTensorLike?filter * IntOrIntArray?strides * T<string>?pad * !? IntOrIntArray?dilations * !? T<string>?dataFormat ^-> Tensor
                "maxPool3d" => TensorOrTensorLike?x * IntOrIntArray?filterSize * IntOrIntArray?strides * StringOrInt?pad * !? T<string>?dimRoundingMode * !? T<string>?dataFormat ^-> Tensor
                "maxPoolWithArgmax" => TensorOrTensorLike?x * IntOrIntArray?filterSize * IntOrIntArray?strides * StringOrInt?pad * !? T<bool>?includeBatchInIndex ^-> T<obj[]>
                "pool" => TensorOrTensorLike?input * IntOrIntArray?windowShape * T<string>?poolingType  * StringOrIntOrIntArray?pad * !? IntOrIntArray?dilations * !? IntOrIntArray?strides * !? T<string>?dimRoundingMode ^-> Tensor
                "separableConv2d" => TensorOrTensorLike?x * TensorOrTensorLike?depthwiseFilter  * TensorOrTensorLike?pointwiseFilter * IntOrIntArray?strides * T<string>?pad * !? IntOrIntArray?dilations * !? T<string>?datFormat ^-> Tensor  

                // Operations / Reduction
                "all" => TensorOrTensorLike?x * !? IntOrIntArray?axis * !? T<bool>?keepDims ^-> Tensor
                "any" => TensorOrTensorLike?x * !? IntOrIntArray?axis * !? T<bool>?keepDims ^-> Tensor
                "argMax" => TensorOrTensorLike?x * !? T<int>?axis ^-> Tensor
                "argMin" => TensorOrTensorLike?x * !? T<int>?axis ^-> Tensor
                "bincount" => TensorOrTensorLike?x * TensorOrTensorLike?weights * T<int>?size ^-> Tensor
                "denseBincount" => TensorOrTensorLike?x * TensorOrTensorLike?weights * T<int>?size * !? T<bool>?binaryOutput ^-> Tensor
                "logSumExp" => TensorOrTensorLike?x * !? IntOrIntArray?axis * !? T<bool>?keepDims ^-> Tensor
                "max" => TensorOrTensorLike?x * !? IntOrIntArray?axis * !? T<bool>?keepDims ^-> Tensor
                "mean" => TensorOrTensorLike?x * !? IntOrIntArray?axis * !? T<bool>?keepDims ^-> Tensor
                "min" => TensorOrTensorLike?x * !? IntOrIntArray?axis * !? T<bool>?keepDims ^-> Tensor
                "prod" => TensorOrTensorLike?x * !? IntOrIntArray?axis * !? T<bool>?keepDims ^-> Tensor
                "sum" => TensorOrTensorLike?x * !? IntOrIntArray?axis * !? T<bool>?keepDims ^-> Tensor    

                // Operations / Normalization
                "batchNorm" => TensorOrTensorLike?x * TensorOrTensorLike?mean * TensorOrTensorLike?variance * !? TensorOrTensorLike?offset * !? TensorOrTensorLike?scale * !? T<float>?varianceEpsilon ^-> Tensor
                "localResponseNormalization" => TensorOrTensorLike?x * !? T<int>?depthRadius * !? T<float>?bias * !? T<float>?alpha * !? T<float>?beta ^-> Tensor
                "logSoftmax" => TensorOrTensorLike?logits * !? T<int>?axis ^-> Tensor
                "moments" => TensorOrTensorLike?x * !? IntOrIntArray?axis * !? T<bool>?keepDims ^-> MomentsResult
                "softmax" => TensorOrTensorLike?logits * !? T<int>?dim ^-> Tensor
                "sparseToDense" => TensorOrTensorLike?sparseIndices * TensorOrTensorLike?sparseValues * T<int[]>?outputShape * !? TensorOrTensorLike?defaultValue ^-> Tensor

                // Operations / Images
                "Image.CropAndResize" => TensorOrTensorLike?image * TensorOrTensorLike?boxes * TensorOrTensorLike?boxInd * IntOrIntArray?cropSize * !? T<string>?method * !? T<float>?extrapolationValue ^-> Tensor
                "Image.FlipLeftRight" => TensorOrTensorLike?image ^-> Tensor
                "Image.GrayscaleToRGB" => TensorOrTensorLike?image ^-> Tensor
                "Image.NonMaxSuppression" => TensorOrTensorLike?boxes * TensorOrTensorLike?scores * T<int>?maxOutputSize * !? T<float>?iouThreshold * !? T<float>?scoreThreshold ^-> Tensor
                "Image.NonMaxSuppressionAsync" => TensorOrTensorLike?boxes * TensorOrTensorLike?scores * T<int>?maxOutputSize * !? T<float>?iouThreshold * !? T<float>?scoreThreshold ^-> T<Promise<_>>[Tensor]
                "Image.NonMaxSuppressionPadded" => TensorOrTensorLike?boxes * TensorOrTensorLike?scores * T<int>?maxOutputSize * !? T<float>?iouThreshold * !? T<float>?scoreThreshold * !? T<bool>?padToMaxOutputSize ^-> T<obj[]>
                "Image.NonMaxSuppressionPaddedAsync" => TensorOrTensorLike?boxes * TensorOrTensorLike?scores * T<int>?maxOutputSize * !? T<float>?iouThreshold * !? T<float>?scoreThreshold * !? T<bool>?padToMaxOutputSize ^-> T<Promise<obj[]>>
                "Image.NonMaxSuppressionWithScore" => TensorOrTensorLike?boxes * TensorOrTensorLike?scores * T<int>?maxOutputSize * !? T<float>?iouThreshold * !? T<float>?scoreThreshold * !? T<float>?softNmsSigma ^-> T<obj[]>
                "Image.NonMaxSuppressionWithScoreAsync" => TensorOrTensorLike?boxes * TensorOrTensorLike?scores * T<int>?maxOutputSize * !? T<float>?iouThreshold * !? T<float>?scoreThreshold * !? T<float>?softNmsSigma ^-> T<Promise<obj[]>>
                "Image.ResizeBilinear" => TensorOrTensorLike?images * T<int[]>?size * !? T<bool>?alignCorners * !? T<bool>?halfPixelCenters ^-> Tensor
                "Image.ResizeNearestNeighbor" => TensorOrTensorLike?images * T<int[]>?size * !? T<bool>?alignCorners * !? T<bool>?halfPixelCenters ^-> Tensor
                "Image.RgbToGrayscale" => TensorOrTensorLike?image ^-> Tensor
                "Image.RotateWithOffset" => TensorOrTensorLike?image * T<float>?radians * !? FloatOrFloatArray?fillValue * !? FloatOrFloatArray?center ^-> Tensor
                "Image.Transform" => TensorOrTensorLike?image * TensorOrTensorLike?transforms * !? T<string>?interpolation * !? T<string>?fillMode * !? T<float>?fillValue * !? T<int[]>?outputShape ^-> Tensor
         
                // Operations / RNN
                "basicLSTMCell" => TensorOrTensorLike?forgetBias * TensorOrTensorLike?lstmKernel * TensorOrTensorLike?lstmBias * TensorOrTensorLike?data * TensorOrTensorLike?c * TensorOrTensorLike?h ^-> !|Tensor
                "multiRNNCell" => (!|LSTMCellFunc)?lstmCells * TensorOrTensorLike?data * (!|TensorOrTensorLike)?c * (!|TensorOrTensorLike)?h ^-> !| (!|Tensor)
                
                // Operations / Logical
                "bitwiseAnd" => Tensor?x * Tensor?y ^-> Tensor
                "equal" => TensorOrTensorLike?a * TensorOrTensorLike?b ^-> Tensor
                "greater" => TensorOrTensorLike?a * TensorOrTensorLike?b ^-> Tensor
                "greaterEqual" => TensorOrTensorLike?a * TensorOrTensorLike?b ^-> Tensor
                "less" => TensorOrTensorLike?a * TensorOrTensorLike?b ^-> Tensor
                "lessEqual" => TensorOrTensorLike?a * TensorOrTensorLike?b ^-> Tensor
                "logicalAnd" => TensorOrTensorLike?a * TensorOrTensorLike?b ^-> Tensor
                "logicalNot" => Tensor?x ^-> Tensor
                "logicalOr" => TensorOrTensorLike?a * TensorOrTensorLike?b ^-> Tensor
                "logicalXor" => TensorOrTensorLike?a * TensorOrTensorLike?b ^-> Tensor
                "notEqual" => TensorOrTensorLike?a * TensorOrTensorLike?b ^-> Tensor
                "where" => TensorOrTensorLike?condition * TensorOrTensorLike?a * TensorOrTensorLike?b ^-> Tensor
                "whereAsync" => Tensor?condition ^-> T<Promise<_>>[Tensor]

                // Operations / Scan
                "cumprod" => TensorOrTensorLike?x * !?T<int>?axis * !?T<bool>?exclusive * !?T<bool>?reverse ^-> Tensor
                "cumsum" => TensorOrTensorLike?x * !?T<int>?axis * !?T<bool>?exclusive * !?T<bool>?reverse ^-> Tensor
   
                // Operations / Evaluation
                "confusionMatrix" => TensorOrTensorLike?labels * TensorOrTensorLike?predictions * T<int>?numClasses ^-> Tensor
                "inTopKAsync" => TensorOrTensorLike?predictions * TensorOrTensorLike?targets * !?T<int>?k ^-> T<Promise<_>>[Tensor]
                "lowerBound" => TensorOrTensorLike?sortedSequence * TensorOrTensorLike?values ^-> Tensor
                "searchSorted" => TensorOrTensorLike?sortedSequence * TensorOrTensorLike?values * !?T<string>?side ^-> Tensor
                "topk" => TensorOrTensorLike?x * !?T<int>?k * !?T<bool>?sorted ^-> TopkResult
                "unique" => TensorOrTensorLike?x * !?T<int>?axis ^-> UniqueResult
                "upperBound" => TensorOrTensorLike?sortedSequence * TensorOrTensorLike?values ^-> Tensor

                // Operations / Slicing and Joining
                "gatherND" => TensorOrTensorLike?x * TensorOrTensorLike?indices ^-> Tensor
                "meshgrid" => !?TensorOrTensorLike?x * !?TensorOrTensorLike?y * !?MeshgridThridParameter ^-> !|Tensor
                "scatterND" => TensorOrTensorLike?indices * TensorOrTensorLike?updates * Shape?shape ^-> Tensor
                "stridedSlice" => TensorOrTensorLike?x * T<int[]>?``begin`` * T<int[]>?``end`` * !?T<int[]>?strides * !?T<int>?beginMask * !?T<int>?endMask * !?T<int>?ellipsisMask * !?T<int>?newAxisMask * !?T<int>?shrinkAxisMask ^-> Tensor
                "tensorScatterUpdate" => TensorOrTensorLike?tensor * TensorOrTensorLike?indices * TensorOrTensorLike?updates ^-> Tensor
    
                // Operations / Ragged
                "raggedTensorToTensor" => TensorOrTensorLike?shape * TensorOrTensorLike?values * TensorOrTensorLike?defaultValue * (!|Tensor)?rowPartitionTensors * T<string[]>?rowPartitionTypes ^-> Tensor
                
                // Operations / Spectral
                "spectral.Fft" => Tensor?input ^-> Tensor
                "spectral.Ifft" => Tensor?input ^-> Tensor
                "spectral.Irfft" => Tensor?input ^-> Tensor
                "spectral.Rfft" => Tensor?input * !?T<int>?fftLength ^-> Tensor

                // Operations / Segment
                "unsortedSegmentSum" => TensorOrTensorLike?x * TensorOrTensorLike?segmentIds * T<int>?numSegments ^-> Tensor

                // Operations / Moving Average
                "movingAverage" => TensorOrTensorLike?v * TensorOrTensorLike?x * IntOrTensor?decay * !?IntOrTensor?step * !?T<bool>?zeroDebias ^-> Tensor

                // Operations / Dropout
                "dropout" => TensorOrTensorLike?x * T<int>?rate * !?Shape?noiseShape * !?(T<int> + T<string>)?seed ^-> Tensor

                //Operations / Signal
                "signal.Frame" => Tensor?signal * T<int>?frameLength * T<int>?frameStep * !?T<bool>?padEnd * !?T<int>?padValue ^-> Tensor
                "signal.HammingWindow" => T<int>?windowLength ^-> Tensor
                "signal.HannWindow" => T<int>?windowLength ^-> Tensor
                "signal.Stft" => Tensor?signal * T<int>?frameLength * T<int>?frameStep * !?T<int>?fftLength * !?WindowFn?windowFn ^-> Tensor

                // Operations / Linear Algebra
                "linalg.BandPart" => TensorOrTensorLike?a * IntOrTensor?numLower * IntOrTensor?numUpper ^-> Tensor
                "linalg.GramSchmidt" => TensorOrTensorArray?xs ^-> Tensor + !|Tensor
                "linalg.Qr" => Tensor?x * !?T<bool>?fullMatrices ^-> !|Tensor

                // Operations / Sparse
                "sparse.SparseFillEmptyRows" => TensorOrTensorLike?indices * TensorOrTensorLike?values * TensorOrTensorLike?denseShape * TensorOrTensorLike?defaultValue ^-> T<obj[]>
                "sparse.SparseReshape" => TensorOrTensorLike?inputIndices * TensorOrTensorLike?inputShape * TensorOrTensorLike?newShape ^-> T<obj[]>
                "sparse.SparseSegmentMean" => TensorOrTensorLike?data * TensorOrTensorLike?indices * TensorOrTensorLike?segmentIds ^-> Tensor
                "sparse.SparseSegmentSum" => TensorOrTensorLike?data * TensorOrTensorLike?indices * TensorOrTensorLike?segmentIds ^-> Tensor

                // Operations / String
                "staticRegexReplace" => TensorOrTensorLike?input * T<string>?pattern * T<string>?rewrite * !?T<bool>?replaceGlobal ^-> Tensor
                "stringNGrams" => TensorOrTensorLike?data * TensorOrTensorLike?dataSplits * T<string>?separator * T<int[]>?nGramWidths * T<string>?leftPad * T<string>?rightPad * T<int>?padWidth * T<bool>?preserveShortSequences ^-> T<obj[]>
                "stringSplit" => TensorOrTensorLike?input * TensorOrTensorLike?delimiter * !?T<bool>?skipEmpty ^-> T<obj[]>
                "stringToHashBucketFast" => TensorOrTensorLike?input * T<int>?numBuckets ^-> Tensor
    
                // Training / Gradients
                "grad" => GradFunc?f ^-> GradResult          
                "grads" => GradsFunc?f ^-> GradsResult                
                "customGrad" => CustomGradientFunc?f ^-> ((!|Tensor)?args ^-> Tensor)                
                "valueAndGrad" => (Tensor?x ^-> Tensor)?f ^-> ValueAndGradResult             
                "valueAndGrads" => ((!|Tensor)?args ^-> Tensor)?f ^-> ValueAndGradsResult            
                "variableGrads" => (T<unit> ^-> Tensor)?f * !?(!|Variable)?varList ^-> VariableGradsResult      
                
                // Training / Optimizers
                "train.Sgd" => T<float>?learningRate ^-> SGDOptimizer              
                "train.Momentum" => T<float>?learningRate * T<float>?momentum * !?T<bool>?useNesterov ^-> MomentumOptimizer    
                "train.Adagrad" => T<float>?learningRate * T<float>?initialAccumulatorValue ^-> AdagradOptimizer
                "train.Adadelta" => !?T<float>?learningRate * !?T<float>?rho * !?T<float>?epsilon ^-> AdadeltaOptimizer
                "train.Adam" => !?T<float>?learningRate * !?T<float>?beta1 * !?T<float>?beta2 * !?T<float>?epsilon ^-> AdamOptimizer
                "train.Adamax" => !?T<float>?learningRate * !?T<float>?beta1 * !?T<float> * !?T<float>?epsilon * !?T<float>?decay ^-> AdamaxOptimizer
                "train.Rmsprop" => T<float>?learningRate * !?T<float>?decay * !?T<float>?momentum * !?T<float>?epsilon * !?T<bool>?centered ^-> RMSPropOptimizer

                // Training / Losses
                "losses.AbsoluteDifference" => TensorOrTensorLike?labels * TensorOrTensorLike?predictions * !?TensorOrTensorLike?weights * !?Reduction?reduction ^-> Tensor
                "losses.ComputeWeightedLoss" => TensorOrTensorLike?losses * !?TensorOrTensorLike?weights * !?Reduction?reduction ^-> Tensor
                "losses.CosineDistance" => TensorOrTensorLike?labels * TensorOrTensorLike?predictions * T<int>?axis * !?TensorOrTensorLike?weights * !?Reduction?reduction ^-> Tensor
                "losses.CingeLoss" => TensorOrTensorLike?labels * TensorOrTensorLike?predictions * !?TensorOrTensorLike?weights * !?Reduction?reduction ^-> Tensor
                "losses.HuberLoss" => TensorOrTensorLike?labels * TensorOrTensorLike?predictions * !?TensorOrTensorLike?weights * !?T<int>?delta * !?Reduction?reduction ^-> Tensor
                "losses.LogLoss" => TensorOrTensorLike?labels * TensorOrTensorLike?predictions * !?TensorOrTensorLike?weights * !?T<int>?epsilon * !?Reduction?reduction ^-> Tensor
                "losses.MeanSquaredError" => TensorOrTensorLike?labels * TensorOrTensorLike?predictions * !?TensorOrTensorLike?weights * !?Reduction?reduction ^-> Tensor
                "losses.SigmoidCrossEntropy" => TensorOrTensorLike?multiClassLabels * TensorOrTensorLike?logits * !?TensorOrTensorLike?weights * !?T<int>?labelSmoothing * !?Reduction?reduction ^-> Tensor
                "losses.SoftmaxCrossEntropy" => TensorOrTensorLike?onehotLabels * TensorOrTensorLike?logits * !?TensorOrTensorLike?weights * !?T<int>?labelSmoothing * !?Reduction?reduction ^-> Tensor
                
                // Performance / Memory
                "tidy" => (T<string> + ScopeFn)?nameOrFn * !?ScopeFn?fn ^-> TensorContainer
                "dispose" => TensorContainer?container ^-> T<unit>
                "keep" => Tensor?result ^-> Tensor
                "memory" => T<unit> ^-> MemoryInfo

                // Performance / Timing
                "time" => (T<unit> ^-> T<unit>)?f ^-> T<Promise<_>>[TimingInfo]
                "nextFrame" => T<unit> ^-> T<Promise<unit>>

                // Performance / Profile
                "profile" => ProfileFunc?f ^-> T<Promise<_>>[ProfileInfo]

                // Environment
                "disposeVariables" => T<unit> ^-> T<unit>
                "enableDebugMode" => T<unit> ^-> T<unit>
                "enableProdMode" => T<unit> ^-> T<unit>
                "engine" => T<unit> ^-> Engine
                "env" => T<unit> ^-> Environment

                // Constraints
                "constraints.MaxNorm" => MaxNormArgs?args ^-> Constraint
                "constraints.MinMaxNorm" => MinMaxNormArgs?config ^-> Constraint
                "constraints.NonNeg" => T<unit> ^-> Constraint
                "constraints.UnitNorm" => UnitNormArgs?args ^-> Constraint

                // Initializers
                "initializers.Constant" => ConstantArgs?args ^-> Initializer               
                "initializers.GlorotNormal" => GlorotNormalArgs?args ^-> Initializer
                "initializers.GlorotUniform" => GlorotUniformArgs?args ^-> Initializer
                "initializers.HeNormal" => HeNormalArgs?args ^-> Initializer
                "initializers.HeUniform" => HeUniformArgs?args ^-> Initializer
                "initializers.Identity" => IdentityArgs?args ^-> Initializer
                "initializers.LeCunNormal" => LeCunNormalArgs?args ^-> Initializer
                "initializers.LeCunUniform" => LeCunUniformArgs?args ^-> Initializer
                "initializers.Ones" => T<unit> ^-> Initializer
                "initializers.Orthogonal" => OrthogonalArgs?args ^-> Initializer
                "initializers.RandomNormal" => RandomNormalArgs?args ^-> Initializer
                "initializers.RandomUniform" => RandomUniformArgs?args ^-> Initializer
                "initializers.TruncatedNormal" => TruncatedNormalArgs?args ^-> Initializer
                "initializers.VarianceScaling" => VarianceScalingArgs?config ^-> Initializer
                "initializers.Zeros" => T<unit> ^-> Zeros

                // Regularizers               
                "regularizers.L1" => !?L1Config?config ^-> Regularizer
                "regularizers.L1l2" => !?L1L2Config?config ^-> Regularizer
                "regularizers.L2" => !?L2Config?config ^-> Regularizer

                // Data / Creation
                "data.Array" => (!|TensorContainer)?items ^-> Dataset[TensorContainer]
                "data.Csv" => T<string>?source * CSVConfig?csvConfig ^-> CSVDataset
                "data.Generator" => (T<unit> ^-> Iterator + T<Promise<_>>[Iterator])?generator ^-> Dataset[TensorContainer]
                "data.Microphone" => !?MicrophoneConfig?microphoneConfig ^-> T<Promise<_>>[MicrophoneIterator]
                "data.Webcam" => !?T<HTMLVideoElement>?webcamVideoElement * !?WebcamConfig?webcamConfig ^-> T<Promise<_>>[WebcamIterator]

                // Data / Operations
                "data.Zip" => T<obj>?datasets ^-> Dataset[TensorContainer]

                // Util
                "util.Assert" => T<bool>?expr * (T<unit> ^-> T<string>)?msg ^-> T<unit>
                "util.CreateShuffledIndices" => T<int>?n ^-> T<Uint32Array>
                "util.DecodeString" => T<Uint8Array>?bytes * !?T<string>?encoding ^-> T<string>
                "util.EncodeString" => T<string>?s * !?T<string>?encoding ^-> T<Uint8Array>
                "util.Fetch" => T<string>?path * !?T<Request>?requestInits ^-> T<Promise<Response>>
                "util.Flatten" => (FlattenType + T<obj[]>)?arr * !?(!|FlattenType)?result * !?T<bool>?skipTypedArray ^-> !|FlattenType
                "util.Now" => T<unit> ^-> T<int>
                "util.Shuffle" => NumberTypedArrayOrObjArray?array ^-> T<unit>
                "util.ShuffleCombo" => NumberTypedArrayOrObjArray?array * NumberTypedArrayOrObjArray?array2 ^-> T<unit>
                "util.SizeFromShape" => Shape?shape ^-> T<int>

                // Backend
                "backend" => T<unit> ^-> KernelBackend
                "getBackend" => T<unit> ^-> T<string>
                "ready" => T<unit> ^-> T<Promise<unit>>
                "registerBackend" => T<string>?name * (T<unit> ^-> KernelBackend + T<Promise<_>>[KernelBackend])?factory * !?T<int>?priority ^-> T<bool>
                "removeBackend" => T<string>?name ^-> T<unit>
                "setBackend" => T<string>?backendName ^-> T<Promise<bool>>

                // Browser
                "browser.Draw" => TensorOrTensorLike?image * T<HTMLCanvasElement>?canvas * !?DrawOptions?ptions ^-> T<unit>
                "browser.FromPixels" => PixelDataOrImageDataOrHTMLElement?pixels * !?T<int>?numChannels ^-> Tensor
                "browser.FromPixelsAsync" => PixelDataOrImageDataOrHTMLElement?pixels * !?T<int>?numChannels ^-> T<Promise<_>>[Tensor]
                "browser.ToPixels" => TensorOrTensorLike?img * !?T<HTMLCanvasElement>?canvas ^-> T<Promise<Uint8ClampedArray>>

                // Metrics
                "metrics.BinaryAccuracy" => Tensor?yTrue * Tensor?yPred ^-> Tensor
                "metrics.BinaryCrossentropy" => Tensor?yTrue * Tensor?yPred ^-> Tensor
                "metrics.CategoricalAccuracy" => Tensor?yTrue * Tensor?yPred ^-> Tensor
                "metrics.CategoricalCrossentropy" => Tensor?yTrue * Tensor?yPred ^-> Tensor
                "metrics.CosineProximity" => Tensor?yTrue * Tensor?yPred ^-> Tensor
                "metrics.MeanAbsoluteError" => Tensor?yTrue * Tensor?yPred ^-> Tensor
                "metrics.MeanAbsolutePercentageError" => Tensor?yTrue * Tensor?yPred ^-> Tensor
                "metrics.MeanSquaredError" => Tensor?yTrue * Tensor?yPred ^-> Tensor
                "metrics.Precision" => Tensor?yTrue * Tensor?yPred ^-> Tensor
                "metrics.R2Score" => Tensor?yTrue * Tensor?yPred ^-> Tensor
                "metrics.Recall" => Tensor?yTrue * Tensor?yPred ^-> Tensor
                "metrics.SparseCategoricalAccuracy" => Tensor?yTrue * Tensor?yPred ^-> Tensor

                // Callbacks
                "callbacks.EarlyStopping" => !?EarlyStoppingCallbackArgs?args ^-> EarlyStopping
            ]

    let Assembly =
        Assembly [
            Namespace "WebSharper.TensorFlowJs" [
                TF
                Tensor
                TensorBuffer
                Variable
                GraphModel
                LayersModel
                Sequential
                SymbolicTensor
                Dataset
                CSVDataset
                Layer
                Environment
                Optimizer
                Regularizer
                Initializer
                Constraint


                // Enum
                ParamType; Category; WeightGroup; Rank; WebGLChannels; ActivationIdentifier; Reduction

                // Interface
                CustomCallbackArgs; ModelPredictArgs; ModelEvaluateDatasetArgs; IOHandlerSync
                IOHandler; ModelPredictConfig; SaveResult; ModelArtifacts; ModelArtifactsInfo
                SaveConfig; WeightsManifestEntry; Quantization; TrainingConfig; ModelTensorInfo
                WebGPUData; WebGLData; DataToGPUWebGLOption; DataTypeMap; TensorInfo;PrintFnType
                SingleValueMap; Memory; TimingInfo; KernelInfo; BackendTimingInfo; MemoryInfo
                GlorotNormalArgs; ConstantArgs; UnitNormArgs; MinMaxNormArgs; MaxNormArgs
                DisposeResult; InputSpecArgs; Serializable; SerializableConstructor; RequestDetails
                IORouterRegistry; ModelStoreManager; ModelJSON; WeightsManifestGroupConfig
                EncodeWeightsResult; CompositeArrayBuffer; InputConfig; LoadOptions
                WebcamIterator; MicrophoneIterator; CaptureResult; WebcamConfig
                MicrophoneConfig; CSVConfig; DataSource; ByteChunkIterator; StringIterator
                LazyIterator; DeepMapResult; ComputeGradientsResult; NamedTensor; GPUData; 
                ConvLayerArgs; BaseConvLayerArgs; SpatialDropout1DLayerConfig; RepeatVectorLayerArgs
                ReshapeLayerArgs; PermuteLayerArgs; FlattenLayerArgs; EmbeddingLayerArgs
                DropoutLayerArgs; DenseLayerArgs; ActivationLayerArgs; PReLULayerArgs
                ThresholdedReLULayerArgs; SoftmaxLayerArgs; ReLULayerArgs; LeakyReLULayerArgs
                ELULayerArgs; RNNCell; Function; ProfileInfo; KernelBackend; InputSpec
                InferenceModel; LSTMLayerArgs; SimpleRNNLayerArgs; StackedRNNCellsArgs
                SimpleRNNCellLayerArgs; ConvRNN2DCellArgs; BaseRNNLayerArgs; LayerArgs
                GlobalPooling2DLayerArgs; Pooling3DLayerArgs; Pooling2DLayerArgs
                Pooling1DLayerArgs; LayerNormalizationLayerArgs; BatchNormalizationLayerArgs
                DotLayerArgs; ConcatenateLayerArgs; UpSampling2DLayerArgs;SeparableConvLayerArgs
                DepthwiseConv2DLayerArgs; Cropping2DLayerArgs; BaseConv; ConvLSTM2DCellArgs
                BidirectionalLayerArgs; RNN; RandomWidthArgs; BaseRandomLayerArgs
                CategoryEncodingArgs; ResizingArgs; CenterCropArgs;RescalingArgs
                MaskingArgs; GaussianNoiseArgs; GaussianDropoutArgs; AlphaDropoutArgs
                ZeroPadding2DLayerArgs; InputLayerArgs; RNNLayerArgs; WrapperLayerArgs
                GRUCellLayerArgs; GRULayerArgs; LSTMCellLayerArgs; ConvLSTM2DArgs
                ConvRNN2DLayerArgs; RepeatVector; Permute; Flatten; Embedding; Dropout
                Dense; Activation; ThresholdedReLU; Softmax; ReLU; PReLU; LeakyReLU
                ELU; UpSampling2D; SeparableConv2D; SeparableConv; DepthwiseConv2D
                Cropping2D; Conv3D; Conv2DTranspose; Conv2D; Conv1D; Conv; MaxPooling1D
                GlobalMaxPooling2D; GlobalMaxPooling1D; GlobalAveragePooling2D; 
                GlobalPooling2D; GlobalAveragePooling1D; GlobalPooling1D; AveragePooling3D
                Pooling3D; AveragePooling2D; Pooling2D; AveragePooling1D; Pooling1D
                LayerNormalization; BatchNormalization; Multiply; Minimum; Maximum
                Dot; Concatenate; Average; Add; Merge; SpatialDropout1D; Reshape
                RMSPropOptimizer; AdadeltaOptimizer; AdamOptimizer; AdagradOptimizer
                MomentumOptimizer; SGDOptimizer; RandomWidth; CategoryEncoding; Resizing
                CenterCrop; Rescaling; Masking; GaussianNoise; GaussianDropout; AlphaDropout
                ZeroPadding2D; InputLayer; TimeDistributed; Bidirectional; Wrapper
                StackedRNNCells; SimpleRNNCell; SimpleRNN; ConvLSTM2DCell; LSTMCell
                EarlyStopping; EarlyStoppingCallbackArgs; PixelData; DrawOptions
                ContextOptions; WebGLContextAttributes; ImageOptions; L2Config; L1L2Config
                L1Config; Zeros; VarianceScalingArgs; TruncatedNormalArgs; RandomUniformArgs
                RandomNormalArgs; OrthogonalArgs; LeCunUniformArgs; LeCunNormalArgs
                IdentityArgs; HeUniformArgs; HeNormalArgs; GlorotUniformArgs; Engine
                CustomGradientFuncResult; ValueAndGradsResultReturn; VariableGradsResult
                ValueAndGradResultReturn; MeshgridThridParameter; UniqueResult
                TopkResult; MomentsResult; Platform; LayerVariable; SequentialArgs
                History; BaseCallback; ModelFitDatasetArgs; ModelFitArgs; ModelEvaluateArgs
                ModelCompileArgs; AdamaxOptimizer; ContainerArgs
            ]
        ]

[<Sealed>]
type Extension() =  
    interface IExtension with
        member ext.Assembly =
            Definition.Assembly

[<assembly: Extension(typeof<Extension>)>]
do ()
