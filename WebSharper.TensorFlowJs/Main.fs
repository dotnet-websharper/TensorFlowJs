namespace WebSharper.TensorFlowJs

open WebSharper
open WebSharper.JavaScript
open WebSharper.InterfaceGenerator

module Definition =
    [<AutoOpen>]
    module Types = 
        let DataId = T<obj>
        let Kwargs = T<obj[]>
        let DataType = T<string>
        let NamedTensorMap = T<obj[]>
        let Shape = T<int[]>
        let ShapeOrArray = Shape + !|Shape
        let StringOrArray = T<string> + T<string[]>
        let IntOrIntArray = T<int> + T<int[]>
        let TypedArray = T<Uint8Array> + T<Uint8ClampedArray> + T<Int32Array> + T<Float32Array>
        let TensorLike = TypedArray + T<int> + T<bool> + T<string> + T<int[]>
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
        (*let FromConfigMethod = SerializableConstructor[T]?cls * ConfigDict?config ^-> T*)       

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

    [<AutoOpen>]
    module TensorFlow = 
        let TFTensor =
            Class "Tf.Tensor"
            |=> Inherits TensorInfo
            |+> Static [
                Constructor (Shape?shape * T<string>?dtype * DataId?dataId * T<int>?id)
            ]
            |+> Instance [
                "id" =? T<int> 
                "dataId" =? DataId 
                "shape" =? Shape
                "size" =? T<int> 
                "dtype" =? T<string>
                "rankType" =? Rank 
                "kept" =@ T<bool>
                "scopeId" =@ T<int> 
                "kerasMask" =? !? TSelf 
                "strides" =? T<int[]> 
            ] 

        let TensorAndArrayType = TFTensor + TypedArray + T<float[]>
        let LossOrMetricFn = TFTensor?yTrue * TFTensor?yPred ^-> TFTensor

        let GPUData =
            Pattern.Config "GPUData" {
                Required = [
                    "tensorRef", TFTensor.Type
                ]
                Optional = [
                    "texture", T<WebGL.Texture>
                    "buffer", T<WebGPU.GPUBuffer>
                    "texShape", !|T<int>
                ]
            }

        let TFTensorBuffer = 
            Class "Tf.TensorBuffer" 
            |+> Static [
                Constructor (Shape?shape * T<string>?dtype * !?DataTypeMap?values)
            ]
            |+> Instance [       
                "size" =@ T<int>
                "shape" =@ Shape
                "strides" =@ T<int[]>
                "values" =@ DataTypeMap

                "set" => SingleValueMap?value * (!|T<int>)?locs ^-> T<unit>
                "get" => (!|T<int>)?locs ^-> SingleValueMap
                "toTensor" => T<unit> ^-> TFTensor
            ]

        let TFVariable = 
            Class "Tf.Variable"
            |=> Inherits TFTensor
            |+> Static [
                Constructor (TFTensor?initialValue * T<bool>?trainable * T<string>?name * T<int>?tensorId)
            ]
            |+> Instance [
                "name" =@ T<string>

                "assign" => TFTensor?newValue ^-> T<unit>
            ]

        TFTensor 
        |+> Instance [
            "buffer" => T<unit> ^-> T<Promise<_>>[TFTensorBuffer]
            "bufferSync" => T<unit> ^-> TFTensorBuffer
            "array" => T<unit> ^-> T<Promise<_>>[!|T<int>]
            "arraySync" => T<unit> ^-> !|T<int>
            "data" => T<unit> ^-> T<Promise<_>>[DataTypeMap]
            "dataToGPU" => !?DataToGPUOptions?options ^-> GPUData
            "dataSync" => T<unit> ^-> DataTypeMap
            "dispose" => T<unit> ^-> T<unit>
            "print" => !?T<bool>?verbose ^-> T<unit>
            "clone" => T<unit> ^-> TSelf
            "toString" => !?T<bool>?verbose ^-> T<string>
        ] |> ignore

        let TensorOrArray = TFTensor + !| TFTensor
        let TensorOrArrayOrMap = TFTensor + !| TFTensor + T<obj>
        let UnitToTensorFunc = T<unit> ^-> TFTensor

        let NamedTensor = 
            Pattern.Config "NamedTensor" {
                Required = [
                    "name", T<string>
                    "tensor", TFTensor.Type
                ]
                Optional = []
            } 

        let ComputeGradientsResult = 
            Pattern.Config "ComputeGradientsResult" {
                Required = [
                    "value", TFTensor.Type
                    "grads", NamedTensorMap
                ]
                Optional = []
            }

        let TFTrainOptimizer =
            Class "tf.Train.Optimizer"
            |=> Inherits Serializable
            |+> Instance [
                "minimize" => UnitToTensorFunc?f * !?T<bool>?returnCost * !?(!|TFVariable)?varList ^-> TFTensor + T<unit>
                "computeGradients" => UnitToTensorFunc?f * !?(!|TFVariable)?varList ^-> ComputeGradientsResult
                "applyGradients" => (NamedTensorMap + !|NamedTensor)?variableGradients ^-> T<unit>
            ]

        let TFInitializer = 
            Class "tf.Constraints.Constraint"
            |=> Inherits Serializable

        let TFConstraint = 
            Class "tf.Initializers.Initializer"
            |=> Inherits Serializable

        let TensorContainer = T<unit> + TFTensor + T<string> + T<float> + T<bool> + T<obj[]> + !|TSelf + T<Float32Array> + T<Int32Array> + T<Uint8Array>

        let TFDataset =
            Generic -- fun t o ->
                let tToBoolFunc = t?value ^-> T<bool>
                let tToUnitFunc = t?input ^-> T<unit>
                let tToOFunc = t?value ^-> o
                let tToOFuncAsync = t?value ^-> T<Promise<_>>[o]

                Class "tf.Data.Dataset"
                |+> Instance [
                    "batch" => T<int>?batchSize * !?T<bool>?smallLastBatch ^-> TSelf[t]
                    "concatenate" => TSelf[t]?dataset ^-> TSelf[t]
                    "filter" => tToBoolFunc?predicate ^-> TSelf[t]
                    "forEachAsync" => tToUnitFunc?f ^-> T<Promise<unit>>
                    "map" => tToOFunc?transform ^-> TSelf[o]
                    "mapAsync" => tToOFuncAsync?transform ^-> TSelf[o]
                    "prefetch" => T<int>?bufferSize ^-> TSelf[t]
                    "repeat" => !?T<int>?count ^-> TSelf[t]
                    "skip" => T<int>?count ^-> TSelf[t]
                    "shuffle" => T<int>?bufferSize * !?T<string>?seed * !? T<bool>?reshuffleEachIteration ^-> TSelf[t]
                    "take" => T<int>?count ^-> TSelf[t]
                    "toArray" => T<unit> ^-> T<Promise<_>>[!|t]
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
                    "weights", !|TFTensor
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

        let TFLayer = 
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
                    "outboundLayer", TFLayer.Type
                    "inboundLayers", !|TFLayer
                    "nodeIndices", T<int[]>
                    "tensorIndices", T<int[]>
                    "inputTensors", !|SymbolicTensor
                    "outputTensors", !|SymbolicTensor
                    "inputMasks", !|TFTensor
                    "outputMasks", !|TFTensor
                    "inputShapes", ShapeOrArray
                    "outputShapes", ShapeOrArray
                ]
                Optional = []
            }

        SymbolicTensor
        |+> Static [
                 Constructor (DataType?dtype * Shape?shape * TFLayer?sourceLayer * 
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
                "apply" => TFTensor?x ^-> TFTensor
            ]

        let SymbolicTensorOrArray = SymbolicTensor + !|SymbolicTensor

        let Initializer =
            Class "Initializer"
            |=> Inherits Serializable
            |+> Instance [
                "fromConfigUsesCustomObjects" => T<unit> ^-> T<bool> 
                "apply" => Shape?shape * !?DataType?dtype ^-> TFTensor 
                "getConfig" => T<unit> ^-> ConfigDict 
            ]

        let Constraint =
            Class "Constraint"
            |=> Inherits Serializable
            |+> Instance [
                "apply" => TFTensor?w ^-> TFTensor 
                "getConfig" => T<unit> ^-> ConfigDict 
            ]

        let LayerVariable =
            Class "LayerVariable"
            |+> Static [
                Constructor (TFTensor?``val`` * !?DataType?dtype * !?T<string>?name * !?T<bool>?trainable * !?Constraint?``constraint``)
            ]
            |+> Instance [
                "read" => T<unit> ^-> TFTensor 
                "write" => TFTensor?newVal ^-> TSelf 
                "dispose" => T<unit> ^-> T<unit> 
            ]

        let RegularizerFn = T<unit> ^-> TFTensor
        let RegularizerFnOrArray = RegularizerFn + !|RegularizerFn

        TFLayer
        |=> Inherits Serializable
        |+> Static [
            Constructor (LayerArgs?args ^-> T<obj>)
            "nodeKey" => TFLayer?layer * T<int>?nodeIndex ^-> T<unit>
        ]
        |+> Instance [
            "apply" => (TensorOrArray + SymbolicTensorOrArray)?inputs * !?Kwargs?kwargs ^-> TensorOrArray + SymbolicTensorOrArray
            "countParams" => T<unit> ^-> T<int> 
            "build" => ShapeOrArray?inputShape ^-> T<unit> 
            "getWeights" => T<bool>?trainableOnly ^-> !|TFTensor 
            "setWeights" => (!|TFTensor)?weights ^-> T<unit> 
            "addWeight" => T<string>?name * Shape?shape * !?DataType?dtype * !?Initializer?initializer * !?Regularizer?regularizer * !?T<bool>?trainable * !?Constraint?``constraint`` * !?T<Function>?getInitializerFunc ^-> LayerVariable 
            "addLoss" => RegularizerFnOrArray?losses ^-> T<unit> 
            "computeOutputShape" => ShapeOrArray?inputShape ^-> ShapeOrArray
            "getConfig" => T<unit> ^-> ConfigDict 
            "dispose" => T<unit> ^-> DisposeResult 
        ] |> ignore

        let GraphNode =
            Pattern.Config "GraphNode" {
                Required = [
                    "inputs", !|TFTensor
                    "attrs", T<obj[]>
                ]
                Optional = []
            }


        let OpExecutor = GraphNode?node ^-> TensorOrArray + T<Promise<_>>[TensorOrArray]      
        
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
                    "tensor", TFTensor.Type
                    "tensors", !|TFTensor
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
                "validationData" =@ TensorOrArray
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
                    "optimizer", T<string> + TFTrainOptimizer
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
                    "sampleWeight", TFTensor.Type
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
                    "validationData", !| (TFTensor + !|TFTensor) 
                    "shuffle", T<bool>
                    "classWeight", T<obj>
                    "sampleWeight", TFTensor.Type
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
                        "validationData", !| TensorOrArrayOrMap + TFDataset[t]
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
                    "layers", !| TFLayer
                    "name", T<string>
                ]
            }

        let TFSymbolicTensor = 
            Class "Tf.SymbolicTensor"
            |+> Instance [
                "id" =? T<int>
                "name" =? T<string>
                "originalName" =? T<string>
                "rank" =? T<int>
                "nodeIndex" =@ T<int>
                "tensorIndex" =@ T<int>
            ]

        let TFSymbolicTensorArray = !| TFSymbolicTensor

        TFSymbolicTensor
            |+> Static [
                Constructor (T<string>?dtype * Shape?shape * TFLayer?sourceLayer * TFSymbolicTensorArray?inputs * T<obj>?callArgs * !? T<string>?name * !? T<int>?outputTensorIndex)
                
            ] |> ignore

        let ContainerArgs =
            Pattern.Config "ContainerArgs" {
                Required = [
                    "inputs", TFSymbolicTensor + !|TFSymbolicTensor
                    "outputs", TFSymbolicTensor + !|TFSymbolicTensor
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
                "execute" => TensorOrArrayOrMap?inputs * (T<string> + !| T<string>)?outputs ^-> TensorOrArray
            ]

        let TFGraphModel =
            Class "Tf.GraphModel"
            |=> Inherits InferenceModel
            |+> Instance [ 
                "load" => T<unit> ^-> T<Promise<bool>>
                "loadSync" => ModelArtifacts?artifacts ^-> T<bool>
                "save" => HandleOrString?handlerOrURL * !? SaveConfig?config ^-> T<Promise<_>>[SaveResult]
                "predict" => TensorOrArrayOrMap * !? ModelPredictConfig?config ^-> TensorOrArrayOrMap
                "predictAsync" => TensorOrArrayOrMap?input * !? ModelPredictConfig?config ^-> T<Promise<_>>[TensorOrArrayOrMap]
                "execute" => TensorOrArrayOrMap?input * !? (T<string> + !| T<string>)?outputs ^-> (TFTensor + !| TFTensor)
                "executeAsync" => TensorOrArrayOrMap?input * !? (T<string> + !| T<string>)?outputs ^-> T<Promise<_>>[(TFTensor + !| TFTensor)]
                "getIntermediateTensors" => T<unit> ^-> T<obj>
                "disposeIntermediateTensors" => T<unit> ^-> T<unit>
                "dispose" => T<unit> ^-> T<unit>
            ]

        let TFLayersModel =
            Class "tf.LayersModel"
            |=> Inherits InferenceModel
            |+> Instance [
                "summary" => !?T<int>?lineLength * T<int[]>?positions * !?PrintFnType?printFn ^-> T<unit>
                "compile" => ModelCompileArgs?args ^-> T<unit>
                "evaluate" => TensorOrArray?x * TensorOrArray?y * !?ModelEvaluateArgs?args ^-> TensorOrArray
                "evaluateDataset" => TFDataset?dataset * !?ModelEvaluateDatasetArgs?args ^-> T<Promise<_>>[TensorOrArray]
                "predict" => TensorOrArray?x * !?ModelPredictArgs?args ^-> TensorOrArray
                "predictOnBatch" => TensorOrArray?x ^-> TensorOrArray
                "fit" => TensorOrArrayOrMap?x * TensorOrArrayOrMap?y * !?ModelFitArgs?args ^-> T<Promise<_>>[History]
                Generic - fun t ->
                    "fitDataset" => TFDataset[t]?dataset * !?ModelFitDatasetArgs[t]?args ^-> T<Promise<_>>[History]
                "trainOnBatch" => TensorOrArrayOrMap?x * TensorOrArrayOrMap?y ^-> T<Promise<_>>[T<int> + T<int[]>]
                "save" => HandleOrString?handlerOrURL * !?SaveConfig?config ^-> T<Promise<_>>[SaveResult]
                "getLayer" => T<string>?name ^-> TFLayer
            ]

        let TFSequential = 
            Class "tf.Sequential"
            |=> Inherits TFLayersModel
            |+> Static [
                Constructor !? SequentialArgs?args
            ]
            |+> Instance [
                "add" => TFLayer?layer ^-> T<unit>
                "pop" => T<unit> ^-> T<unit>
            ]

        let TFFunction = 
            Class "TFFunction"
            |=> Inherits TFLayersModel

        let RNNCell =
            Class "tf.RNNCell"
            |=> Inherits TFLayer
            |+> Instance [
                "stateSize" =@ (T<int> + T<int[]>)
                "dropoutMask" =@ TensorOrArray
                "recurrentDropoutMask" =@ TensorOrArray
            ]

        let ELULayerArgs = 
            Pattern.Config "ELULayerArgs" {
                Required = []
                Optional = [
                    "alpha", T<float>
                ]
            }
            |=> Inherits LayerArgs

        let LeakyReLULayerArgs = ELULayerArgs

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
                Required = ["layer", TFLayer.Type]
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
            |=> Inherits TFLayer
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

                "getState" => T<unit> ^-> !| TFTensor
                "setSate" => TFTensor?states ^-> T<unit>
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
            |=> Inherits TFLayer
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
            |=> Inherits TFLayer
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
            |=> Inherits TFLayer
            |+> Static [
                Constructor (UpSampling2DLayerArgs?args)
            ] 

        let ELU = 
            Class "ELU"
            |=> Inherits TFLayer
            |+> Static [
                Constructor !?ELULayerArgs?args
            ]

        let LeakyReLU = 
            Class "LeakyReLU"
            |=> Inherits TFLayer
            |+> Static [
                Constructor !?LeakyReLULayerArgs?args
            ]

        let PReLU = 
            Class "PReLU"
            |=> Inherits TFLayer
            |+> Static [
                Constructor !?PReLULayerArgs?args
            ]

        let ReLU = 
            Class "ReLU"
            |=> Inherits TFLayer
            |+> Static [
                Constructor !?ReLULayerArgs?args
            ]

        let Softmax = 
            Class "Softmax"
            |=> Inherits TFLayer
            |+> Instance [
                Constructor !?SoftmaxLayerArgs?args
            ]

        let ThresholdedReLU = 
            Class "ThresholdedReLU"
            |=> Inherits TFLayer
            |+> Instance [
                Constructor !?ThresholdedReLULayerArgs?args
            ]

        let Activation = 
            Class "Activation"
            |=> Inherits TFLayer
            |+> Instance [
                Constructor ActivationLayerArgs?args
            ]

        let Dense = 
            Class "Dense"
            |=> Inherits TFLayer
            |+> Instance [
                Constructor DenseLayerArgs?args
            ]

        let Dropout = 
            Class "Dropout"
            |=> Inherits TFLayer
            |+> Instance [
                Constructor DropoutLayerArgs?args
            ]

        let Embedding = 
            Class "Embedding"
            |=> Inherits TFLayer
            |+> Instance [
                Constructor EmbeddingLayerArgs?args
            ]

        let Flatten = 
            Class "Flatten"
            |=> Inherits TFLayer
            |+> Instance [
                Constructor !?FlattenLayerArgs?args
            ]

        let Permute = 
            Class "Permute"
            |=> Inherits TFLayer
            |+> Instance [
                Constructor PermuteLayerArgs?args
            ]

        let RepeatVector = 
            Class "RepeatVector"
            |=> Inherits TFLayer
            |+> Instance [
                Constructor RepeatVectorLayerArgs?args
            ]

        let Reshape = 
            Class "Reshape"
            |=> Inherits TFLayer
            |+> Instance [
                Constructor ReshapeLayerArgs?args
            ]

        let SpatialDropout1D = 
            Class "SpatialDropout1D"
            |=> Inherits TFLayer
            |+> Instance [
                Constructor SpatialDropout1DLayerConfig?args
            ]

        let Merge = 
            Class "Merge" 
            |=> Inherits TFLayer
            |+> Static [
                Constructor !?LayerArgs?args
            ]
            |+> Instance [
                "mergeFunction" => (!|TFTensor)?inputs ^-> TFTensor
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
            |=> Inherits TFLayer
            |+> Static [
                Constructor !?BatchNormalizationLayerArgs?args
            ]

        let LayerNormalization = 
            Class "LayerNormalization" 
            |=> Inherits TFLayer
            |+> Static [
                Constructor !?LayerNormalizationLayerArgs?args
            ]

        let Pooling1D = 
            Class "Pooling1D" 
            |=> Inherits TFLayer
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
                "poolingFunction" => TFTensor?input * T<int[]>?poolSize * T<int[]>?strides * T<string>?padding * T<string>?dataFormat ^-> TFTensor 
            ]

        let Pooling2D = 
            Class "Pooling2D" 
            |=> Inherits TFLayer
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
                "poolingFunction" => TFTensor?input * T<int[]>?poolSize * T<int[]>?strides * T<string>?padding * T<string>?dataFormat ^-> TFTensor 
            ]

        let Pooling3D = 
            Class "Pooling3D" 
            |=> Inherits TFLayer
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
                "poolingFunction" => TFTensor?input * T<int[]>?poolSize * T<int[]>?strides * T<string>?padding * T<string>?dataFormat ^-> TFTensor 
            ]

        let GlobalPooling1D = 
            Class "GlobalPooling1D" 
            |=> Inherits TFLayer
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
            |=> Inherits TFLayer
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
                "poolingFunction" => TFTensor?input * T<int[]>?poolSize * T<int[]>?strides * T<string>?padding * T<string>?dataFormat ^-> TFTensor 
            ]

        let MaxPooling2D = 
            Class "MaxPooling2D" 
            |=> Inherits Pooling2D
            |+> Static [
                Constructor Pooling2DLayerArgs?args
            ]
            |+> Instance [
                "poolingFunction" => TFTensor?input * T<int[]>?poolSize * T<int[]>?strides * T<string>?padding * T<string>?dataFormat ^-> TFTensor 
            ]

        let MaxPooling3D = 
            Class "MaxPooling3D" 
            |=> Inherits Pooling3D
            |+> Static [
                Constructor Pooling3DLayerArgs?args
            ]
            |+> Instance [
                "poolingFunction" => TFTensor?input * T<int[]>?poolSize * T<int[]>?strides * T<string>?padding * T<string>?dataFormat ^-> TFTensor 
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
            |=> Inherits TFLayer
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
            |=> Inherits TFLayer
            |+> Static [
                Constructor (InputLayerArgs?args)
            ]

        let ZeroPadding2D =
            Class "ZeroPadding2D"
            |=> Inherits TFLayer
            |+> Static [
                Constructor (!?ZeroPadding2DLayerArgs?args)
            ]

        let AlphaDropout =
            Class "AlphaDropout"
            |=> Inherits TFLayer
            |+> Static [
                Constructor (AlphaDropoutArgs?args)
            ]
            |+> Instance [
                "_getNoiseShape" => (TFTensor + !|TFTensor)?inputs ^-> T<unit> 
            ]

        let GaussianDropout =
            Class "GaussianDropout"
            |=> Inherits TFLayer
            |+> Static [
                Constructor (GaussianDropoutArgs?args)
            ]

        let GaussianNoise =
            Class "GaussianNoise"
            |=> Inherits TFLayer
            |+> Static [
                Constructor (GaussianNoiseArgs?args)
            ]

        let Masking =
            Class "Masking"
            |=> Inherits TFLayer
            |+> Static [
                Constructor (!?MaskingArgs?args)
            ]

        let Rescaling =
            Class "Rescaling"
            |=> Inherits TFLayer
            |+> Static [
                Constructor (RescalingArgs?args)
            ]

        let CenterCrop =
            Class "CenterCrop"
            |=> Inherits TFLayer
            |+> Static [
                Constructor (CenterCropArgs?args)
            ]
            |+> Instance [
                "centerCrop" => TFTensor?inputs * T<int>?hBuffer * T<int>?wBuffer * T<int>?height * T<int>?width * T<int>?inputHeight * T<int>?inputWidth * DataType?dtype ^-> TFTensor + !|TFTensor
                "upsize" => TFTensor?inputs * T<int>?height * T<int>?width * DataType?dtype ^-> TFTensor + !|TFTensor
            ]

        let Resizing =
            Class "Resizing"
            |=> Inherits TFLayer
            |+> Static [
                Constructor (ResizingArgs?args)
            ]

        let CategoryEncoding =
            Class "CategoryEncoding"
            |=> Inherits TFLayer
            |+> Static [
                Constructor (CategoryEncodingArgs?args)
            ]

        let RandomWidth =
            Class "RandomWidth"
            |=> Inherits TFLayer
            |+> Static [
                Constructor (RandomWidthArgs?args)
            ]

    [<AutoOpen>]
    module TFCore = 
        let TF = 
            Class "tf"
            |+> Static [
                // Tensors / Creation
                "tensor" => TensorValuesType?values * !?Shape?shape * !?T<string>?dtype ^-> TFTensor
                "scalar" => ScalarLike?value * !?T<string>?dtype ^-> TFTensor
                "tensor1d" => TensorLike1D?values * !?Shape?shape * !?T<string>?dtype ^-> TFTensor
                "tensor2d" => TensorLike2D?values * !?Shape?shape * !?T<string>?dtype ^-> TFTensor
                "tensor3d" => TensorLike3D?values * !?Shape?shape * !?T<string>?dtype ^-> TFTensor
                "tensor4d" => TensorLike4D?values * !?Shape?shape * !?T<string>?dtype ^-> TFTensor
                "tensor5d" => TensorLike5D?values * !?Shape?shape * !?T<string>?dtype ^-> TFTensor
                "tensor6d" => TensorLike6D?values * !?Shape?shape * !?T<string>?dtype ^-> TFTensor
                "buffer" => Shape?shape * !?T<string>?dtype * !?DataTypeMap?values ^-> TFTensorBuffer
                "fill" => Shape?shape * (T<int> + T<string>)?value * !?T<string>?dtype ^-> TFTensor
                "eye" => T<float>?numRows * !?T<float>?numColumns * !?T<float[]>?batchShape * !?T<string>?dtype ^-> TFTensor
                "ones" => Shape?shape * !?T<string>?dtype ^-> TFTensor
                "onesLike" => TensorAndArrayType?x ^-> TFTensor
                "zeros" => Shape?shape * !?T<string>?dtype ^-> TFTensor
                "zerosLike" => TensorAndArrayType?x ^-> TFTensor
                "clone" => TensorAndArrayType?x  ^-> TFTensor
                "complex" => TensorAndArrayType?real * TensorAndArrayType?imag ^-> TFTensor
                "diag" => TFTensor?x ^-> TFTensor
                "range" => T<int>?start * T<int>?stop * !?T<int>?step * !?T<string>?dtype ^-> TFTensor
                "real" => TensorAndArrayType?input ^-> TFTensor
                "imag" => TensorAndArrayType?input ^-> TFTensor
                "variable" => TFTensor?initialValue  * !?T<bool>?trainable * !?T<string>?name * !?T<string>?dtype ^-> TFVariable
                "print" => TFTensor?x * !?T<bool>?verbose ^-> T<unit>
                "truncatedNormal" => Shape?shape * !?T<float>?mean * !?T<float>?stdDev * !?T<string>?dtype * !?T<float>?seed ^-> TFTensor
                "oneHot" => TensorAndArrayType?indices  * T<float>?depth * !?T<float>?onValue * !?T<float>?offValue * !?T<string>?dtype ^-> TFTensor
                "linspace" => T<float>?start * T<float>?stop * T<float>?num ^-> TFTensor

                // Tensors / Transformations
                "batchToSpaceND" => TensorAndArrayType * T<uint[]>?blockShape * T<uint[][]>?crops ^-> TFTensor
                "broadcastArgs" => TensorAndArrayType?s0 * TensorAndArrayType?s1 ^-> TFTensor
                "broadcastTo" => TensorAndArrayType?x * Shape?shape ^-> TFTensor
                "cast" => TensorAndArrayType?x * T<string>?dtype ^-> TFTensor
                "depthToSpace" => TensorAndArrayType?x * T<int>?blockSize * !?T<string>?dataFormat ^-> TFTensor
                "ensureShape" => TFTensor?x * Shape?shape ^-> TFTensor
                "expandDims" => TensorAndArrayType?x * !?T<int>?axis ^-> TFTensor
                "mirrorPad" => TensorAndArrayType?x * T<int[][]>?paddings * T<string>?mode ^-> TFTensor
                "pad" => TensorAndArrayType?x * T<int[][]>?paddings * !?T<int>?constantValue ^-> TFTensor
                "reshape" => TensorAndArrayType?x * Shape?shape ^-> TFTensor
                "setdiff1dAsync" => TensorAndArrayType?x * TensorAndArrayType?y ^-> T<Promise<_>>[!| !| TFTensor]
                "spaceToBatchND" => TensorAndArrayType?x * T<int[]>?blockShape * T<int[][]>?paddings ^-> TFTensor
                "squeeze" => TensorAndArrayType?x * !?T<int[]>?axis ^-> TFTensor

                // Tensors / Slicing and Joining
                "booleanMaskAsync" => TensorAndArrayType?tensor * TensorAndArrayType?mask * !?T<int>?axis ^-> T<Promise<_>>[TFTensor]
                "concat" => (!|TFTensor)?tensors * !?T<int>?axis ^-> TFTensor
                "gather" => TensorAndArrayType?x * TensorAndArrayType?indices * !?T<int>?axis * !?T<int>?batchDims ^-> TFTensor
                "reverse" => TensorAndArrayType?x * !?IntOrIntArray?axis ^-> TFTensor
                "slice" => TensorAndArrayType?x * IntOrIntArray?``begin`` * !?IntOrIntArray?size ^-> TFTensor
                "split" => TensorAndArrayType?x * IntOrIntArray?numOrSizeSplits * !?T<int>?axis ^-> !|TFTensor
                "stack" => (!|TFTensor)?tensors * !?T<int>?axis ^-> TFTensor
                "tile" => TensorAndArrayType?x * T<int[]>?reps ^-> TFTensor
                "unstack" => TensorAndArrayType?x * !?T<int>?axis ^-> !|TFTensor

                // Tensors / Matrices
                "einsum" => T<string>?equation * (!|TFTensor)?tensors ^-> TFTensor

                // Tensors / Random
                "multinomial" => LogitsType?logits * T<int>?numSamples * !?T<int>?seed * !?T<bool>?normalized ^-> (TensorLike1D + TensorLike2D)
                "rand" => Shape?shape * (T<unit> ^-> T<float>)?randFunction * !?T<string>?dtype ^-> TFTensor
                "randomGamma" => Shape?shape * T<float>?alpha * !?T<float>?beta * !?T<string>?dtype * !?T<int>?seed ^-> TFTensor
                "randomNormal" => Shape?shape * !?T<float>?mean * !?T<float>?stdDev * !?T<string>?dtype * !?T<int>?seed ^-> TFTensor
                "randomStandardNormal" => Shape?shape * !?T<string>?dtype * !?T<int>?seed ^-> TFTensor
                "randomUniform" => Shape?shape * !?T<float>?minval * !?T<float>?maxval * !?T<string>?dtype * !?(T<int> + T<string>)?seed ^-> TFTensor
                "randomUniformInt" => Shape?shape * T<int>?minval * T<int>?maxval * !?(T<int> + T<string>)?seed ^-> TFTensor

                // Models / Creation
                "sequential" => !?SequentialArgs?config ^-> TFSequential
                "model" => ContainerArgs?args ^-> TFLayersModel

                // Models / Inputs
                "input" => InputConfig?config ^-> TFSymbolicTensor

                // Models / Loading
                "loadGraphModel" => HandleOrString?modelUrl * LoadOptions?options * IO?tfio ^-> T<Promise<_>>[TFGraphModel]
                "browserDownloads" => !?T<string>?fileNamePrefix  ^-> IOHandler
                "browserFiles" => T<File[]>?files  ^-> IOHandler
                "http" => T<string>?path * !?LoadOptions?loadOptions ^-> IOHandler
                "loadGraphModelSync" => (IOHandlerSync + ModelArtifacts + (ModelJSON * T<ArrayBuffer>))?modelSource ^-> TFGraphModel

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

                // Layers

                // Operations

                // Training

                // Performance

                // Environment

                // Constraints

                // Initializers

                // Regularizers

                // Data

                // Util

                // Backends

                // Browser

                // Metrics

                // Callbacks
            ]

        let TFLayers = 
            Class "tf.Layers"
            |+> Instance [
                // Layers / Advanced Activation
                "elu" => !?ELULayerArgs?args ^-> ELU
                "leakyReLU" => !?LeakyReLULayerArgs?args ^-> LeakyReLU
                "prelu" => !?PReLULayerArgs?args ^-> PReLU
                "reLU" => !?ReLULayerArgs?args ^-> ReLU
                "softmax" => !?SoftmaxLayerArgs?args ^-> Softmax
                "thresholdedReLU" => !?ThresholdedReLULayerArgs?args ^-> ThresholdedReLU

                // Layers / Basic
                "activation" => ActivationLayerArgs?args ^-> Activation
                "dense" => DenseLayerArgs?args ^-> Dense
                "dropout" => DropoutLayerArgs?args ^-> Dropout
                "embedding" => EmbeddingLayerArgs?args ^-> Embedding
                "flatten" => !?FlattenLayerArgs?args ^-> Flatten
                "permute" => PermuteLayerArgs?args ^-> Permute
                "repeatVector" => RepeatVectorLayerArgs?args ^-> RepeatVector
                "reshape" => ReshapeLayerArgs?args ^-> Reshape
                "spatialDropout1d" => SpatialDropout1DLayerConfig?args ^-> SpatialDropout1D
    
                // Layers / Convolutional
                "conv1d" => ConvLayerArgs?args ^-> Conv1D
                "conv2d" => ConvLayerArgs?args ^-> Conv2D
                "conv2dTranspose" => ConvLayerArgs?args ^-> Conv2DTranspose
                "conv3d" => ConvLayerArgs?args ^-> Conv3D
                "cropping2D" => Cropping2DLayerArgs?args ^-> Cropping2D
                "depthwiseConv2d" => DepthwiseConv2DLayerArgs?args ^-> DepthwiseConv2D
                "separableConv2d" => SeparableConvLayerArgs?args ^-> SeparableConv2D
                "upSampling2d" => UpSampling2DLayerArgs?args ^-> UpSampling2D

                // Layers / Merge
                "add" => !?LayerArgs?args ^-> Add
                "average" => !?LayerArgs?args ^-> Average
                "concatenate" => !?ConcatenateLayerArgs?args ^-> Concatenate
                "maximum" => !?LayerArgs?args ^-> Maximum
                "minimum" => !?LayerArgs?args ^-> Minimum
                "multiply" => !?LayerArgs?args ^-> Multiply

                // Layers / Normalization
                "batchNormalization" => !?BatchNormalizationLayerArgs?args ^-> BatchNormalization
                "layerNormalization" => !?LayerNormalizationLayerArgs?args ^-> LayerNormalization

                // Layers / Pooling
                "averagePooling1d" => Pooling1DLayerArgs?args ^-> AveragePooling1D 
                "averagePooling2d" =>  Pooling2DLayerArgs?args ^-> AveragePooling2D 
                "averagePooling3d" =>  Pooling3DLayerArgs?args ^-> AveragePooling3D 
                "globalAveragePooling1d" => !?LayerArgs?args ^-> GlobalAveragePooling1D 
                "globalAveragePooling2d" => GlobalPooling2DLayerArgs?args ^-> GlobalAveragePooling2D 
                "globalMaxPooling1d" =>  !?LayerArgs?args ^-> GlobalMaxPooling1D 
                "globalMaxPooling2d" =>  GlobalPooling2DLayerArgs?args ^-> GlobalMaxPooling2D 
                "maxPooling1d" =>  Pooling1DLayerArgs?args ^-> MaxPooling1D 
                "maxPooling2d" =>  Pooling2DLayerArgs?args ^-> MaxPooling2D 
                "maxPooling3d" =>  Pooling3DLayerArgs?args ^-> MaxPooling3D 
                
                // Layers / Recurrent
                "convLstm2d" => ConvLSTM2DArgs?args ^-> ConvLSTM2D 
                "convLstm2dCell" => ConvLSTM2DCellArgs?args ^-> ConvLSTM2DCell
                "gru" => GRULayerArgs?args ^-> GRU
                "gruCell" => GRUCellLayerArgs?args ^-> GRUCell
                "lstm" => LSTMLayerArgs?args ^-> LSTM
                "lstmCell" => LSTMCellLayerArgs?args ^-> LSTMCell
                "rnn" => RNNLayerArgs?args ^-> RNN
                "simpleRNN" => SimpleRNNLayerArgs?args ^-> SimpleRNN
                "simpleRNNCell" => SimpleRNNCellLayerArgs?args ^-> SimpleRNNCell
                "stackedRNNCells" => StackedRNNCellsArgs?args ^-> StackedRNNCells

                // Layers / Wrapper
                "bidirectional" => BidirectionalLayerArgs?args ^-> Bidirectional
                "timeDistributed" => WrapperLayerArgs?args ^-> TimeDistributed

                // Layers / Inputs
                "inputLayer" => InputLayerArgs?args ^-> InputLayer

                // Layers / Padding
                "zeroPadding2d" => !?ZeroPadding2DLayerArgs?args ^-> ZeroPadding2D

                // Layers / Noise
                "alphaDropout" => AlphaDropoutArgs?args ^-> AlphaDropout
                "gaussianDropout" => GaussianDropoutArgs?args ^-> GaussianDropout
                "gaussianNoise" => GaussianNoiseArgs?args ^-> GaussianNoise

                // Layers / Mask
                "masking" => MaskingArgs?args ^-> Masking

                // Layers / Rescaling
                "rescaling" => !?RescalingArgs?args ^-> Rescaling

                // Layers / CenterCrop
                "rescaling" => !?CenterCropArgs?args ^-> CenterCrop

                // Layers / Resizing
                "resizing" => !?ResizingArgs?args ^-> Resizing

                // Layers / CategoryEncoding
                "categoryEncoding" => CategoryEncodingArgs?args ^-> CategoryEncoding

                //Layers / RandomWidth
                "randomWidth" => RandomWidthArgs?args ^-> RandomWidth
            ]

    let Assembly =
        Assembly [
            Namespace "WebSharper.TensorFlowJs" [
                
            ]
        ]

[<Sealed>]
type Extension() =
    interface IExtension with
        member ext.Assembly =
            Definition.Assembly

[<assembly: Extension(typeof<Extension>)>]
do ()
