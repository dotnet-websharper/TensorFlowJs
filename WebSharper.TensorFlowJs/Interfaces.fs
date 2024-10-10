namespace WebSharper.TensorFlowJs.Interfaces

open WebSharper
open WebSharper.JavaScript
open WebSharper.InterfaceGenerator
open WebSharper.TensorFlowJs.Types
open WebSharper.TensorFlowJs.Enumeration

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

    let DeepMapResult = 
        Pattern.Config "DeepMapResult" {
            Required = [
                "value", T<obj>
                "recurse", T<bool>
            ]
            Optional = []
        }
