namespace WebSharper.TensorFlowJs.MainClasses

open WebSharper
open WebSharper.JavaScript
open WebSharper.InterfaceGenerator
open WebSharper.TensorFlowJs.Types
open WebSharper.TensorFlowJs.Enumeration
open WebSharper.TensorFlowJs.Interfaces

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

    let Dataset =
        Generic - fun t ->
            let tToBoolFunc = t?value ^-> T<bool>
            let tToUnitFunc = t?input ^-> T<unit>

            Class "tf.Data.Dataset"
            |+> Instance [
                "size" =? T<int>

                "iterator" => T<unit> ^-> T<Promise<_>>[LazyIterator[t]] 
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
                "toArrayForTest" => T<unit> ^-> T<unit>
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
        |+> Static [
            Constructor (DataSource?input * !?CSVConfig?csvConfig)
        ]
        |+> Instance [
            "columnNames" => T<unit> ^-> T<unit>
            "constructor" => DataSource?input * CSVConfig?csvConfig ^-> T<unit>
            "iterator" => T<unit> ^-> T<Promise<_>>[LazyIterator[TensorContainer]]
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
        |+> Static [
            Constructor (DataType?dtype * Shape?shape * Layer?sourceLayer * (!|TSelf)?inputs * Kwargs?callArgs * !?T<string>?name * T<int>?outputTensorIndex)
        ] 
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
            "id" =? T<int> // read-only(=?) property(=@)

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

    let SymbolicTensorOrSymbolicTensorArray = SymbolicTensor + !|SymbolicTensor

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
    let CallHook = TensorOrTensorArray?inputs * Kwargs?kwargs ^-> T<unit>

    Layer
    |=> Inherits Serializable
    |+> Static [
        Constructor (LayerArgs?args ^-> T<obj>)
        "nodeKey" => Layer?layer * T<int>?nodeIndex ^-> T<unit>
    ]
    |+> Instance [
        "name" =@ T<string> 
        "inputSpec" =@ !|InputSpec 
        "supportsMasking" =@ T<bool> 
        "trainable_" =@ T<bool> 
        "batchInputShape" =@ Shape 
        "dtype" =@ DataType 
        "initialWeights" =@ !|Tensor 
        "inboundNodes" =@ !|Node 
        "outboundNodes" =@ !|Node 
        "activityRegularizer" =@ Regularizer 
        "_trainableWeights" =@ !|LayerVariable 
        "_nonTrainableWeights" =@ !|LayerVariable 
        "_losses" =@ !|RegularizerFn 
        "_updates" =@ !|Tensor 
        "_built" =@ T<bool> 
        "_callHook" =@ CallHook 
        "_addedWeightNames" =@ !|T<string> 
        "id" =? T<int> 
        "_stateful" =@ T<bool> 
        "_refCount" =@ T<int> + T<unit>

        "getInputAt" => T<int>?nodeIndex ^-> (SymbolicTensor + !|SymbolicTensor)
        "getOutputAt" => T<int>?nodeIndex ^-> (SymbolicTensor + !|SymbolicTensor)
        "calculateLosses" => T<unit> ^-> !|Tensor
        "resetStates" => T<unit> ^-> T<unit> 
        "assertInputCompatibility" => (TensorOrTensorArray + SymbolicTensorOrSymbolicTensorArray)?inputs ^-> T<unit>
        "call" => (TensorOrTensorArray)?inputs * Kwargs?kwargs ^-> (TensorOrTensorArray)
        "invokeCallHook" => (TensorOrTensorArray)?inputs * Kwargs?kwargs ^-> T<unit>
        "setCallHook" => CallHook?callHook ^-> T<unit>
        "clearCallHook" => T<unit> ^-> T<unit>
        "apply" => (TensorOrTensorArray + SymbolicTensorOrSymbolicTensorArray)?inputs * !?Kwargs?kwargs ^-> TensorOrTensorArray + SymbolicTensorOrSymbolicTensorArray
        "countParams" => T<unit> ^-> T<int> 
        "build" => ShapeOrArray?inputShape ^-> T<unit> 
        "warnOnIncompatibleInputShape" => Shape?inputShape ^-> T<unit>
        "setFastWeightInitDuringBuild" => T<bool>?value ^-> T<unit>
        "computeMask" => (TensorOrTensorArray)?inputs * !?TensorOrTensorArray?mask ^-> Tensor
        "disposeWeights" => T<unit> ^-> T<int>
        "assertNotDisposed" => T<unit> ^-> T<unit>
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
