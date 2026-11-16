namespace WebSharper.TensorFlowJs

open WebSharper
open WebSharper.JavaScript
open WebSharper.InterfaceGenerator
open WebSharper.TensorFlowJs.Types
open WebSharper.TensorFlowJs.Enumeration
open WebSharper.TensorFlowJs.Interfaces
open WebSharper.TensorFlowJs.MainClasses

module Definition =
    [<AutoOpen>]
    module TFCore = 
        // A union-typed parameter makes WebSharper emit one overload per
        // constituent, and the product across several union parameters explodes
        // (e.g. `basicLSTMCell` has six of them -> 10^6 overloads). Keep the
        // union to the two common cases - a Tensor or a flat number array - so
        // the generated surface stays small.
        let TensorOrTensorLike = Tensor + T<float[]>

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
            |> Import "*" "@tensorflow/tfjs"
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
                    "registerClass" => SerializableConstructor[t]?cls * !?T<string>?pkg * !?T<string>?name ^-> SerializableConstructor[t]

                // Models / Op Registry
                "deregisterOp" => T<string>?name  ^-> T<unit>
                "getRegisteredOp" => T<string>?name  ^-> OpMapper
                "registerOp" => T<string>?name  * OpExecutor?opFunc ^-> T<unit>

                // Layers / Advanced Activation
                "layers.elu" => !?ELULayerArgs?args ^-> ELU
                |> WithSourceName "Layers_Elu"
                "layers.leakyReLU" => !?LeakyReLULayerArgs?args ^-> LeakyReLU
                |> WithSourceName "Layers_LeakyReLU"
                "layers.prelu" => !?PReLULayerArgs?args ^-> PReLU
                |> WithSourceName "Layers_Prelu"
                "layers.reLU" => !?ReLULayerArgs?args ^-> ReLU
                |> WithSourceName "Layers_ReLU"
                "layers.softmax" => !?SoftmaxLayerArgs?args ^-> Softmax
                |> WithSourceName "Layers_Softmax"
                "layers.thresholdedReLU" => !?ThresholdedReLULayerArgs?args ^-> ThresholdedReLU
                |> WithSourceName "Layers_ThresholdedReLU"

                // Layers / Basic
                "layers.activation" => ActivationLayerArgs?args ^-> Activation
                |> WithSourceName "Layers_Activation"
                "layers.dense" => DenseLayerArgs?args ^-> Dense
                |> WithSourceName "Layers_Dense"
                "layers.dropout" => DropoutLayerArgs?args ^-> Dropout
                |> WithSourceName "Layers_Dropout"
                "layers.embedding" => EmbeddingLayerArgs?args ^-> Embedding
                |> WithSourceName "Layers_Embedding"
                "layers.flatten" => !?FlattenLayerArgs?args ^-> Flatten
                |> WithSourceName "Layers_Flatten"
                "layers.permute" => PermuteLayerArgs?args ^-> Permute
                |> WithSourceName "Layers_Permute"
                "layers.repeatVector" => RepeatVectorLayerArgs?args ^-> RepeatVector
                |> WithSourceName "Layers_RepeatVector"
                "layers.reshape" => ReshapeLayerArgs?args ^-> Reshape
                |> WithSourceName "Layers_Reshape"
                "layers.spatialDropout1d" => SpatialDropout1DLayerConfig?args ^-> SpatialDropout1D
                |> WithSourceName "Layers_SpatialDropout1d"
    
                // Layers / Convolutional
                "layers.conv1d" => ConvLayerArgs?args ^-> Conv1D
                |> WithSourceName "Layers_Conv1d"
                "layers.conv2d" => ConvLayerArgs?args ^-> Conv2D
                |> WithSourceName "Layers_Conv2d"
                "layers.conv2dTranspose" => ConvLayerArgs?args ^-> Conv2DTranspose
                |> WithSourceName "Layers_Conv2dTranspose"
                "layers.conv3d" => ConvLayerArgs?args ^-> Conv3D
                |> WithSourceName "Layers_Conv3d"
                "layers.cropping2D" => Cropping2DLayerArgs?args ^-> Cropping2D
                |> WithSourceName "Layers_Cropping2D"
                "layers.depthwiseConv2d" => DepthwiseConv2DLayerArgs?args ^-> DepthwiseConv2D
                |> WithSourceName "Layers_DepthwiseConv2d"
                "layers.separableConv2d" => SeparableConvLayerArgs?args ^-> SeparableConv2D
                |> WithSourceName "Layers_SeparableConv2d"
                "layers.upSampling2d" => UpSampling2DLayerArgs?args ^-> UpSampling2D
                |> WithSourceName "Layers_UpSampling2d"

                // Layers / Merge
                "layers.add" => !?LayerArgs?args ^-> Add
                |> WithSourceName "Layers_Add"
                "layers.average" => !?LayerArgs?args ^-> Average
                |> WithSourceName "Layers_Average"
                "layers.concatenate" => !?ConcatenateLayerArgs?args ^-> Concatenate
                |> WithSourceName "Layers_Concatenate"
                "layers.maximum" => !?LayerArgs?args ^-> Maximum
                |> WithSourceName "Layers_Maximum"
                "layers.minimum" => !?LayerArgs?args ^-> Minimum
                |> WithSourceName "Layers_Minimum"
                "layers.multiply" => !?LayerArgs?args ^-> Multiply
                |> WithSourceName "Layers_Multiply"

                // Layers / Normalization
                "layers.batchNormalization" => !?BatchNormalizationLayerArgs?args ^-> BatchNormalization
                |> WithSourceName "Layers_BatchNormalization"
                "layers.layerNormalization" => !?LayerNormalizationLayerArgs?args ^-> LayerNormalization
                |> WithSourceName "Layers_LayerNormalization"

                // Layers / Pooling
                "layers.averagePooling1d" => Pooling1DLayerArgs?args ^-> AveragePooling1D 
                |> WithSourceName "Layers_AveragePooling1d"
                "layers.averagePooling2d" => Pooling2DLayerArgs?args ^-> AveragePooling2D 
                |> WithSourceName "Layers_AveragePooling2d"
                "layers.averagePooling3d" => Pooling3DLayerArgs?args ^-> AveragePooling3D 
                |> WithSourceName "Layers_AveragePooling3d"
                "layers.globalAveragePooling1d" => !?LayerArgs?args ^-> GlobalAveragePooling1D 
                |> WithSourceName "Layers_GlobalAveragePooling1d"
                "layers.globalAveragePooling2d" => GlobalPooling2DLayerArgs?args ^-> GlobalAveragePooling2D 
                |> WithSourceName "Layers_GlobalAveragePooling2d"
                "layers.globalMaxPooling1d" => !?LayerArgs?args ^-> GlobalMaxPooling1D 
                |> WithSourceName "Layers_GlobalMaxPooling1d"
                "layers.globalMaxPooling2d" => GlobalPooling2DLayerArgs?args ^-> GlobalMaxPooling2D 
                |> WithSourceName "Layers_GlobalMaxPooling2d"
                "layers.maxPooling1d" => Pooling1DLayerArgs?args ^-> MaxPooling1D 
                |> WithSourceName "Layers_MaxPooling1d"
                "layers.maxPooling2d" => Pooling2DLayerArgs?args ^-> MaxPooling2D 
                |> WithSourceName "Layers_MaxPooling2d"
                "layers.maxPooling3d" => Pooling3DLayerArgs?args ^-> MaxPooling3D 
                |> WithSourceName "Layers_MaxPooling3d"
                
                // Layers / Recurrent
                "layers.convLstm2d" => ConvLSTM2DArgs?args ^-> ConvLSTM2D 
                |> WithSourceName "Layers_ConvLstm2d"
                "layers.convLstm2dCell" => ConvLSTM2DCellArgs?args ^-> ConvLSTM2DCell
                |> WithSourceName "Layers_ConvLstm2dCell"
                "layers.gru" => GRULayerArgs?args ^-> GRU
                |> WithSourceName "Layers_Gru"
                "layers.gruCell" => GRUCellLayerArgs?args ^-> GRUCell
                |> WithSourceName "Layers_GruCell"
                "layers.lstm" => LSTMLayerArgs?args ^-> LSTM
                |> WithSourceName "Layers_Lstm"
                "layers.lstmCell" => LSTMCellLayerArgs?args ^-> LSTMCell
                |> WithSourceName "Layers_LstmCell"
                "layers.rnn" => RNNLayerArgs?args ^-> RNN
                |> WithSourceName "Layers_Rnn"
                "layers.simpleRNN" => SimpleRNNLayerArgs?args ^-> SimpleRNN
                |> WithSourceName "Layers_SimpleRNN"
                "layers.simpleRNNCell" => SimpleRNNCellLayerArgs?args ^-> SimpleRNNCell
                |> WithSourceName "Layers_SimpleRNNCell"
                "layers.stackedRNNCells" => StackedRNNCellsArgs?args ^-> StackedRNNCells
                |> WithSourceName "Layers_StackedRNNCells"

                // Layers / Wrapper
                "layers.bidirectional" => BidirectionalLayerArgs?args ^-> Bidirectional
                |> WithSourceName "Layers_Bidirectional"
                "layers.timeDistributed" => WrapperLayerArgs?args ^-> TimeDistributed
                |> WithSourceName "Layers_TimeDistributed"

                // Layers / Inputs
                "layers.inputLayer" => InputLayerArgs?args ^-> InputLayer
                |> WithSourceName "Layers_InputLayer"

                // Layers / Padding
                "layers.zeroPadding2d" => !?ZeroPadding2DLayerArgs?args ^-> ZeroPadding2D
                |> WithSourceName "Layers_ZeroPadding2d"

                // Layers / Noise
                "layers.alphaDropout" => AlphaDropoutArgs?args ^-> AlphaDropout
                |> WithSourceName "Layers_AlphaDropout"
                "layers.gaussianDropout" => GaussianDropoutArgs?args ^-> GaussianDropout
                |> WithSourceName "Layers_GaussianDropout"
                "layers.gaussianNoise" => GaussianNoiseArgs?args ^-> GaussianNoise
                |> WithSourceName "Layers_GaussianNoise"

                // Layers / Mask
                "layers.masking" => MaskingArgs?args ^-> Masking
                |> WithSourceName "Layers_Masking"

                // Layers / Rescaling
                "layers.rescaling" => !?RescalingArgs?args ^-> Rescaling
                |> WithSourceName "Layers_Rescaling"

                // Layers / CenterCrop
                "layers.centerCrop" => !?CenterCropArgs?args ^-> CenterCrop
                |> WithSourceName "Layers_CenterCrop"

                // Layers / Resizing
                "layers.resizing" => !?ResizingArgs?args ^-> Resizing
                |> WithSourceName "Layers_Resizing"

                // Layers / CategoryEncoding
                "layers.categoryEncoding" => CategoryEncodingArgs?args ^-> CategoryEncoding
                |> WithSourceName "Layers_CategoryEncoding"

                //Layers / RandomWidth
                "layers.randomWidth" => RandomWidthArgs?args ^-> RandomWidth
                |> WithSourceName "Layers_RandomWidth"

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
                Layer
                RNNCell
                Optimizer
                Environment
                Constraint
                Initializer
                Dataset
                CSVDataset             
                Regularizer

                // Enum
                ParamType; Category; WeightGroup; Rank; WebGLChannels; ActivationIdentifier; Reduction

                // Interface
                CustomCallbackArgs; ModelPredictArgs; ModelEvaluateDatasetArgs; IOHandlerSync
                IOHandler; ModelPredictConfig; SaveResult; ModelArtifacts; ModelArtifactsInfo
                SaveConfig; WeightsManifestEntry; Quantization; TrainingConfig; ModelTensorInfo
                WebGPUData; WebGLData; DataToGPUWebGLOption; DataTypeMap; TensorInfo;PrintFnType
                GPUBuffer
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
                ELULayerArgs; Function; ProfileInfo; KernelBackend; InputSpec
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
                StackedRNNCells; SimpleRNNCell; SimpleRNN; ConvLSTM2DCell; LSTM; LSTMCell; GRU; GRUCell
                EarlyStopping; EarlyStoppingCallbackArgs; PixelData; DrawOptions
                ContextOptions; WebGLContextAttributes; ImageOptions; L2Config; L1L2Config
                L1Config; Zeros; VarianceScalingArgs; TruncatedNormalArgs; RandomUniformArgs
                RandomNormalArgs; OrthogonalArgs; LeCunUniformArgs; LeCunNormalArgs
                IdentityArgs; HeUniformArgs; HeNormalArgs; GlorotUniformArgs; Engine
                CustomGradientFuncResult; ValueAndGradsResultReturn; VariableGradsResult
                ValueAndGradResultReturn; MeshgridThridParameter; UniqueResult
                TopkResult; MomentsResult; Platform; LayerVariable; SequentialArgs
                History; BaseCallback; ModelFitDatasetArgs; ModelFitArgs; ModelEvaluateArgs
                ModelCompileArgs; AdamaxOptimizer; ContainerArgs; Node; NodeArgs
                GraphNode; OpMapper; ParamMapper; InputParamMapper; AttrParamMapper
                ValueType; ConvLSTM2D; ConvRNN2D; MaxPooling2D; MaxPooling3D
            ]
        ]

[<Sealed>]
type Extension() =  
    interface IExtension with
        member ext.Assembly =
            Definition.Assembly

[<assembly: Extension(typeof<Extension>)>]
do ()
