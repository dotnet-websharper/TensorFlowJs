namespace WebSharper.TensorFlowJs.Enumeration

open WebSharper.InterfaceGenerator

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
