namespace WebSharper.TensorFlowJs.Sample

open WebSharper
open WebSharper.JavaScript
open WebSharper.UI
open WebSharper.UI.Client
open WebSharper.UI.Templating
open WebSharper.TensorFlowJs

[<JavaScript>]
module Client =

    // The template is the whole document, loaded from the DOM, so editing
    // index.html and refreshing the browser is enough.
    type IndexTemplate = Template<"index.html", ClientLoad.FromDocument>

    // ---------------------------------------------------------------
    // State
    // ---------------------------------------------------------------

    let notice = Var.Create "TensorFlow.js is loaded from the npm package and bundled with esbuild."
    let themeLabel = Var.Create "🌙 Dark mode"
    let tensorOut = Var.Create "Press “Create” to build a tensor from the values below."
    let modelOut = Var.Create "Press “Train” to fit a dense model to y = 2x + 1."
    let backend = Var.Create "-"
    let numTensors = Var.Create "-"
    let numBytes = Var.Create "-"

    let valuesInput = Var.Create "1, 2, 3, 4"
    let shapeInput = Var.Create "2, 2"
    let predictX = Var.Create "3"

    // Light is the default; the choice is persisted in localStorage. The
    // storage key is shared with the bootstrap script in index.html.
    let themeStorageKey = "websharper-tensorflowjs-theme"
    let theme = Var.Create "light"

    // Tensors the sample owns and can dispose.
    let mutable owned : Tensor list = []

    // ---------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------

    let splitOn (s: string) =
        s.Split([| ','; ';'; ' '; '\t' |], System.StringSplitOptions.RemoveEmptyEntries)

    let parseFloats (s: string) = splitOn s |> Array.map float
    let parseInts (s: string) = splitOn s |> Array.map int

    // `Tensor.arraySync()` returns nested JS arrays whose shape depends on the
    // tensor's rank, while `dataSync()` always returns a flat typed array.
    [<Inline "$0.dataSync()[0]">]
    let firstValue (t: Tensor) : float = X<float>

    let describe (name: string) (t: Tensor) =
        sprintf "%s: shape [%s]  dtype %s\n%s"
            name
            (t.Shape |> Array.map string |> String.concat ", ")
            t.Dtype
            (t.ToString())

    let track (t: Tensor) =
        owned <- t :: owned
        t

    let disposeTensors () =
        owned |> List.iter (fun t -> Tf.Dispose(t))
        owned <- []

    let applyTheme () =
        JS.Document.DocumentElement.SetAttribute("data-theme", theme.Value)
        themeLabel.Value <-
            if theme.Value = "dark" then "☀️ Light mode" else "🌙 Dark mode"

    let toggleTheme () =
        theme.Value <- (if theme.Value = "dark" then "light" else "dark")
        try JS.Window.LocalStorage.SetItem(themeStorageKey, theme.Value) with _ -> ()
        applyTheme ()

    let refreshRuntime () =
        backend.Value <- Tf.GetBackend()
        let mem = Tf.Memory()
        numTensors.Value <- string mem.NumTensors
        numBytes.Value <- string mem.NumBytes

    // ---------------------------------------------------------------
    // Tensors & operations
    // ---------------------------------------------------------------

    let mutable a : option<Tensor> = None
    let mutable b : option<Tensor> = None

    let create () =
        try
            let values = parseFloats valuesInput.Value
            let shape = parseInts shapeInput.Value
            disposeTensors ()
            a <- Some (track (Tf.Tensor2d(values, shape)))
            b <- Some (track (Tf.Ones(shape)))
            match a, b with
            | Some ta, Some tb ->
                tensorOut.Value <- describe "a" ta + "\n" + describe "b (ones)" tb
            | _ -> ()
            refreshRuntime ()
        with e ->
            tensorOut.Value <- "Could not create the tensor: " + e.Message

    let withAB (name: string) (op: Tensor -> Tensor -> Tensor) =
        match a, b with
        | Some ta, Some tb ->
            let r = track (op ta tb)
            tensorOut.Value <- name + "\n" + describe "result" r
            refreshRuntime ()
        | _ ->
            tensorOut.Value <- "Press “Create” first."

    let withA (name: string) (op: Tensor -> Tensor) =
        match a with
        | Some ta ->
            let r = track (op ta)
            tensorOut.Value <- name + "\n" + describe "result" r
            refreshRuntime ()
        | _ ->
            tensorOut.Value <- "Press “Create” first."

    let dispose () =
        disposeTensors ()
        a <- None
        b <- None
        tensorOut.Value <- "Disposed every tensor created by the sample."
        refreshRuntime ()

    // ---------------------------------------------------------------
    // Training a tiny model
    // ---------------------------------------------------------------

    let mutable model : option<Sequential> = None

    let train () =
        try
            let n = 128
            let xs = Array.init n (fun i -> float i / float (n - 1) * 5.0)
            let ys = xs |> Array.map (fun x -> 2.0 * x + 1.0)
            let x = track (Tf.Tensor2d(xs, [| n; 1 |]))
            let y = track (Tf.Tensor2d(ys, [| n; 1 |]))

            let m = Tf.Sequential()
            m.Add(Tf.Layers_Dense(DenseLayerArgs(8, Activation = ActivationIdentifier.Relu, InputShape = Optional.ofOption (Some [| 1 |]))))
            m.Add(Tf.Layers_Dense(DenseLayerArgs(1)))
            m.Compile(ModelCompileArgs("adam", "meanSquaredError", "mse"))
            model <- Some m

            modelOut.Value <- "Training…"
            m.Fit(x, y, ModelFitArgs(16, 200, 0))
                .Then(fun history ->
                    let raw : obj = history.History?loss
                    let losses = raw :?> float[]
                    let last = if losses.Length = 0 then nan else losses.[losses.Length - 1]
                    modelOut.Value <-
                        sprintf "Trained 200 epochs on y = 2x + 1 (x in [0, 5]).\nfinal loss = %.5f\nnow predict a value." last
                    refreshRuntime ())
            |> ignore
        with e ->
            modelOut.Value <- "Training failed: " + e.Message

    let predict () =
        match model with
        | None ->
            modelOut.Value <- "Train the model first."
        | Some m ->
            try
                let x = float predictX.Value
                let input = track (Tf.Tensor2d([| x |], [| 1; 1 |]))
                let pred : Tensor = m.Predict(input)
                let value = firstValue pred
                modelOut.Value <-
                    sprintf "x = %.4f\npredicted y = %.5f\nexpected  y = %.5f\nerror = %.5f"
                        x value (2.0 * x + 1.0) (abs (value - (2.0 * x + 1.0)))
                refreshRuntime ()
            with e ->
                modelOut.Value <- "Prediction failed: " + e.Message

    // ---------------------------------------------------------------
    // Entry point
    // ---------------------------------------------------------------

    [<SPAEntryPoint>]
    let Main () =
        // Restore a previously chosen theme (light is the default).
        let savedTheme =
            try JS.Window.LocalStorage.GetItem(themeStorageKey) with _ -> null
        if savedTheme = "dark" || savedTheme = "light" then
            theme.Value <- savedTheme
        applyTheme ()

        // `.Bind()` renders the whole document body, so the DOM exists only
        // after this call.
        IndexTemplate()
            .notice(notice.V)
            .themeLabel(themeLabel.V)
            .tensorOut(tensorOut.V)
            .modelOut(modelOut.V)
            .backend(backend.V)
            .numTensors(numTensors.V)
            .numBytes(numBytes.V)
            .Values(valuesInput)
            .Shape(shapeInput)
            .PredictX(predictX)
            .ToggleTheme(fun _ -> toggleTheme ())
            .Create(fun _ -> create ())
            .Add(fun _ -> withAB "a + b" (fun x y -> Tf.Add(x, y)))
            .Mul(fun _ -> withAB "a * b" (fun x y -> Tf.Mul(x, y)))
            .MatMul(fun _ -> withAB "a · b" (fun x y -> Tf.MatMul(x, y)))
            .Relu(fun _ -> withA "relu(a)" Tf.Relu)
            .Softmax(fun _ -> withA "softmax(a)" Tf.Softmax)
            .Sum(fun _ -> withA "sum(a)" (fun t -> Tf.Sum(t)))
            .Mean(fun _ -> withA "mean(a)" (fun t -> Tf.Mean(t)))
            .Dispose(fun _ -> dispose ())
            .Train(fun _ -> train ())
            .Predict(fun _ -> predict ())
            .RefreshRuntime(fun _ -> refreshRuntime ())
            .Bind()

        refreshRuntime ()
