using System.Diagnostics;
using NN;
using NN.Mnist;

// Handwritten digit recognition on MNIST: 28x28 greyscale images of digits 0-9, the standard
// first real dataset. XOR has 4 examples and 2 inputs; this has 60,000 and 784.
//
//   dotnet run -c Release --project src/NN.Mnist                    train (or reuse) and evaluate
//   dotnet run -c Release --project src/NN.Mnist -- --epochs 5      quicker
//   dotnet run -c Release --project src/NN.Mnist -- --retrain       ignore the saved model
//   dotnet run -c Release --project src/NN.Mnist -- --predict 42    classify one test image
//   dotnet run -c Release --project src/NN.Mnist -- --image d.png   classify your own image
//   dotnet run -c Release --project src/NN.Mnist -- --loss mse      the older, worse setup

int epochs = ArgValue("--epochs", 20);
int hidden = ArgValue("--hidden", 128);
int trainLimit = ArgValue("--train", 0);   // 0 = all 60,000
int predictIndex = ArgValue("--predict", -1);
bool retrain = HasFlag("--retrain");
string? imagePath = ArgPath("--image");

// Softmax cross-entropy by default; --loss mse reproduces the older, worse setup for comparison.
bool useMse = string.Equals(ArgPath("--loss"), "mse", StringComparison.OrdinalIgnoreCase);
string lossName = useMse ? "mse" : "softmax";

// MSE over sigmoid outputs needs an enormous step size to compensate for a gradient two shrinking
// factors have already flattened; cross-entropy needs no such compensation. See below.
float learningRate = useMse ? 1.0f : 0.1f;

// Both the architecture and the training-set size go in the filename, because a saved model is
// only interchangeable with one trained the same way. ModelIO records the architecture and would
// reload a 256-unit model perfectly happily — and a model trained on 5000 images would reload
// with no complaint at all, since nothing about the file says how much data it saw. Neither
// mismatch is an error; both are silently *wrong results*, which is harder to notice.
string modelPath = ArgPath("--model") ?? Path.Combine(
    MnistData.CacheDirectory,
    trainLimit > 0 ? $"mnist-{hidden}-{lossName}-{trainLimit}.nnm" : $"mnist-{hidden}-{lossName}.nnm");

Console.WriteLine("Handwritten digit recognition — MNIST\n");

// ── Reading a digit out of an image file.
//
//    This path deliberately runs before anything touches MNIST: classifying your own image needs
//    the 397 KB of trained weights and nothing else. No dataset, no download, no training. That
//    is what a saved model is for, and the repository ships one so this works on a fresh clone. ──
if (imagePath is not null)
    return ReadDigitFromImage(imagePath, modelPath, hidden);

MnistData.Split? data = MnistData.TryLoad(trainLimit, testLimit: 0, log: Console.WriteLine);

if (data is null)
{
    // The offline path. Everything above this point still printed, so the reader knows what the
    // demo would have done and exactly why it didn't.
    Console.WriteLine($"""

        MNIST is not cached and could not be downloaded.

        This demo needs the dataset (~11 MB), which the repository deliberately does not
        ship — see src/NN.Mnist/MnistData.cs. Run it once with a network connection and it
        will cache to:

            {MnistData.CacheDirectory}

        Every later run, including offline ones, then works from that cache.

        Nothing is broken: the other demo (dotnet run --project src/NN.Demo) needs no data
        at all and covers backpropagation, XOR, and the two-moons dataset.
        """);

    return 0;
}

Console.WriteLine($"\n{data.TrainCount:N0} training images, {data.TestCount:N0} test images, {Idx.PixelCount} pixels each\n");

// One image, so it's clear what the network is actually reading: 784 floats in 0-1.
Console.WriteLine($"A training example (label {data.TrainDigits[0]}):\n");
Console.Write(Render(data.TrainX, 0));

// ── Train, or reuse a model trained by an earlier run.
//
//    Training is the expensive step and its result is just 101,770 numbers. Once they are on
//    disk, everything below — evaluation, the confusion matrix, single-image prediction — is
//    inference against a file, which is exactly how a trained model gets used in practice.
//    XOR's 17 parameters make the same point but are too small to make it convincingly. ──
Network net;
bool trained;
var clock = Stopwatch.StartNew();

if (!retrain && File.Exists(modelPath))
{
    net = ModelIO.Load(modelPath);
    trained = false;

    Console.WriteLine($"""

        Loaded a trained model — no training needed.

            {modelPath}
            {new FileInfo(modelPath).Length / 1024:N0} KB, {net.ParameterCount:N0} parameters, loaded in {clock.Elapsed.TotalMilliseconds:F0} ms

        The architecture came out of the file too: ModelIO stores each layer's type and shape
        alongside its weights, so nothing here had to know it was a 784-{hidden}-10 network.
        Pass --retrain to train from scratch instead.
        """);

    Console.WriteLine();
    Console.Write(net.Summary());
}
else
{
    // Softmax + cross-entropy is the standard classification setup; the ten-independent-sigmoids
    // version is kept behind --loss mse purely so the two can be compared on equal terms.
    net = useMse
        ? new Sequential(inputs: Idx.PixelCount).Dense<Tanh>(hidden).Dense<Sigmoid>(10).Build(seed: 42)
        : new Sequential(inputs: Idx.PixelCount).Dense<Tanh>(hidden).SoftmaxOutput(10).Build(seed: 42);

    trained = true;

    Console.WriteLine();
    Console.Write(net.Summary());

    Console.WriteLine(useMse
        ? $"""

            Ten independent sigmoid outputs scored by mean squared error.
            Training {epochs} epochs, batch size 32, learning rate {learningRate}.

            That learning rate is enormous — the study guide suggests 0.1 to 0.5 — and it is
            compensating for a gradient that MSE-over-sigmoid has already flattened twice over:
            dL/da = 2(a-y)/10 divides by the ten outputs, and sigmoid's own derivative a(1-a)
            peaks at 0.25 and collapses toward 0 as outputs saturate — that is, exactly when the
            network is confidently wrong and most needs to learn.

            Run without --loss mse to see what removing that handicap is worth.

            """
        : $"""

            Softmax output with cross-entropy loss — the standard way to choose among mutually
            exclusive categories. Training {epochs} epochs, batch size 32, learning rate {learningRate}.

            Softmax turns the last layer's raw scores into a probability distribution that sums
            to 1, so the ten digits compete rather than each answering yes/no independently.
            Cross-entropy then scores only the probability given to the right answer.

            The payoff is in the gradient. Differentiated separately, softmax gives a full
            Jacobian and cross-entropy gives a 1/p that explodes; composed, almost everything
            cancels and what is left is just  p - y  — prediction minus target. No vanishing
            factor, so no need for the learning rate of 1.0 that --loss mse requires.

            """);

    Console.WriteLine("  epoch   train loss   test acc      elapsed");

    clock.Restart();

    net.Train(data.TrainX, data.TrainY, epochs: epochs, learningRate: learningRate, batchSize: 32,
        onEpoch: (epoch, loss) =>
        {
            if (epoch % 5 != 0 && epoch != 1) return;

            Console.WriteLine($"  {epoch,5}   {loss,10:F5}   {Accuracy(net!, data.TestX, data.TestDigits),7:P2}   {clock.Elapsed.TotalSeconds,8:F1}s");
        });

    Console.WriteLine($"\n  Trained in {clock.Elapsed.TotalSeconds:F1}s.");

    SaveAndVerify(net, modelPath, data);
}

float trainAccuracy = Accuracy(net, data.TrainX, data.TrainDigits);
float testAccuracy = Accuracy(net, data.TestX, data.TestDigits);

Console.WriteLine($"\n  Train accuracy {trainAccuracy:P2}, test accuracy {testAccuracy:P2}");

// ── Using the model: classify one image on demand. This is what a saved model is *for* — the
//    training data is not even needed for this path, only the 407 KB of weights. ──
if (predictIndex >= 0 && predictIndex < data.TestCount)
{
    ReadOnlySpan<float> output = net.Predict(data.TestX.AsSpan(predictIndex * Idx.PixelCount, Idx.PixelCount));
    int predicted = Classify(net, data.TestX, predictIndex);

    Console.WriteLine($"\n─────────────────────────────────────────────");
    Console.WriteLine($"Predicting test image {predictIndex}:\n");
    Console.Write(Render(data.TestX, predictIndex));

    Console.WriteLine($"\n  This is a {data.TestDigits[predictIndex]}. The network says {predicted}" +
                      $" — {(predicted == data.TestDigits[predictIndex] ? "correct" : "wrong")}.\n");

    // All ten outputs, not just the winner: the runners-up show what it nearly said, which is
    // where a confident mistake and a lucky guess start to look very different.
    for (int d = 0; d < 10; d++)
    {
        int bar = (int)(output[d] * 40);
        Console.WriteLine($"    {d}  {output[d]:F3}  {new string('█', bar)}");
    }

    return 0;
}

// ── Confusion matrix: which digits does it mix up? A single accuracy number hides the structure,
//    and the structure is the interesting part — the mistakes are not uniformly distributed. ──
var confusion = new int[10, 10];

for (int i = 0; i < data.TestCount; i++)
    confusion[data.TestDigits[i], Classify(net, data.TestX, i)]++;

Console.WriteLine("\nConfusion matrix — rows are the true digit, columns the prediction:\n");
Console.WriteLine("        " + string.Join("", Enumerable.Range(0, 10).Select(d => $"{d,6}")) + "    accuracy");

for (int actual = 0; actual < 10; actual++)
{
    int total = 0, correct = confusion[actual, actual];
    for (int predicted = 0; predicted < 10; predicted++) total += confusion[actual, predicted];

    var row = new System.Text.StringBuilder($"    {actual}   ");

    for (int predicted = 0; predicted < 10; predicted++)
    {
        int n = confusion[actual, predicted];
        // Dot for zero: an empty cell reads as "nothing here" far faster than a 0 does.
        row.Append(n == 0 ? "     ·" : $"{n,6}");
    }

    Console.WriteLine($"{row}    {(float)correct / total,7:P1}");
}

(int worstActual, int worstPredicted, int worstCount) = WorstConfusion(confusion);
Console.WriteLine($"\n  Most common mistake: a real {worstActual} called a {worstPredicted}, {worstCount} times.");

// ── The mistakes themselves. This is the payoff: most of them are digits a human would also
//    hesitate over, which is a more honest picture of "96% accurate" than the number alone. ──
Console.WriteLine("\nSome of its mistakes:\n");

int shown = 0;

for (int i = 0; i < data.TestCount && shown < 4; i++)
{
    int predicted = Classify(net, data.TestX, i);
    if (predicted == data.TestDigits[i]) continue;

    ReadOnlySpan<float> output = net.Predict(data.TestX.AsSpan(i * Idx.PixelCount, Idx.PixelCount));

    Console.WriteLine($"  test image {i}: this is a {data.TestDigits[i]}, the network says {predicted} " +
                      $"(confidence {output[predicted]:F2}, and {output[data.TestDigits[i]]:F2} for the right answer)");
    Console.Write(Render(data.TestX, i));
    Console.WriteLine();

    shown++;
}

Console.WriteLine($"""
    ─────────────────────────────────────────────
    {testAccuracy:P2} on digits it has never seen, from a network written with no ML library:
    two dense layers, backpropagation, and mini-batch gradient descent.

    {(useMse
        ? """
        That is short of the ~98% this architecture normally reaches, and the reason is the
        loss rather than anything mysterious: MSE treats the digits as ten unrelated yes/no
        questions instead of one ten-way choice, and its gradient shrinks exactly when the
        network is confidently wrong. The learning rate of 1.0 is that weakness made visible.

        Re-run without --loss mse to see the difference. It is worth about +0.6 points, at a
        tenth of the learning rate.
        """
        : """
        That is par for this architecture, and it is what softmax with cross-entropy buys:
        the same 784-128-10 network scored by MSE reaches only ~97.4%, and needs a learning
        rate of 1.0 to get there rather than the 0.1 used here (try --loss mse).

        The remaining headroom is study guide §25 item 2 — plain SGD, no momentum or Adam,
        so every step is the same size regardless of the terrain. That is exercise 10.
        """)}
    """);

Console.WriteLine(trained
    ? $"""

        The trained model is now on disk, so re-running this skips straight to inference:

            dotnet run -c Release --project src/NN.Mnist                  reuse it
            dotnet run -c Release --project src/NN.Mnist -- --predict 42  classify one image
            dotnet run -c Release --project src/NN.Mnist -- --retrain     start over
        """
    : $"""

        Everything above ran against the saved model — no training. Try:

            dotnet run -c Release --project src/NN.Mnist -- --predict 42  classify one image
            dotnet run -c Release --project src/NN.Mnist -- --retrain     train from scratch
        """);

return 0;

// ── Helpers ──

/// <summary>
/// Classifies a digit in an image file, using only a trained model — no dataset involved.
/// </summary>
static int ReadDigitFromImage(string imagePath, string preferredModel, int hidden)
{
    if (!File.Exists(imagePath))
    {
        Console.Error.WriteLine($"No such image: {imagePath}");
        return 1;
    }

    // Prefer a model this machine trained; fall back to the one checked into the repository.
    string shipped = Path.Combine(AppContext.BaseDirectory, "mnist-784-128-10.nnm");
    string model = File.Exists(preferredModel) ? preferredModel : shipped;

    if (!File.Exists(model))
    {
        Console.Error.WriteLine($"No trained model found at {preferredModel} or {shipped}.");
        return 1;
    }

    Network net = ModelIO.Load(model);
    Console.WriteLine($"Model:  {model}\nImage:  {imagePath}\n");

    GreyImage image;

    try
    {
        image = ImageFile.Load(imagePath);
    }
    catch (Exception e) when (e is NotSupportedException or InvalidDataException)
    {
        Console.Error.WriteLine($"Could not read the image: {e.Message}");
        return 1;
    }

    // The whole point of DigitPreprocessor: a network trained on MNIST has learned MNIST's
    // conventions, not "digits". See its class comment for what those are and why.
    float[] pixels = DigitPreprocessor.ToMnist(image);

    Console.WriteLine($"  {image.Width}x{image.Height} image, normalized to MNIST's 28x28 convention:\n");
    Console.Write(RenderPixels(pixels));

    ReadOnlySpan<float> output = net.Predict(pixels);

    int best = 0, runnerUp = -1;
    for (int d = 1; d < output.Length; d++)
        if (output[d] > output[best]) best = d;

    for (int d = 0; d < output.Length; d++)
        if (d != best && (runnerUp < 0 || output[d] > output[runnerUp])) runnerUp = d;

    Console.WriteLine($"\n  This is a {best}.  (confidence {output[best]:F3})\n");

    for (int d = 0; d < 10; d++)
        Console.WriteLine($"    {d}  {output[d]:F3}  {new string('█', (int)(output[d] * 40))}");

    // A low winner, or a close second, usually means the preprocessing went wrong rather than
    // that the network is confused — an off-centre or inverted digit produces exactly this.
    if (output[best] < 0.5f || output[runnerUp] > output[best] * 0.5f)
        Console.WriteLine($"""

              Not a confident answer ({best} at {output[best]:F2}, then {runnerUp} at {output[runnerUp]:F2}).
              Check the 28x28 rendering above: the digit should be white on black, filling most
              of the frame. If it looks inverted or tiny, the input needed different preparation
              rather than the network needing more training.
            """);

    return 0;
}

/// <summary>Draws an already-normalized 784-float MNIST image.</summary>
static string RenderPixels(float[] pixels)
{
    const string Ramp = " .:-=+*#%@";

    var sb = new System.Text.StringBuilder();

    for (int row = 0; row < Idx.ImageSize; row++)
    {
        sb.Append("    ");

        for (int column = 0; column < Idx.ImageSize; column++)
        {
            char c = Ramp[Math.Clamp((int)(pixels[row * Idx.ImageSize + column] * Ramp.Length), 0, Ramp.Length - 1)];
            sb.Append(c).Append(c);
        }

        sb.AppendLine();
    }

    return sb.ToString();
}

/// <summary>
/// Writes the trained model, reloads it, and proves the reload is exact.
///
/// <para>The check is not ceremony. A serializer that drops or reorders parameters produces a
/// model that still loads, still predicts, and is merely <i>worse</i> — the same class of silent
/// failure as a wrong gradient. Comparing predictions bit-for-bit across the round trip is the
/// cheapest way to know it didn't happen, and it is how the round-trip regression test in the
/// suite is written too.</para>
/// </summary>
static void SaveAndVerify(Network net, string path, MnistData.Split data)
{
    Directory.CreateDirectory(Path.GetDirectoryName(path)!);
    ModelIO.Save(net, path);

    long bytes = new FileInfo(path).Length;

    // 4 bytes per float, plus a small header naming each layer's type and shape.
    Console.WriteLine($"  Saved to {path}");
    Console.WriteLine($"    {bytes / 1024:N0} KB for {net.ParameterCount:N0} parameters " +
                      $"({(float)bytes / net.ParameterCount:F1} bytes each — float32 plus a header)");

    Network reloaded = ModelIO.Load(path);

    int compared = Math.Min(1000, data.TestCount);
    int identical = 0;

    for (int i = 0; i < compared; i++)
    {
        ReadOnlySpan<float> a = net.Predict(data.TestX.AsSpan(i * Idx.PixelCount, Idx.PixelCount));
        float[] before = a.ToArray();   // Predict lends a buffer the next call overwrites.

        ReadOnlySpan<float> after = reloaded.Predict(data.TestX.AsSpan(i * Idx.PixelCount, Idx.PixelCount));

        if (before.AsSpan().SequenceEqual(after)) identical++;
    }

    Console.WriteLine(identical == compared
        ? $"    Reloaded and verified: all {compared:N0} sampled predictions are bit-for-bit identical."
        : $"    MISMATCH — {compared - identical} of {compared} predictions changed across the round trip.");
}

/// <summary>Index of the largest output — the digit the network is most confident about.</summary>
static int Classify(Network network, float[] images, int index)
{
    ReadOnlySpan<float> output = network.Predict(images.AsSpan(index * Idx.PixelCount, Idx.PixelCount));

    int best = 0;
    for (int d = 1; d < output.Length; d++)
        if (output[d] > output[best]) best = d;

    return best;
}

static float Accuracy(Network network, float[] images, byte[] digits)
{
    int correct = 0;

    for (int i = 0; i < digits.Length; i++)
        if (Classify(network, images, i) == digits[i]) correct++;

    return (float)correct / digits.Length;
}

/// <summary>
/// Draws one image as ASCII. Two characters per pixel, because terminal cells are about twice as
/// tall as they are wide and a 28x28 grid drawn one-for-one comes out squashed.
/// </summary>
static string Render(float[] images, int index)
{
    const string Ramp = " .:-=+*#%@";

    var sb = new System.Text.StringBuilder();
    ReadOnlySpan<float> image = images.AsSpan(index * Idx.PixelCount, Idx.PixelCount);

    for (int row = 0; row < Idx.ImageSize; row++)
    {
        sb.Append("    ");

        for (int column = 0; column < Idx.ImageSize; column++)
        {
            float v = image[row * Idx.ImageSize + column];
            char c = Ramp[Math.Clamp((int)(v * Ramp.Length), 0, Ramp.Length - 1)];

            sb.Append(c).Append(c);
        }

        sb.AppendLine();
    }

    return sb.ToString();
}

static (int Actual, int Predicted, int Count) WorstConfusion(int[,] confusion)
{
    int actual = 0, predicted = 0, count = 0;

    for (int a = 0; a < 10; a++)
        for (int p = 0; p < 10; p++)
            if (a != p && confusion[a, p] > count)
                (actual, predicted, count) = (a, p, confusion[a, p]);

    return (actual, predicted, count);
}

int ArgValue(string name, int fallback)
{
    string[] argv = Environment.GetCommandLineArgs();

    for (int i = 0; i < argv.Length - 1; i++)
        if (argv[i] == name && int.TryParse(argv[i + 1], out int value))
            return value;

    return fallback;
}

string? ArgPath(string name)
{
    string[] argv = Environment.GetCommandLineArgs();

    for (int i = 0; i < argv.Length - 1; i++)
        if (argv[i] == name)
            return argv[i + 1];

    return null;
}

bool HasFlag(string name) => Environment.GetCommandLineArgs().Contains(name);
