using NN;

// ── 1. Perceptron on AND (linearly separable, so the perceptron rule converges) ──
float[] andX = [0, 0,
                0, 1,
                1, 0,
                1, 1];
float[] andY = [0, 0, 0, 1];

var p = new Perceptron(inputs: 2);
int epochs = p.Train(andX, andY, epochs: 100, learningRate: 0.1f);

Console.WriteLine($"Perceptron on AND: converged in {epochs} epochs");
for (int i = 0; i < andY.Length; i++)
{
    var xi = andX.AsSpan(i * 2, 2);
    Console.WriteLine($"  {xi[0]} AND {xi[1]} -> {p.Predict(xi)} (expected {andY[i]})");
}

// ── 2. Backprop on XOR — not linearly separable, so a single perceptron cannot do it.
//      A hidden layer plus backprop can.
float[] xorX = [0, 0,
                0, 1,
                1, 0,
                1, 1];
float[] xorY = [0, 1, 1, 0];

// Sequential builder: state the input width once, and each layer infers its input count
// from the layer before it.
var net = new Sequential(inputs: 2)
    .Dense<Tanh>(4)
    .Dense<Sigmoid>(1)
    .Build(seed: 42);

Console.WriteLine();
Console.Write(net.Summary());

Console.WriteLine("\nNetwork on XOR (2 -> 4 tanh -> 1 sigmoid):");
net.Train(xorX, xorY, epochs: 4000, learningRate: 0.5f, onEpoch: (epoch, loss) =>
{
    if (epoch % 1000 == 0) Console.WriteLine($"  epoch {epoch,5}  loss {loss:F6}");
});

for (int i = 0; i < xorY.Length; i++)
{
    var xi = xorX.AsSpan(i * 2, 2);
    Console.WriteLine($"  {xi[0]} XOR {xi[1]} -> {net.Predict(xi)[0]:F4} (expected {xorY[i]})");
}

// ── 3. Save the trained model, load it back, and confirm it predicts identically. ──
string modelPath = Path.Combine(AppContext.BaseDirectory, "xor.nnm");
ModelIO.Save(net, modelPath);

var loaded = ModelIO.Load(modelPath);

Console.WriteLine($"\nSaved to {Path.GetFileName(modelPath)} ({new FileInfo(modelPath).Length} bytes), reloaded:");

bool identical = true;
for (int i = 0; i < xorY.Length; i++)
{
    var xi = xorX.AsSpan(i * 2, 2);
    float before = net.Predict(xi)[0];
    float after = loaded.Predict(xi)[0];
    if (before != after) identical = false;

    Console.WriteLine($"  {xi[0]} XOR {xi[1]} -> {after:F4}");
}
Console.WriteLine(identical
    ? "  round-trip is bit-for-bit identical"
    : "  MISMATCH — the loaded model differs from the trained one");

// ── 4. Gradient check: compare backprop's analytic gradient against a finite-difference
//      estimate. This is the standard way to prove a backward pass is correct.
float error = GradientCheck.MaxRelativeError(net, xorX.AsSpan(0, 2), xorY.AsSpan(0, 1));
Console.WriteLine($"\nGradient check: max relative error = {error:E3}");

// ── 5. A real classification problem: two interleaving noisy crescents.
//
//      XOR cannot demonstrate the three things that dominate practical training, because it has
//      four noise-free examples and no held-out data:
//        * mini-batching   — four examples is one full batch
//        * generalization  — the four points *are* the problem; there is nothing to generalize to
//        * overfitting     — nothing to memorize
//      This section shows all three. ──
Console.WriteLine("\n─────────────────────────────────────────────");
Console.WriteLine("Two moons: 1500 noisy points, 1000 train / 500 test\n");

(float[] moonX, float[] moonY) = Datasets.Moons(count: 1500, noise: 0.2f, seed: 0);

// Classes alternate in the generated data, so a positional split stays balanced.
const int TrainCount = 1000;
float[] trainX = moonX[..(TrainCount * 2)];
float[] trainY = moonY[..TrainCount];
float[] testX = moonX[(TrainCount * 2)..];
float[] testY = moonY[TrainCount..];

var moons = new Sequential(inputs: 2)
    .Dense<Tanh>(16)
    .Dense<Tanh>(16)
    .Dense<Sigmoid>(1)
    .Build(seed: 7);

Console.Write(moons.Summary());

// Mini-batches of 32, so each epoch performs ~31 updates rather than one. Compare XOR above,
// which runs full batch and so needs 4000 epochs to make 4000 updates.
Console.WriteLine("\nTraining (batch size 32, learning rate 0.3):");
Console.WriteLine("  epoch   train loss   train acc   test acc");

moons.Train(trainX, trainY, epochs: 150, learningRate: 0.3f, batchSize: 32,
    onEpoch: (epoch, loss) =>
    {
        if (epoch % 30 != 0 && epoch != 1) return;

        // Safe mid-training: Predict is inference-only and disturbs no gradient state.
        Console.WriteLine($"  {epoch,5}   {loss,10:F4}   {Accuracy(moons, trainX, trainY),8:P1}   {Accuracy(moons, testX, testY),7:P1}");
    });

Console.WriteLine($"\n  Final: train {Accuracy(moons, trainX, trainY):P1}, test {Accuracy(moons, testX, testY):P1}");
Console.WriteLine("  Test accuracy tracking train accuracy is what generalization looks like.");
Console.WriteLine("  Neither reaches 100%: at this noise level the two moons genuinely overlap.");

Console.WriteLine("\nLearned decision boundary (· = class 0, # = class 1):\n");
Console.Write(PlotBoundary(moons));

// ── 6. The same problem, deliberately overfitted: far too much capacity for far too few
//      examples. Train accuracy goes to 100% while test accuracy falls behind — the network
//      memorizes the training points, noise included, instead of learning the shape. ──
Console.WriteLine("\n─────────────────────────────────────────────");
Console.WriteLine("Overfitting: same problem, 20 training points, a much larger network\n");

const int TinyCount = 20;
float[] tinyX = trainX[..(TinyCount * 2)];
float[] tinyY = trainY[..TinyCount];

var overfit = new Sequential(inputs: 2)
    .Dense<Tanh>(64)
    .Dense<Tanh>(64)
    .Dense<Sigmoid>(1)
    .Build(seed: 7);

Console.WriteLine($"  {overfit.ParameterCount} parameters, {tinyY.Length} training examples " +
                  $"— {(double)overfit.ParameterCount / tinyY.Length:F0}x more knobs than data points\n");
Console.WriteLine("  epoch    train acc   test acc");

overfit.Train(tinyX, tinyY, epochs: 3000, learningRate: 0.5f, batchSize: 8,
    onEpoch: (epoch, _) =>
    {
        if (epoch % 1000 != 0 && epoch != 1) return;

        Console.WriteLine($"  {epoch,5}    {Accuracy(overfit, tinyX, tinyY),8:P1}   {Accuracy(overfit, testX, testY),7:P1}");
    });

Console.WriteLine($"\n  Perfect on the {TinyCount} points it has seen, but worse on the held-out set than");
Console.WriteLine($"  the smaller network trained on {TrainCount} ({Accuracy(overfit, testX, testY):P1} vs {Accuracy(moons, testX, testY):P1}).");
Console.WriteLine("  That gap is overfitting: capacity spent memorizing noise rather than shape.");
Console.WriteLine("  Nothing in this library detects it for you — only the held-out split does.");

// ── Helpers ──

/// <summary>
/// Fraction of examples whose thresholded prediction matches the label. Accuracy is the metric
/// worth watching for classification; MSE loss is what training actually minimizes, and the two
/// can move in opposite directions.
/// </summary>
static float Accuracy(Network network, ReadOnlySpan<float> x, ReadOnlySpan<float> y)
{
    int correct = 0;

    for (int i = 0; i < y.Length; i++)
    {
        float predicted = network.Predict(x.Slice(i * network.Inputs, network.Inputs))[0];
        if (MathF.Round(predicted) == y[i]) correct++;
    }

    return (float)correct / y.Length;
}

/// <summary>
/// Samples the network across the input plane and prints the class it assigns to each cell.
/// The point is to make it visible that the network has learned a genuinely curved boundary —
/// something no single perceptron could represent, and the whole reason for the hidden layers.
/// </summary>
static string PlotBoundary(Network network)
{
    const int Rows = 15, Columns = 60;
    const float MinX = -1.6f, MaxX = 2.6f, MinY = -1.2f, MaxY = 1.6f;

    var sb = new System.Text.StringBuilder();
    var point = new float[2];

    for (int r = 0; r < Rows; r++)
    {
        // Screen rows run top to bottom; the y axis runs bottom to top.
        point[1] = MaxY - (MaxY - MinY) * r / (Rows - 1);
        sb.Append("  ");

        for (int c = 0; c < Columns; c++)
        {
            point[0] = MinX + (MaxX - MinX) * c / (Columns - 1);
            sb.Append(network.Predict(point)[0] >= 0.5f ? '#' : '·');
        }

        sb.AppendLine();
    }

    return sb.ToString();
}
