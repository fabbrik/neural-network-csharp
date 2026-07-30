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
