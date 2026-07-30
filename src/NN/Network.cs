namespace NN;

/// <summary>
/// A feed-forward stack trained by backpropagation with mini-batch gradient descent
/// and mean-squared-error loss.
///
/// All working buffers are allocated once in the constructor, so training allocates nothing
/// per example — only the shuffled index array is per-network state.
///
/// <para><b>Not thread-safe, and it lends out its buffers.</b> Those two facts are related and
/// both matter:</para>
/// <list type="bullet">
///   <item><description>One instance cannot be trained or queried from two threads at once —
///   the activation buffers, gradient accumulators and shuffle order are all shared mutable
///   state. Give each thread its own <see cref="Network"/>.</description></item>
///   <item><description><see cref="Predict"/> returns a view of an internal buffer that the
///   <i>next</i> call overwrites. Copy it (<c>.ToArray()</c>) if you need to keep it, and never
///   hold two prediction results at once.</description></item>
/// </list>
/// </summary>
public sealed class Network
{
    private readonly ILayer[] _layers;
    private readonly float[][] _activations;   // _activations[i] = output of layer i
    private readonly float[][] _grads;         // _grads[i]       = dL/da for layer i's outputs

    // Two generators on purpose. Weight initialization draws as many numbers as the network has
    // parameters, so a single shared generator would make the shuffle sequence depend on the
    // architecture — resize a hidden layer and the batch composition silently changes too.
    private readonly Random _shuffleRng;
    private int[] _order = [];

    public int Inputs => _layers[0].Inputs;
    public int Outputs => _layers[^1].Units;
    public IReadOnlyList<ILayer> Layers => _layers;

    /// <summary>
    /// How this network is scored, and how its raw output is presented. Mean squared error by
    /// default; <see cref="SoftmaxCrossEntropy"/> turns it into a classifier whose outputs are
    /// probabilities. Saved with the model, so a reloaded network behaves identically.
    /// </summary>
    public ILoss LossFunction { get; }

    /// <summary>Creates a network from an ordered layer stack and randomizes its weights.</summary>
    /// <param name="seed">Fixed seed so runs are reproducible while you experiment.</param>
    /// <param name="layers">Layers in forward order; adjacent shapes must match.</param>
    public Network(int seed, params ILayer[] layers) : this(seed, layers, initialize: true, loss: null) { }

    /// <summary>Creates a network with the default seed (42).</summary>
    /// <param name="layers">Layers in forward order; adjacent shapes must match.</param>
    public Network(params ILayer[] layers) : this(seed: 42, layers) { }

    /// <summary>Creates a network with an explicit loss.</summary>
    /// <param name="seed">Seed for weight initialization.</param>
    /// <param name="loss">How to score outputs; null means <see cref="MeanSquaredError"/>.</param>
    /// <param name="layers">Layers in forward order; adjacent shapes must match.</param>
    public Network(int seed, ILoss? loss, params ILayer[] layers)
        : this(seed, layers, initialize: true, loss) { }

    /// <param name="seed">Seed for weight initialization.</param>
    /// <param name="layers">Layers in forward order.</param>
    /// <param name="initialize">
    /// False when the layers already carry trained parameters — a model being loaded from disk.
    /// Randomizing them at that point would silently destroy the very weights just read.
    /// </param>
    /// <param name="loss">How to score outputs; null means <see cref="MeanSquaredError"/>.</param>
    private Network(int seed, ILayer[] layers, bool initialize, ILoss? loss)
    {
        ArgumentOutOfRangeException.ThrowIfZero(layers.Length);

        for (int i = 1; i < layers.Length; i++)
            if (layers[i].Inputs != layers[i - 1].Units)
                throw new ArgumentException(
                    $"Layer {i} takes {layers[i].Inputs} inputs but layer {i - 1} emits {layers[i - 1].Units}.", nameof(layers));

        LossFunction = loss ?? Losses.Default;

        // Checked once, here, rather than per example: softmax cross-entropy's fused gradient is
        // only valid over a linear output layer, and a mismatch would produce a wrong gradient
        // rather than an error.
        LossFunction.Validate(layers[^1]);

        _layers = layers;
        _shuffleRng = new Random(seed);
        _activations = new float[layers.Length][];
        _grads = new float[layers.Length][];

        var initRng = new Random(seed);

        for (int i = 0; i < layers.Length; i++)
        {
            _activations[i] = new float[layers[i].Units];
            _grads[i] = new float[layers[i].Units];

            if (initialize) layers[i].Initialize(initRng);
        }
    }

    /// <summary>
    /// Wraps layers whose parameters are already trained, skipping weight initialization.
    /// Used by <see cref="ModelIO.Load(string)"/>.
    /// </summary>
    internal static Network FromTrainedLayers(ILayer[] layers, ILoss? loss = null) =>
        new(seed: 0, layers, initialize: false, loss);

    /// <summary>
    /// Runs one example through the stack and returns the output buffer, which is owned by the
    /// network and overwritten by the next call — copy it if you need to keep it.
    ///
    /// <para>Inference only: it disturbs no training state, so it is safe to call in the middle
    /// of a gradient accumulation (to log a prediction, say) without corrupting the pending
    /// backward pass.</para>
    /// </summary>
    public ReadOnlySpan<float> Predict(ReadOnlySpan<float> x)
    {
        if (x.Length != Inputs) throw new ArgumentException($"Expected {Inputs} inputs, got {x.Length}.", nameof(x));

        _layers[0].Forward(x, _activations[0]);

        for (int i = 1; i < _layers.Length; i++)
            _layers[i].Forward(_activations[i - 1], _activations[i]);

        // For a softmax network this converts logits into probabilities, so callers always see
        // the output in its meaningful form. It is safe to rewrite the buffer in place: the
        // backward pass reads each layer's own cache, not this one.
        LossFunction.Transform(_activations[^1]);

        return _activations[^1];
    }

    /// <summary>Total trainable parameters across every layer.</summary>
    public int ParameterCount
    {
        get
        {
            int total = 0;
            foreach (ILayer layer in _layers) total += layer.ParameterCount;
            return total;
        }
    }

    /// <summary>
    /// A Keras-style textual summary of the architecture: one row per layer with its output
    /// width and parameter count, then the total.
    /// </summary>
    public string Summary()
    {
        var sb = new System.Text.StringBuilder();
        string rule = new('─', 42);

        sb.AppendLine("Layer                     Output    Params");
        sb.AppendLine(rule);

        foreach (ILayer layer in _layers)
            sb.AppendLine($"{layer.Descriptor,-24}{layer.Units,8}{layer.ParameterCount,10}");

        sb.AppendLine(rule);
        sb.AppendLine($"Input width: {Inputs}");
        sb.AppendLine($"Trainable parameters: {ParameterCount}");

        return sb.ToString();
    }

    /// <summary>Clears accumulated gradients in every layer.</summary>
    public void ZeroGradients()
    {
        foreach (ILayer layer in _layers)
            layer.ZeroGradients();
    }

    /// <summary>
    /// Scores one example under this network's <see cref="LossFunction"/>. Touches no training
    /// state — neither the gradient accumulators nor the backward pass's activation cache — which
    /// is what lets <see cref="GradientCheck"/> evaluate it thousands of times mid-check.
    /// </summary>
    public float Loss(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
    {
        if (y.Length != Outputs) throw new ArgumentException($"Expected {Outputs} targets, got {y.Length}.", nameof(y));

        return LossFunction.Evaluate(Predict(x), y);
    }

    /// <summary>
    /// Forward pass, then backward pass, accumulating gradients without applying them.
    /// Returns this example's mean squared error.
    /// </summary>
    public float AccumulateGradients(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
    {
        if (y.Length != Outputs) throw new ArgumentException($"Expected {Outputs} targets, got {y.Length}.", nameof(y));

        // ForwardTrain, not Predict: the backward pass below needs each layer to have cached the
        // inputs and activations of *this* example.
        _layers[0].ForwardTrain(x, _activations[0]);

        for (int i = 1; i < _layers.Length; i++)
            _layers[i].ForwardTrain(_activations[i - 1], _activations[i]);

        int last = _layers.Length - 1;

        // Present the raw output the way the loss defines it — softmax turns logits into
        // probabilities here, MSE leaves them alone.
        LossFunction.Transform(_activations[last]);

        ReadOnlySpan<float> a = _activations[last];

        // Seed the backward pass. What the loss writes is dL/d(the last layer's own output):
        // for MSE that is 2(a - y)/m, and for softmax cross-entropy the fused p - y, which is
        // correct precisely because that layer is linear and its derivative is 1.
        LossFunction.Gradient(a, y, _grads[last]);

        float loss = LossFunction.Evaluate(a, y);

        // Walk backwards; each layer hands its input gradient to the one before it.
        // The first layer gets an empty span — nothing consumes dL/dx of the raw input.
        for (int i = last; i >= 0; i--)
            _layers[i].Backward(_grads[i], i > 0 ? _grads[i - 1] : Span<float>.Empty);

        return loss;
    }

    /// <summary>
    /// Trains on a contiguous row-major dataset: example i is <c>x[i * Inputs ..]</c>
    /// with target <c>y[i * Outputs ..]</c>.
    /// </summary>
    /// <param name="x">Inputs, row-major: <c>Inputs</c> floats per example.</param>
    /// <param name="y">Targets, row-major: <c>Outputs</c> floats per example.</param>
    /// <param name="epochs">Number of full passes over the data.</param>
    /// <param name="learningRate">Gradient descent step size.</param>
    /// <param name="batchSize">Examples per update; 0 (the default) means full batch.</param>
    /// <param name="onEpoch">Optional per-epoch callback receiving (epoch, mean loss).</param>
    /// <returns>Mean loss over the final epoch.</returns>
    public float Train(
        ReadOnlySpan<float> x,
        ReadOnlySpan<float> y,
        int epochs = 1000,
        float learningRate = 0.1f,
        int batchSize = 0,
        Action<int, float>? onEpoch = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(epochs);
        if (float.IsNaN(learningRate) || float.IsInfinity(learningRate) || learningRate < 0f)
            throw new ArgumentOutOfRangeException(nameof(learningRate), learningRate, "Learning rate must be finite and non-negative.");
        ArgumentOutOfRangeException.ThrowIfNegative(batchSize);

        // Validate both operands against each other. Deriving the sample count from y alone and
        // trusting x to match is how a mismatched pair turns into a slice exception thousands of
        // examples into training, or — worse — into silent training on misaligned rows.
        if (y.Length % Outputs != 0)
            throw new ArgumentException(
                $"Targets length {y.Length} is not a multiple of the {Outputs} network outputs.", nameof(y));

        int samples = y.Length / Outputs;
        if (samples == 0) throw new ArgumentException("No samples.", nameof(y));

        if (x.Length != samples * Inputs)
            throw new ArgumentException(
                $"Targets describe {samples} examples, so inputs should be {samples * Inputs} floats " +
                $"({samples} × {Inputs}), but got {x.Length}.", nameof(x));

        if (batchSize == 0) batchSize = samples;   // 0 means full batch

        if (_order.Length != samples)
        {
            _order = new int[samples];
            for (int i = 0; i < samples; i++) _order[i] = i;
        }

        float epochLoss = 0f;

        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            _shuffleRng.Shuffle(_order);
            epochLoss = 0f;

            for (int start = 0; start < samples; start += batchSize)
            {
                int count = Math.Min(batchSize, samples - start);

                for (int k = 0; k < count; k++)
                {
                    int i = _order[start + k];
                    epochLoss += AccumulateGradients(x.Slice(i * Inputs, Inputs), y.Slice(i * Outputs, Outputs));
                }

                foreach (ILayer layer in _layers)
                    layer.ApplyGradients(learningRate, count);
            }

            epochLoss /= samples;
            onEpoch?.Invoke(epoch, epochLoss);
        }

        return epochLoss;
    }
}
