namespace NN;

/// <summary>
/// A feed-forward stack trained by backpropagation with mini-batch gradient descent
/// and mean-squared-error loss.
///
/// All working buffers are allocated once in the constructor, so training allocates nothing
/// per example — only the shuffled index array is per-network state.
/// </summary>
public sealed class Network
{
    private readonly ILayer[] _layers;
    private readonly float[][] _activations;   // _activations[i] = output of layer i
    private readonly float[][] _grads;         // _grads[i]       = dL/da for layer i's outputs
    private readonly Random _rng;
    private int[] _order = [];

    public int Inputs => _layers[0].Inputs;
    public int Outputs => _layers[^1].Units;
    public IReadOnlyList<ILayer> Layers => _layers;

    /// <summary>Creates a network from an ordered layer stack and randomizes its weights.</summary>
    /// <param name="seed">Fixed seed so runs are reproducible while you experiment.</param>
    /// <param name="layers">Layers in forward order; adjacent shapes must match.</param>
    public Network(int seed, params ILayer[] layers) : this(seed, layers, initialize: true) { }

    /// <summary>Creates a network with the default seed (42).</summary>
    /// <param name="layers">Layers in forward order; adjacent shapes must match.</param>
    public Network(params ILayer[] layers) : this(seed: 42, layers) { }

    /// <param name="seed">Seed for weight initialization.</param>
    /// <param name="layers">Layers in forward order.</param>
    /// <param name="initialize">
    /// False when the layers already carry trained parameters — a model being loaded from disk.
    /// Randomizing them at that point would silently destroy the very weights just read.
    /// </param>
    private Network(int seed, ILayer[] layers, bool initialize)
    {
        ArgumentOutOfRangeException.ThrowIfZero(layers.Length);

        for (int i = 1; i < layers.Length; i++)
            if (layers[i].Inputs != layers[i - 1].Units)
                throw new ArgumentException(
                    $"Layer {i} takes {layers[i].Inputs} inputs but layer {i - 1} emits {layers[i - 1].Units}.", nameof(layers));

        _layers = layers;
        _rng = new Random(seed);
        _activations = new float[layers.Length][];
        _grads = new float[layers.Length][];

        for (int i = 0; i < layers.Length; i++)
        {
            _activations[i] = new float[layers[i].Units];
            _grads[i] = new float[layers[i].Units];

            if (initialize) layers[i].Initialize(_rng);
        }
    }

    /// <summary>
    /// Wraps layers whose parameters are already trained, skipping weight initialization.
    /// Used by <see cref="ModelIO.Load(string)"/>.
    /// </summary>
    internal static Network FromTrainedLayers(ILayer[] layers) => new(seed: 0, layers, initialize: false);

    /// <summary>Runs one example through the stack and returns the output buffer (owned by the network — copy it if you keep it).</summary>
    public ReadOnlySpan<float> Predict(ReadOnlySpan<float> x)
    {
        _layers[0].Forward(x, _activations[0]);

        for (int i = 1; i < _layers.Length; i++)
            _layers[i].Forward(_activations[i - 1], _activations[i]);

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
            sb.AppendLine($"{DescribeType(layer),-24}{layer.Units,8}{layer.ParameterCount,10}");

        sb.AppendLine(rule);
        sb.AppendLine($"Input width: {Inputs}");
        sb.AppendLine($"Trainable parameters: {ParameterCount}");

        return sb.ToString();
    }

    /// <summary>Renders <c>Dense`1[Tanh]</c> as the readable <c>Dense&lt;Tanh&gt;</c>.</summary>
    private static string DescribeType(ILayer layer)
    {
        Type t = layer.GetType();
        if (!t.IsGenericType) return t.Name;

        string name = t.Name[..t.Name.IndexOf('`')];
        string args = string.Join(", ", Array.ConvertAll(t.GetGenericArguments(), a => a.Name));
        return $"{name}<{args}>";
    }

    /// <summary>Clears accumulated gradients in every layer.</summary>
    public void ZeroGradients()
    {
        foreach (ILayer layer in _layers)
            layer.ZeroGradients();
    }

    /// <summary>Mean squared error for one example, without touching gradients.</summary>
    public float Loss(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
    {
        ReadOnlySpan<float> a = Predict(x);
        float loss = 0f;

        for (int j = 0; j < a.Length; j++)
        {
            float e = a[j] - y[j];
            loss += e * e;
        }

        return loss / a.Length;
    }

    /// <summary>
    /// Forward pass, then backward pass, accumulating gradients without applying them.
    /// Returns this example's mean squared error.
    /// </summary>
    public float AccumulateGradients(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
    {
        ReadOnlySpan<float> a = Predict(x);

        // MSE = mean over outputs of (a - y)²  →  dL/da = 2(a - y) / outputs
        int last = _layers.Length - 1;
        Span<float> gradOut = _grads[last];
        float loss = 0f;

        for (int j = 0; j < a.Length; j++)
        {
            float e = a[j] - y[j];
            loss += e * e;
            gradOut[j] = 2f * e / a.Length;
        }

        // Walk backwards; each layer hands its input gradient to the one before it.
        // The first layer gets an empty span — nothing consumes dL/dx of the raw input.
        for (int i = last; i >= 0; i--)
            _layers[i].Backward(_grads[i], i > 0 ? _grads[i - 1] : Span<float>.Empty);

        return loss / a.Length;
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
        int samples = y.Length / Outputs;
        if (samples == 0) throw new ArgumentException("No samples.", nameof(y));
        if (batchSize <= 0) batchSize = samples;   // 0 means full batch

        if (_order.Length != samples)
        {
            _order = new int[samples];
            for (int i = 0; i < samples; i++) _order[i] = i;
        }

        float epochLoss = 0f;

        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            _rng.Shuffle(_order);
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
