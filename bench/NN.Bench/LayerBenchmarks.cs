using BenchmarkDotNet.Attributes;

namespace NN.Bench;

/// <summary>
/// The two design decisions the README singles out, measured against the alternatives they were
/// chosen over: the generic activation (versus a delegate field) and the unit-major weight layout
/// (versus NumPy's feature-major one).
///
/// <para>Sizes matter here. At 2×4 — the XOR network — everything fits in a single cache line and
/// no layout can be wrong; the effects only appear once a layer is bigger than the cache line it
/// is walking. That is itself worth knowing, so the small case is measured rather than skipped.
/// </para>
/// </summary>
[MemoryDiagnoser(displayGenColumns: false)]
public class LayerBenchmarks
{
    private Dense<Tanh> _generic = null!;
    private Reference.DelegateDense _delegated = null!;
    private Reference.FeatureMajorDense _featureMajor = null!;

    // The same generic-vs-delegate comparison with a one-instruction activation. Tanh is a
    // transcendental costing tens of cycles, which can easily swamp an indirect call and hide
    // whatever the dispatch is worth; ReLU cannot hide anything.
    private Dense<ReLU> _genericCheap = null!;
    private Reference.DelegateDense _delegatedCheap = null!;

    // Same inlined generic activation, applied per unit instead of per layer — the shape the
    // forward pass had before it started handing whole vectors to TensorPrimitives.Tanh.
    private Reference.ScalarActivationDense<Tanh> _scalarActivation = null!;

    private float[] _input = [];
    private float[] _output = [];

    /// <summary>Inputs × units. 2×4 is the XOR layer; 784×128 is an MNIST-sized first layer.</summary>
    [Params("2x4", "64x64", "784x128")]
    public string Shape { get; set; } = "";

    [GlobalSetup]
    public void Setup()
    {
        string[] parts = Shape.Split('x');
        int inputs = int.Parse(parts[0]), units = int.Parse(parts[1]);

        var rng = new Random(42);

        _generic = new Dense<Tanh>(inputs, units);
        _generic.Initialize(rng);

        _delegated = new Reference.DelegateDense(inputs, units, MathF.Tanh);
        _featureMajor = new Reference.FeatureMajorDense(inputs, units);

        _genericCheap = new Dense<ReLU>(inputs, units);
        _delegatedCheap = new Reference.DelegateDense(inputs, units, z => z > 0f ? z : 0f);

        _scalarActivation = new Reference.ScalarActivationDense<Tanh>(inputs, units);

        // Same weights everywhere, so the only variables are dispatch and memory order.
        for (int j = 0; j < units; j++)
            for (int k = 0; k < inputs; k++)
            {
                float w = _generic.Weights[j * inputs + k];

                _delegated.Weights[j * inputs + k] = w;
                _genericCheap.Weights[j * inputs + k] = w;
                _delegatedCheap.Weights[j * inputs + k] = w;
                _scalarActivation.Weights[j * inputs + k] = w;
                _featureMajor.Weights[k * units + j] = w;
            }

        _input = new float[inputs];
        for (int i = 0; i < inputs; i++) _input[i] = rng.NextSingle() - 0.5f;

        _output = new float[units];
    }

    /// <summary>What ships: activation inlined via a generic type parameter, weights unit-major.</summary>
    [Benchmark(Baseline = true)]
    public void GenericActivation() => _generic.Forward(_input, _output);

    /// <summary>Same inlined activation, but evaluated per unit rather than over the layer.</summary>
    [Benchmark]
    public void ScalarActivation() => _scalarActivation.Forward(_input, _output);

    /// <summary>Same layout, but the activation is an un-inlinable indirect call per unit.</summary>
    [Benchmark]
    public void DelegateActivation() => _delegated.Forward(_input, _output);

    /// <summary>Same activation cost, but each dot product is a strided gather.</summary>
    [Benchmark]
    public void FeatureMajorLayout() => _featureMajor.Forward(_input, _output);

    /// <summary>Generic dispatch with an activation too cheap to hide the call it replaces.</summary>
    [Benchmark]
    public void GenericActivationCheap() => _genericCheap.Forward(_input, _output);

    /// <summary>Delegate dispatch with the same cheap activation.</summary>
    [Benchmark]
    public void DelegateActivationCheap() => _delegatedCheap.Forward(_input, _output);
}

/// <summary>
/// Is <see cref="Dense{TActivation}.ForwardBatch"/> worth anything today?
///
/// <para>§25 of the study guide claims it is not — that looping the single-example path re-streams
/// the whole weight matrix per example and buys nothing, and that a tiled GEMM would be the real
/// win. The first half of that claim is checkable right now, and this checks it. The result is
/// meant to be a <i>null</i>: if these two rows are within noise of each other, the honest
/// conclusion is that the method is a placeholder, and the docs should say so.</para>
/// </summary>
[MemoryDiagnoser(displayGenColumns: false)]
public class BatchBenchmarks
{
    private Dense<Tanh> _layer = null!;
    private float[] _batch = [];
    private float[] _outputs = [];

    private const int Inputs = 128;
    private const int Units = 128;

    [Params(1, 32, 256)]
    public int BatchSize { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);

        _layer = new Dense<Tanh>(Inputs, Units);
        _layer.Initialize(rng);

        _batch = new float[BatchSize * Inputs];
        for (int i = 0; i < _batch.Length; i++) _batch[i] = rng.NextSingle() - 0.5f;

        _outputs = new float[BatchSize * Units];
    }

    [Benchmark(Baseline = true)]
    public void OneAtATime()
    {
        for (int i = 0; i < BatchSize; i++)
            _layer.Forward(_batch.AsSpan(i * Inputs, Inputs), _outputs.AsSpan(i * Units, Units));
    }

    [Benchmark]
    public void ForwardBatch() => _layer.ForwardBatch(_batch, _outputs, BatchSize);
}
