using BenchmarkDotNet.Attributes;

namespace NN.Bench;

/// <summary>
/// One full training step — forward, backward, and the descent update — on an MNIST-sized network.
///
/// <para>Every other benchmark here measures a primitive or a forward pass. This one measures the
/// thing the library actually spends its time doing, and it is the only place
/// <see cref="SimdOps.AddScaled"/> appears in proportion: the backward pass runs it twice per unit
/// (weight-gradient accumulation and input-gradient propagation), and <c>ApplyGradients</c> runs it
/// twice more per layer. A 2.5× on that primitive in isolation says nothing about training
/// throughput until it is weighed against the dot products it shares the step with.</para>
/// </summary>
[MemoryDiagnoser(displayGenColumns: false)]
public class TrainingBenchmarks
{
    private Network _net = null!;
    private float[] _x = [];
    private float[] _y = [];

    private const int Inputs = 784;
    private const int Hidden = 128;
    private const int Classes = 10;
    private const int BatchSize = 32;

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);

        _net = new Sequential(Inputs)
            .Dense<Tanh>(Hidden)
            .SoftmaxOutput(Classes)
            .Build(seed: 42);

        _x = new float[BatchSize * Inputs];
        for (int i = 0; i < _x.Length; i++) _x[i] = rng.NextSingle();

        _y = new float[BatchSize * Classes];
        for (int i = 0; i < BatchSize; i++) _y[i * Classes + rng.Next(Classes)] = 1f;
    }

    /// <summary>Gradient accumulation over one mini-batch, then one descent step.</summary>
    [Benchmark]
    public void MiniBatchStep()
    {
        for (int i = 0; i < BatchSize; i++)
            _net.AccumulateGradients(_x.AsSpan(i * Inputs, Inputs), _y.AsSpan(i * Classes, Classes));

        foreach (ILayer layer in _net.Layers)
            layer.ApplyGradients(0.1f, BatchSize);
    }

    /// <summary>The forward half alone, to show how the step divides.</summary>
    [Benchmark]
    public void ForwardOnly()
    {
        for (int i = 0; i < BatchSize; i++)
            _net.Predict(_x.AsSpan(i * Inputs, Inputs));
    }
}
