using BenchmarkDotNet.Attributes;

namespace NN.Bench;

/// <summary>
/// The activation step alone, with the dot products taken out of the picture.
///
/// <para><see cref="LayerBenchmarks"/> measures the activation change where it actually matters —
/// inside a real layer, where the dot products dominate and dilute it. That dilution is the
/// honest number, but it hides how large the underlying effect is, which is what this measures:
/// <c>exp</c> and <c>tanh</c> evaluated one float at a time versus a vector at a time, over a
/// span the size of a layer's output.</para>
///
/// <para>The widths are layer widths, not arbitrary ones: 10 is an MNIST output layer, 128 a
/// typical hidden layer, 1024 a wide one.</para>
/// </summary>
[MemoryDiagnoser(displayGenColumns: false)]
public class ActivationBenchmarks
{
    private float[] _z = [];
    private float[] _work = [];

    [Params(10, 128, 1024)]
    public int Width { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);

        _z = new float[Width];
        for (int i = 0; i < Width; i++) _z[i] = (rng.NextSingle() - 0.5f) * 6f;

        // Each benchmark gets the same starting values, since GlobalSetup runs once per benchmark.
        _work = new float[Width];
        _z.CopyTo(_work, 0);
    }

    // No per-iteration reset. These activations write in place, so each iteration feeds on the
    // previous one's output and the values converge — sigmoid towards its fixed point, tanh
    // towards zero, softmax towards uniform. That is fine here and better than the alternative:
    // an [IterationSetup] forces BenchmarkDotNet down to one invocation per iteration, whose
    // timer overhead would swamp a 40ns operation. It works because exp and tanh cost the same
    // whatever normal float they are handed, and none of these paths reaches a denormal.

    [Benchmark(Baseline = true)]
    public void SigmoidScalar() => IActivation.Elementwise<Sigmoid>(_work);

    [Benchmark]
    public void SigmoidVectorized() => Sigmoid.ApplyAll(_work);

    [Benchmark]
    public void TanhScalar() => IActivation.Elementwise<Tanh>(_work);

    [Benchmark]
    public void TanhVectorized() => Tanh.ApplyAll(_work);

    /// <summary>
    /// Softmax, which the classifier runs once per example on top of its output layer. The
    /// shipping version keeps the max-subtraction explicit — <c>TensorPrimitives.SoftMax</c>
    /// computes <c>exp(z)/Σexp(z)</c> literally and overflows without it — so this measures a
    /// vectorized shift plus a vectorized kernel against three scalar passes.
    /// </summary>
    [Benchmark]
    public void SoftmaxScalar()
    {
        Span<float> outputs = _work;

        float max = float.NegativeInfinity;
        for (int j = 0; j < outputs.Length; j++)
            if (outputs[j] > max) max = outputs[j];

        float sum = 0f;
        for (int j = 0; j < outputs.Length; j++)
        {
            outputs[j] = MathF.Exp(outputs[j] - max);
            sum += outputs[j];
        }

        for (int j = 0; j < outputs.Length; j++)
            outputs[j] /= sum;
    }

    [Benchmark]
    public void SoftmaxVectorized() => SoftmaxCrossEntropy.Instance.Transform(_work);
}
