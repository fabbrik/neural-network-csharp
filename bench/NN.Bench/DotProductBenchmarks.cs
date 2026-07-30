using BenchmarkDotNet.Attributes;

namespace NN.Bench;

/// <summary>
/// Does the SIMD dot product actually beat the obvious loop, and by how much?
///
/// <para>The study guide claims vectorization is worth having and that two accumulators beat one.
/// Both are measured here rather than asserted. The lengths straddle the interesting regimes: 8
/// is roughly one vector wide (so loop setup dominates), 512 fits comfortably in L1, and 4096
/// starts to feel memory bandwidth.</para>
/// </summary>
[MemoryDiagnoser(displayGenColumns: false)]
public class DotProductBenchmarks
{
    private float[] _a = [];
    private float[] _b = [];

    [Params(8, 64, 512, 4096)]
    public int Length { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);

        _a = new float[Length];
        _b = new float[Length];

        for (int i = 0; i < Length; i++)
        {
            _a[i] = rng.NextSingle() - 0.5f;
            _b[i] = rng.NextSingle() - 0.5f;
        }
    }

    /// <summary>The baseline: what you would write without thinking about vectors.</summary>
    [Benchmark(Baseline = true)]
    public float Scalar()
    {
        float sum = 0f;

        for (int i = 0; i < _a.Length; i++)
            sum += _a[i] * _b[i];

        return sum;
    }

    /// <summary>One accumulator: vectorized, but with a single serial dependency chain.</summary>
    [Benchmark]
    public float SimdOneAccumulator() => Reference.DotSingleAccumulator(_a, _b);

    /// <summary>What the library actually ships — two independent accumulator chains.</summary>
    [Benchmark]
    public float SimdTwoAccumulators() => SimdOps.Dot(_a, _b);
}
