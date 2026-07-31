using System.Numerics.Tensors;
using BenchmarkDotNet.Attributes;

namespace NN.Bench;

/// <summary>
/// Does the SIMD dot product actually beat the obvious loop, and by how much?
///
/// <para>The study guide claims vectorization is worth having and that two accumulators beat one.
/// Both are measured here rather than asserted. The lengths straddle the interesting regimes: 8
/// is roughly one vector wide (so loop setup dominates), 512 fits comfortably in L1, and 4096
/// starts to feel memory bandwidth.</para>
///
/// <para>The last row is the obvious alternative to shipping any of this: hand the problem to
/// <c>TensorPrimitives</c> and delete the loop. It is measured here because it is the row that
/// decided <see cref="SimdOps.Dot"/> stays hand-rolled — a dot product ends in a reduction, and
/// the runtime's kernel carries one accumulator chain through it, which is exactly the design
/// <c>SimdOneAccumulator</c> exists to show the cost of.</para>
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

    /// <summary>The runtime's own kernel, which the shipping version is measured against.</summary>
    [Benchmark]
    public float TensorPrimitivesDot() => TensorPrimitives.Dot(_a, _b);
}

/// <summary>
/// The other half of the SIMD surface: <c>dest += src * scale</c>, which the backward pass runs
/// twice per unit (weight-gradient accumulation and input-gradient propagation) and which
/// therefore does more of the training work than the dot product does.
///
/// <para>Unlike the dot product this one is pure streaming — one multiply-add per element with no
/// reduction — so it saturates memory bandwidth quickly and the vector width matters less than
/// whether a fused multiply-add is emitted.</para>
/// </summary>
[MemoryDiagnoser(displayGenColumns: false)]
public class AddScaledBenchmarks
{
    private float[] _dest = [];
    private float[] _src = [];

    [Params(8, 64, 512, 4096)]
    public int Length { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);

        _dest = new float[Length];
        _src = new float[Length];

        for (int i = 0; i < Length; i++)
        {
            _dest[i] = rng.NextSingle() - 0.5f;
            _src[i] = rng.NextSingle() - 0.5f;
        }
    }

    /// <summary>The hand-rolled <c>Vector&lt;float&gt;</c> loop the library shipped before.</summary>
    [Benchmark(Baseline = true)]
    public void HandRolled() => Reference.AddScaled(_dest, _src, 0.01f);

    /// <summary>What ships: <c>TensorPrimitives.MultiplyAdd</c>.</summary>
    [Benchmark]
    public void TensorPrimitivesMultiplyAdd() => SimdOps.AddScaled(_dest, _src, 0.01f);
}
