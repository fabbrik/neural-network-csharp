using System.Numerics;

namespace NN.Bench;

/// <summary>
/// Deliberately unoptimized alternatives to what the library ships, kept here rather than in the
/// library because their only purpose is to be beaten.
///
/// <para>A claim like "the contiguous layout is worth it" is only meaningful against the thing it
/// is supposedly better than, so each of these is the honest version of a design the library
/// rejected: a single accumulator chain, a delegate-dispatched activation, and NumPy's
/// feature-major weight layout.</para>
/// </summary>
internal static class Reference
{
    /// <summary>
    /// Vectorized dot product with one accumulator. Every iteration's add depends on the previous
    /// one, so the chain cannot be pipelined — this is what the shipping two-accumulator version
    /// is meant to improve on.
    /// </summary>
    public static float DotSingleAccumulator(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        int width = Vector<float>.Count;
        int n = a.Length;
        int i = 0;

        var acc = Vector<float>.Zero;

        for (; i <= n - width; i += width)
            acc += new Vector<float>(a.Slice(i, width)) * new Vector<float>(b.Slice(i, width));

        float z = Vector.Sum(acc);

        for (; i < n; i++)
            z += a[i] * b[i];

        return z;
    }

    /// <summary>
    /// A dense layer whose activation is a <c>Func&lt;float, float&gt;</c> field instead of a
    /// generic type parameter — the design <see cref="Dense{TActivation}"/> rejects. Identical
    /// arithmetic and identical memory layout; the only difference is that the activation is an
    /// indirect call the JIT cannot inline into the loop.
    /// </summary>
    public sealed class DelegateDense(int inputs, int units, Func<float, float> activation)
    {
        private readonly Func<float, float> _activation = activation;

        public int Inputs { get; } = inputs;
        public int Units { get; } = units;
        public float[] Weights { get; } = new float[inputs * units];
        public float[] Bias { get; } = new float[units];

        public void Forward(ReadOnlySpan<float> aIn, Span<float> aOut)
        {
            ReadOnlySpan<float> w = Weights;

            for (int j = 0; j < Units; j++)
            {
                float z = SimdOps.Dot(w.Slice(j * Inputs, Inputs), aIn) + Bias[j];
                aOut[j] = _activation(z);
            }
        }
    }

    /// <summary>
    /// A dense layer storing weights feature-major — NumPy's <c>(inputs, units)</c> layout, where
    /// unit j's weights are the column <c>W[:, j]</c> and its elements sit <c>Units</c> floats
    /// apart. Same weights, same result, same arithmetic; only the memory order differs.
    ///
    /// <para>Each unit's dot product becomes a strided gather, which defeats both the SIMD path
    /// and the cache line: the CPU loads 16 consecutive floats and the loop uses one of them.
    /// This is the layout the study guide (§12) claims is the expensive choice.</para>
    /// </summary>
    public sealed class FeatureMajorDense(int inputs, int units)
    {
        public int Inputs { get; } = inputs;
        public int Units { get; } = units;

        /// <summary>Weight of input k into unit j lives at <c>[k * Units + j]</c>.</summary>
        public float[] Weights { get; } = new float[inputs * units];
        public float[] Bias { get; } = new float[units];

        public void Forward(ReadOnlySpan<float> aIn, Span<float> aOut)
        {
            ReadOnlySpan<float> w = Weights;

            for (int j = 0; j < Units; j++)
            {
                float z = Bias[j];

                for (int k = 0; k < Inputs; k++)
                    z += w[k * Units + j] * aIn[k];   // stride of Units floats per step

                aOut[j] = MathF.Tanh(z);
            }
        }
    }
}
