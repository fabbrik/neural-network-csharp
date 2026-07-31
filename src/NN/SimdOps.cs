using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace NN;

/// <summary>
/// Vectorized primitives shared by every layer. Non-generic on purpose: the JIT emits a separate
/// copy of a generic type's code per value-type instantiation, so keeping these out of
/// <see cref="Dense{TActivation}"/> avoids duplicating them for every activation.
///
/// <para>The two operations are implemented differently on purpose, and the split is a measured
/// result rather than a preference. <see cref="AddScaled"/> delegates to
/// <see cref="TensorPrimitives"/>, which beats the hand-rolled loop by about 2.5× on any span
/// worth vectorizing. <see cref="Dot"/> does not, because there it loses: a dot product ends in a
/// reduction, and <c>TensorPrimitives.Dot</c> carries a single accumulator chain, so on a
/// 4096-element span it measures the same as the one-accumulator reference below and about 1.5×
/// slower than the two-accumulator version kept here. It is also a real call where this is
/// inlined, which costs more than the whole operation at layer widths of 8. Both alternatives
/// are benchmarked in <c>NN.Bench</c>; re-measure before changing this, especially on a machine
/// with wider vectors than the ARM64 NEON these numbers came from.</para>
/// </summary>
internal static class SimdOps
{
    /// <summary>Dot product. Both spans must have the same length.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length) throw new ArgumentException($"Span lengths must match: {a.Length} != {b.Length}.", nameof(b));

        int width = Vector<float>.Count;
        int n = a.Length;
        int i = 0;

        // Two accumulators: independent dependency chains keep the multiply-add pipeline fed.
        var acc0 = Vector<float>.Zero;
        var acc1 = Vector<float>.Zero;

        for (; i <= n - 2 * width; i += 2 * width)
        {
            acc0 += new Vector<float>(a.Slice(i, width)) * new Vector<float>(b.Slice(i, width));
            acc1 += new Vector<float>(a.Slice(i + width, width)) * new Vector<float>(b.Slice(i + width, width));
        }

        for (; i <= n - width; i += width)
            acc0 += new Vector<float>(a.Slice(i, width)) * new Vector<float>(b.Slice(i, width));

        float z = Vector.Sum(acc0 + acc1);

        for (; i < n; i++)   // scalar tail, when n isn't a multiple of the SIMD width
            z += a[i] * b[i];

        return z;
    }

    /// <summary>
    /// The pre-activation pass of a dense layer: <c>dest[j] = dot(weights_j, x) + bias[j]</c> for
    /// every unit, where unit j's weights are the contiguous window <c>weights[j*n .. (j+1)*n]</c>.
    ///
    /// <para><b>The <see cref="MethodImplOptions.NoInlining"/> is load-bearing, and removing it
    /// costs a factor of eight.</b> Inlined into
    /// <see cref="Dense{TActivation}.Forward(ReadOnlySpan{float}, Span{float})"/> — which is
    /// generic, and by then also contains an inlined <see cref="Dot"/> and an inlined vectorized
    /// activation — the JIT stops eliminating the bounds checks on the inner vector loads. The
    /// disassembly shows a range-check branch guarding every single <c>ldr q</c>, and the layer
    /// measures 88 µs where this exact code measures 10.7 µs when called from anywhere else.
    /// Compiled on its own, small and non-generic, it optimizes cleanly and the caller pays one
    /// ordinary call per layer. Re-measure <c>LayerBenchmarks</c> before touching this.</para>
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    public static void MatVec(ReadOnlySpan<float> weights, ReadOnlySpan<float> x, ReadOnlySpan<float> bias, Span<float> dest)
    {
        int n = x.Length;

        for (int j = 0; j < dest.Length; j++)
            dest[j] = Dot(weights.Slice(j * n, n), x) + bias[j];
    }

    /// <summary>
    /// Fused scale-and-accumulate: <c>dest += src * scale</c>. This is the workhorse of the
    /// backward pass — it accumulates weight gradients and propagates the input gradient.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AddScaled(Span<float> dest, ReadOnlySpan<float> src, float scale)
    {
        if (dest.Length != src.Length) throw new ArgumentException($"Span lengths must match: {dest.Length} != {src.Length}.", nameof(src));

        // Writing back over an input is explicitly supported when the spans overlap exactly.
        TensorPrimitives.MultiplyAdd(src, scale, dest, dest);
    }

}
