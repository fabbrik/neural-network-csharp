using System.Numerics;
using System.Runtime.CompilerServices;

namespace NN;

/// <summary>
/// Vectorized primitives shared by every layer. Non-generic on purpose: the JIT emits a separate
/// copy of a generic type's code per value-type instantiation, so keeping these out of
/// <see cref="Dense{TActivation}"/> avoids duplicating them for every activation.
/// </summary>
internal static class SimdOps
{
    /// <summary>Dot product. Both spans must have the same length.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
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
    /// Fused scale-and-accumulate: <c>dest += src * scale</c>. This is the workhorse of the
    /// backward pass — it accumulates weight gradients and propagates the input gradient.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AddScaled(Span<float> dest, ReadOnlySpan<float> src, float scale)
    {
        int width = Vector<float>.Count;
        int n = dest.Length;
        int i = 0;

        var vScale = new Vector<float>(scale);

        for (; i <= n - width; i += width)
        {
            var d = new Vector<float>(dest.Slice(i, width));
            var s = new Vector<float>(src.Slice(i, width));
            (d + s * vScale).CopyTo(dest.Slice(i, width));
        }

        for (; i < n; i++)
            dest[i] += src[i] * scale;
    }

}
