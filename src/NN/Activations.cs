using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace NN;

/// <summary>
/// Activation function. Declared with static abstract members so the JIT devirtualizes
/// and inlines the call at each generic instantiation — no delegate dispatch in the inner loop.
/// </summary>
public interface IActivation
{
    /// <summary>Maps the pre-activation <paramref name="z"/> to the unit's output.</summary>
    static abstract float Apply(float z);

    /// <summary>
    /// Applies the activation in place across a whole layer's pre-activations.
    ///
    /// <para>This is what the forward pass actually calls, and it exists because the scalar
    /// <see cref="Apply"/> is the wrong shape for the transcendental activations: <c>exp</c> and
    /// <c>tanh</c> cost tens of cycles each when evaluated one float at a time, and a layer
    /// evaluates one per unit. The vectorized forms compute a whole vector's worth per
    /// instruction.</para>
    ///
    /// <para>An activation with nothing to gain from vectorization implements this as
    /// <c>Elementwise&lt;Self&gt;</c>, which is exactly the loop it replaces. <see cref="ReLU"/>
    /// does: a compare-and-select loop is already negligible next to the dot products that
    /// produced its input, and the scalar form pins down the NaN behaviour that a max-based
    /// vector form would quietly change.</para>
    /// </summary>
    static abstract void ApplyAll(Span<float> z);

    /// <summary>
    /// The element-wise fallback, for activations whose <see cref="ApplyAll"/> has no reason to
    /// be anything cleverer. It lives here, taking the activation as a type argument, because a
    /// <c>static virtual</c> default body has no way to name its own implementing type.
    /// </summary>
    static void Elementwise<TActivation>(Span<float> z) where TActivation : IActivation
    {
        for (int i = 0; i < z.Length; i++)
            z[i] = TActivation.Apply(z[i]);
    }

    /// <summary>
    /// Derivative g'(z), expressed in terms of the already-computed output <c>a = g(z)</c>.
    /// Every activation here has a derivative cheaply recoverable from its output, so backprop
    /// reuses the cached forward activations instead of re-deriving z.
    /// </summary>
    static abstract float DerivativeFromOutput(float a);
}

public readonly struct Sigmoid : IActivation
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Apply(float z) => 1f / (1f + MathF.Exp(-z));

    public static void ApplyAll(Span<float> z) => TensorPrimitives.Sigmoid(z, z);

    // σ'(z) = σ(z)(1 - σ(z))
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DerivativeFromOutput(float a) => a * (1f - a);
}

public readonly struct Tanh : IActivation
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Apply(float z) => MathF.Tanh(z);

    public static void ApplyAll(Span<float> z) => TensorPrimitives.Tanh(z, z);

    // tanh'(z) = 1 - tanh²(z)
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DerivativeFromOutput(float a) => 1f - a * a;
}

public readonly struct ReLU : IActivation
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Apply(float z) => z > 0f ? z : 0f;

    // Not TensorPrimitives.Max(z, 0): that returns NaN for a NaN input, where the scalar form
    // above returns 0. The vectorized form would win nothing worth that change — see IActivation.
    public static void ApplyAll(Span<float> z) => IActivation.Elementwise<ReLU>(z);

    // Undefined at 0; the usual convention picks 0.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DerivativeFromOutput(float a) => a > 0f ? 1f : 0f;
}

public readonly struct Step : IActivation
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Apply(float z) => z >= 0f ? 1f : 0f;

    public static void ApplyAll(Span<float> z) => IActivation.Elementwise<Step>(z);

    /// <summary>
    /// The step function has a zero gradient everywhere it is defined, which is precisely why
    /// backprop cannot train it and why the perceptron needs its own update rule. Throws rather
    /// than silently zeroing every gradient in the network.
    /// </summary>
    public static float DerivativeFromOutput(float a) =>
        throw new NotSupportedException("Step is not differentiable — use Sigmoid, Tanh, or ReLU for backprop.");
}

public readonly struct Identity : IActivation
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Apply(float z) => z;

    /// <summary>Nothing to do — the pre-activations already are the outputs.</summary>
    public static void ApplyAll(Span<float> z) { }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DerivativeFromOutput(float a) => 1f;
}
