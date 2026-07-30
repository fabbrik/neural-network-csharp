using System.Runtime.CompilerServices;

namespace NN;

/// <summary>
/// Dense (fully connected) layer.
///
/// Weights live in one flat array in unit-major order: unit j's <see cref="Inputs"/> weights
/// occupy <c>Weights[j * Inputs .. (j + 1) * Inputs]</c>. That is the transpose of NumPy's
/// <c>(n, j)</c> layout, and it is what turns each dot product into a contiguous SIMD walk
/// instead of a strided gather. The same layout pays off again in the backward pass: both
/// the weight-gradient accumulation and the input-gradient propagation walk a unit's weights
/// contiguously.
/// </summary>
public sealed class Dense<TActivation> : ILayer where TActivation : IActivation
{
    public int Inputs { get; }
    public int Units { get; }
    public readonly float[] Weights;
    public readonly float[] Bias;

    public int ParameterCount => Weights.Length + Bias.Length;

    public string Descriptor => $"Dense<{typeof(TActivation).Name}>";

    // Gradients, accumulated across a mini-batch and drained by ApplyGradients.
    private readonly float[] _weightGrads;
    private readonly float[] _biasGrads;

    // Forward-pass cache. Backprop needs the inputs that produced the current activations,
    // and the activations themselves (every derivative here is expressible in terms of a).
    private readonly float[] _lastInput;
    private readonly float[] _lastOutput;

    public Dense(int inputs, int units)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputs);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(units);

        Inputs = inputs;
        Units = units;
        Weights = new float[inputs * units];
        Bias = new float[units];

        _weightGrads = new float[inputs * units];
        _biasGrads = new float[units];
        _lastInput = new float[inputs];
        _lastOutput = new float[units];
    }

    public Dense(int inputs, int units, float[] weights, float[] bias)
        : this(inputs, units)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(weights.Length, inputs * units);
        ArgumentOutOfRangeException.ThrowIfNotEqual(bias.Length, units);

        weights.CopyTo(Weights, 0);
        bias.CopyTo(Bias, 0);
    }

    /// <summary>Weights of unit <paramref name="j"/>, as a contiguous window into the flat array.</summary>
    public Span<float> UnitWeights(int j) => Weights.AsSpan(j * Inputs, Inputs);

    /// <summary>
    /// Xavier/Glorot uniform initialization: weights drawn from ±sqrt(6 / (fan_in + fan_out)),
    /// which keeps activation variance roughly stable as signals move through the stack.
    /// Biases start at zero — they break no symmetry, so they need no noise.
    /// </summary>
    public void Initialize(Random rng)
    {
        float limit = MathF.Sqrt(6f / (Inputs + Units));

        for (int i = 0; i < Weights.Length; i++)
            Weights[i] = (rng.NextSingle() * 2f - 1f) * limit;

        Array.Clear(Bias);
    }

    /// <summary>
    /// Forward pass over one example. Writes into <paramref name="aOut"/> so the hot path
    /// allocates nothing — reuse the same buffer across calls.
    /// </summary>
    public void Forward(ReadOnlySpan<float> aIn, Span<float> aOut)
    {
        if (aIn.Length != Inputs) throw new ArgumentException($"Expected {Inputs} inputs, got {aIn.Length}.", nameof(aIn));
        if (aOut.Length != Units) throw new ArgumentException($"Expected {Units} outputs, got {aOut.Length}.", nameof(aOut));

        ReadOnlySpan<float> w = Weights;

        for (int j = 0; j < Units; j++)
        {
            float z = SimdOps.Dot(w.Slice(j * Inputs, Inputs), aIn) + Bias[j];
            aOut[j] = TActivation.Apply(z);
        }

        aIn.CopyTo(_lastInput);
        aOut.CopyTo(_lastOutput);
    }

    /// <summary>Allocating convenience overload — prefer the span version inside a training loop.</summary>
    public float[] Forward(ReadOnlySpan<float> aIn)
    {
        var aOut = new float[Units];
        Forward(aIn, aOut);
        return aOut;
    }

    /// <summary>
    /// Forward pass over a batch of <paramref name="count"/> examples stored contiguously,
    /// row-major: example i occupies <c>batch[i * Inputs ..]</c>.
    /// </summary>
    public void ForwardBatch(ReadOnlySpan<float> batch, Span<float> outputs, int count)
    {
        for (int i = 0; i < count; i++)
            Forward(batch.Slice(i * Inputs, Inputs), outputs.Slice(i * Units, Units));
    }

    /// <summary>
    /// Backward pass over the example most recently seen by <see cref="Forward(ReadOnlySpan{float}, Span{float})"/>.
    ///
    /// For each unit j:  δ_j = dL/da_j · g'(z_j)
    /// then              dL/dW_jk += δ_j · x_k        (accumulated, not applied)
    ///                   dL/db_j  += δ_j
    ///                   dL/dx_k  += δ_j · W_jk       (handed to the previous layer)
    /// </summary>
    public void Backward(ReadOnlySpan<float> gradOut, Span<float> gradIn)
    {
        if (gradOut.Length != Units) throw new ArgumentException($"Expected {Units} output grads, got {gradOut.Length}.", nameof(gradOut));

        bool propagate = !gradIn.IsEmpty;
        if (propagate)
        {
            if (gradIn.Length != Inputs) throw new ArgumentException($"Expected {Inputs} input grads, got {gradIn.Length}.", nameof(gradIn));
            gradIn.Clear();
        }

        ReadOnlySpan<float> x = _lastInput;
        ReadOnlySpan<float> a = _lastOutput;

        for (int j = 0; j < Units; j++)
        {
            float delta = gradOut[j] * TActivation.DerivativeFromOutput(a[j]);
            if (delta == 0f) continue;   // dead ReLU: nothing to accumulate or propagate

            int offset = j * Inputs;

            SimdOps.AddScaled(_weightGrads.AsSpan(offset, Inputs), x, delta);
            _biasGrads[j] += delta;

            if (propagate)
                SimdOps.AddScaled(gradIn, Weights.AsSpan(offset, Inputs), delta);
        }
    }

    /// <summary>
    /// Gradient descent step: <c>W -= lr · (dL/dW / batchSize)</c>, then clears the accumulators.
    /// Dividing by the batch size keeps a usable learning rate independent of batch size.
    /// </summary>
    public void ApplyGradients(float learningRate, int batchSize)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);

        float step = -learningRate / batchSize;

        SimdOps.AddScaled(Weights, _weightGrads, step);
        SimdOps.AddScaled(Bias, _biasGrads, step);

        ZeroGradients();
    }

    public void ZeroGradients()
    {
        Array.Clear(_weightGrads);
        Array.Clear(_biasGrads);
    }

    /// <summary>
    /// Flat parameter view: indices <c>[0, Weights.Length)</c> are weights, the rest are biases.
    /// </summary>
    public float GetParameter(int index) =>
        index < Weights.Length ? Weights[index] : Bias[index - Weights.Length];

    public void SetParameter(int index, float value)
    {
        if (index < Weights.Length) Weights[index] = value;
        else Bias[index - Weights.Length] = value;
    }

    public float GetParameterGradient(int index) =>
        index < _weightGrads.Length ? _weightGrads[index] : _biasGrads[index - _weightGrads.Length];

    /// <summary>
    /// Writes weights then biases as raw little-endian float32. Gradients and the forward-pass
    /// cache are deliberately not saved — they're training scratch space, not part of the model.
    /// </summary>
    public void WriteParameters(BinaryWriter writer)
    {
        foreach (float w in Weights) writer.Write(w);
        foreach (float b in Bias) writer.Write(b);
    }

    public void ReadParameters(BinaryReader reader)
    {
        for (int i = 0; i < Weights.Length; i++) Weights[i] = reader.ReadSingle();
        for (int i = 0; i < Bias.Length; i++) Bias[i] = reader.ReadSingle();
    }

}
