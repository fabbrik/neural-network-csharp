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
///
/// <para><b>Not thread-safe.</b> The gradient accumulators and the forward-pass cache are mutable
/// instance state. <see cref="Forward(ReadOnlySpan{float}, Span{float})"/> alone touches no
/// instance state and so is safe to call concurrently on a layer nobody is training; everything
/// else needs exclusive access.</para>
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

    // Forward-pass cache, written only by ForwardTrain. Backprop needs the inputs that produced
    // the current activations, and the activations themselves (every derivative here is
    // expressible in terms of a). _cached guards the pairing: Backward without a preceding
    // ForwardTrain would silently differentiate a stale example.
    private readonly float[] _lastInput;
    private readonly float[] _lastOutput;
    private bool _cached;

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
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(bias);
        ArgumentOutOfRangeException.ThrowIfNotEqual(weights.Length, inputs * units);
        ArgumentOutOfRangeException.ThrowIfNotEqual(bias.Length, units);

        weights.CopyTo(Weights, 0);
        bias.CopyTo(Bias, 0);
    }

    /// <summary>Weights of unit <paramref name="j"/>, as a contiguous window into the flat array.</summary>
    public Span<float> UnitWeights(int j)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(j);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(j, Units);

        return Weights.AsSpan(j * Inputs, Inputs);
    }

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
    /// Forward pass over one example, for inference. Writes into <paramref name="aOut"/> so the
    /// hot path allocates nothing — reuse the same buffer across calls.
    ///
    /// <para>Caches nothing, so it cannot disturb a pending backward pass. Training uses
    /// <see cref="ForwardTrain"/> instead.</para>
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
    }

    /// <summary>
    /// Forward pass over one example, for training: identical arithmetic to
    /// <see cref="Forward(ReadOnlySpan{float}, Span{float})"/>, plus a copy of this example's
    /// inputs and outputs into the cache that <see cref="Backward"/> consumes.
    /// </summary>
    public void ForwardTrain(ReadOnlySpan<float> aIn, Span<float> aOut)
    {
        Forward(aIn, aOut);

        aIn.CopyTo(_lastInput);
        aOut.CopyTo(_lastOutput);
        _cached = true;
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
    ///
    /// <para><b>Inference only</b> — there is no batched backward pass to pair with it. It caches
    /// nothing, so a following <see cref="Backward"/> will correctly refuse rather than quietly
    /// differentiate whichever example happened to be last.</para>
    ///
    /// <para>Today this just loops the single-example path, so it re-streams the whole weight
    /// matrix per example and buys nothing over calling
    /// <see cref="Forward(ReadOnlySpan{float}, Span{float})"/> yourself — see the measured
    /// numbers and the tiled-GEMM discussion in the study guide (§25).</para>
    /// </summary>
    public void ForwardBatch(ReadOnlySpan<float> batch, Span<float> outputs, int count)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count);

        int expectedInputs = checked(count * Inputs);
        if (batch.Length != expectedInputs)
            throw new ArgumentException(
                $"Expected {expectedInputs} input values for {count} examples of width {Inputs}, got {batch.Length}.", nameof(batch));

        int expectedOutputs = checked(count * Units);
        if (outputs.Length != expectedOutputs)
            throw new ArgumentException(
                $"Expected {expectedOutputs} output slots for {count} examples of width {Units}, got {outputs.Length}.", nameof(outputs));

        for (int i = 0; i < count; i++)
            Forward(batch.Slice(i * Inputs, Inputs), outputs.Slice(i * Units, Units));
    }

    /// <summary>
    /// Backward pass over the example most recently seen by <see cref="ForwardTrain"/>, whose
    /// cache it consumes: each backward pass needs its own forward pass.
    ///
    /// For each unit j:  δ_j = dL/da_j · g'(z_j)
    /// then              dL/dW_jk += δ_j · x_k        (accumulated, not applied)
    ///                   dL/db_j  += δ_j
    ///                   dL/dx_k  += δ_j · W_jk       (handed to the previous layer)
    /// </summary>
    public void Backward(ReadOnlySpan<float> gradOut, Span<float> gradIn)
    {
        if (gradOut.Length != Units) throw new ArgumentException($"Expected {Units} output grads, got {gradOut.Length}.", nameof(gradOut));

        if (!_cached)
            throw new InvalidOperationException(
                $"Backward on {Descriptor} without a preceding ForwardTrain — there is no cached " +
                "example to differentiate. Each backward pass consumes exactly one forward pass.");

        _cached = false;

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
            // Nothing to accumulate or propagate: every gradient below is a multiple of delta.
            // Usually a saturated or dead unit (g'(z) = 0), but a zero incoming gradient does it too.
            if (delta == 0f) continue;

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
        if (float.IsNaN(learningRate) || float.IsInfinity(learningRate))
            throw new ArgumentOutOfRangeException(nameof(learningRate), learningRate, "Learning rate must be finite.");

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
    public float GetParameter(int index)
    {
        ValidateParameterIndex(index);

        return index < Weights.Length ? Weights[index] : Bias[index - Weights.Length];
    }

    public void SetParameter(int index, float value)
    {
        ValidateParameterIndex(index);

        if (index < Weights.Length) Weights[index] = value;
        else Bias[index - Weights.Length] = value;
    }

    public float GetParameterGradient(int index)
    {
        ValidateParameterIndex(index);

        return index < _weightGrads.Length ? _weightGrads[index] : _biasGrads[index - _weightGrads.Length];
    }

    private void ValidateParameterIndex(int index)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(index, ParameterCount);
    }

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
