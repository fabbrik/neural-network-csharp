namespace NN;

/// <summary>
/// Non-generic view of a layer, so a network can hold a heterogeneous stack
/// (<c>Dense&lt;ReLU&gt;</c> then <c>Dense&lt;Sigmoid&gt;</c>) in one list.
/// </summary>
public interface ILayer
{
    int Inputs { get; }
    int Units { get; }

    /// <summary>Total trainable parameters (weights + biases). Used by <c>Network.Summary</c>.</summary>
    int ParameterCount { get; }

    /// <summary>Computes activations for one example and caches what the backward pass needs.</summary>
    void Forward(ReadOnlySpan<float> aIn, Span<float> aOut);

    /// <summary>
    /// Propagates the gradient one layer back.
    /// </summary>
    /// <param name="gradOut">dL/da for this layer's outputs, length <see cref="Units"/>.</param>
    /// <param name="gradIn">
    /// Receives dL/dx for this layer's inputs, length <see cref="Inputs"/>. Pass an empty span
    /// for the first layer — nothing downstream consumes it, so computing it is wasted work.
    /// </param>
    void Backward(ReadOnlySpan<float> gradOut, Span<float> gradIn);

    /// <summary>Applies the accumulated gradients and clears them.</summary>
    void ApplyGradients(float learningRate, int batchSize);

    /// <summary>Clears accumulated gradients without applying them.</summary>
    void ZeroGradients();

    /// <summary>Randomizes weights. Zero-initialized layers cannot learn — every unit stays identical.</summary>
    void Initialize(Random rng);

    /// <summary>
    /// Stable identifier for this layer's type, e.g. <c>"Dense&lt;Tanh&gt;"</c>. Written to model
    /// files and used to reconstruct the right type on load, so it must not change casually —
    /// renaming it invalidates every previously saved model.
    /// </summary>
    string Descriptor { get; }

    /// <summary>
    /// Reads one parameter by flat index, weights first then biases, in
    /// <c>[0, <see cref="ParameterCount"/>)</c>. Together with <see cref="SetParameter"/> and
    /// <see cref="GetParameterGradient"/> this gives a layer-type-agnostic view of the
    /// parameters, which is what lets <see cref="GradientCheck"/> verify any network.
    /// </summary>
    float GetParameter(int index);

    /// <summary>Writes one parameter by the same flat index as <see cref="GetParameter"/>.</summary>
    void SetParameter(int index, float value);

    /// <summary>Reads the accumulated gradient for the parameter at the same flat index.</summary>
    float GetParameterGradient(int index);

    /// <summary>Writes this layer's parameters (weights then biases) for serialization.</summary>
    void WriteParameters(BinaryWriter writer);

    /// <summary>Reads parameters previously written by <see cref="WriteParameters"/>.</summary>
    void ReadParameters(BinaryReader reader);
}
