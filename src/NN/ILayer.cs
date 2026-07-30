namespace NN;

/// <summary>
/// Non-generic view of a layer, so a network can hold a heterogeneous stack
/// (<c>Dense&lt;ReLU&gt;</c> then <c>Dense&lt;Sigmoid&gt;</c>) in one list.
///
/// <para><b>Not thread-safe.</b> A layer owns mutable buffers — gradient accumulators and the
/// backward pass's activation cache — so one instance cannot be used from two threads at once.
/// Give each thread its own network, or serialize access.</para>
/// </summary>
public interface ILayer
{
    int Inputs { get; }
    int Units { get; }

    /// <summary>Total trainable parameters (weights + biases). Used by <c>Network.Summary</c>.</summary>
    int ParameterCount { get; }

    /// <summary>
    /// Computes activations for one example. <b>Inference only</b>: it caches nothing, so it is
    /// safe to call at any time — including between <see cref="ForwardTrain"/> and
    /// <see cref="Backward"/> — without disturbing a pending backward pass.
    /// </summary>
    void Forward(ReadOnlySpan<float> aIn, Span<float> aOut);

    /// <summary>
    /// Computes activations for one example <i>and</i> caches the inputs and outputs that the
    /// following <see cref="Backward"/> needs.
    ///
    /// <para>Split from <see cref="Forward"/> deliberately. When a single method both computed
    /// and cached, any incidental forward pass — evaluating a loss, checking a prediction
    /// mid-training — silently overwrote the cache and made the next <see cref="Backward"/>
    /// compute gradients for the wrong example, with no error. Now only the training path
    /// writes the cache, so that hazard cannot arise.</para>
    /// </summary>
    void ForwardTrain(ReadOnlySpan<float> aIn, Span<float> aOut);

    /// <summary>
    /// Propagates the gradient one layer back, using the example cached by the most recent
    /// <see cref="ForwardTrain"/> and consuming that cache.
    /// </summary>
    /// <param name="gradOut">dL/da for this layer's outputs, length <see cref="Units"/>.</param>
    /// <param name="gradIn">
    /// Receives dL/dx for this layer's inputs, length <see cref="Inputs"/>. Pass an empty span
    /// for the first layer — nothing downstream consumes it, so computing it is wasted work.
    /// </param>
    /// <exception cref="InvalidOperationException">
    /// No <see cref="ForwardTrain"/> has run since the last <see cref="Backward"/>. Backprop
    /// needs that example's activations; without them there is nothing to differentiate.
    /// </exception>
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
