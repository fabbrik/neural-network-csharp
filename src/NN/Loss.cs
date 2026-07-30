namespace NN;

/// <summary>
/// How a network scores its output and seeds the backward pass.
///
/// <para>A loss is two things that must agree: a number saying how wrong a prediction was, and
/// the derivative of that number with respect to what the last layer produced. Getting the pair
/// out of step is the classic silent bug — training still runs, loss still falls, and the network
/// optimizes something other than what you are measuring. <see cref="GradientCheck"/> catches it,
/// which is why every loss here is gradient-checked in the test suite.</para>
///
/// <para><b>Why this is an interface with instances, not a static-abstract generic like
/// <see cref="IActivation"/>.</b> Activations run once per unit per layer per example and sit in
/// the innermost loop, so the generic form exists to let the JIT inline them. A loss runs
/// <i>once per example</i>. Benchmarking showed even per-unit delegate dispatch to be
/// unmeasurable (see <c>bench/README.md</c>), so per-example dispatch is beyond irrelevant, and
/// an ordinary interface is far easier to read.</para>
/// </summary>
public interface ILoss
{
    /// <summary>
    /// Stable identifier written into model files, so a saved network reloads with the same
    /// output semantics. Renaming one invalidates previously saved models, exactly as with
    /// <see cref="ILayer.Descriptor"/>.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Converts the final layer's raw output into what the caller should see, in place.
    ///
    /// <para>Most losses leave it alone. Softmax is the reason this exists: it turns a vector of
    /// unbounded scores into a probability distribution, and it cannot be an
    /// <see cref="IActivation"/> because it needs every unit's value at once, while an activation
    /// sees one number at a time.</para>
    /// </summary>
    void Transform(Span<float> outputs);

    /// <summary>Scores one prediction against its target. Lower is better; 0 is perfect.</summary>
    /// <param name="outputs">The network's output, after <see cref="Transform"/>.</param>
    /// <param name="targets">The desired output.</param>
    float Evaluate(ReadOnlySpan<float> outputs, ReadOnlySpan<float> targets);

    /// <summary>
    /// Seeds backpropagation: writes the derivative of the loss with respect to <b>the last
    /// layer's own output</b> — the value it produced before <see cref="Transform"/> ran.
    ///
    /// <para>That distinction is the whole trick behind softmax. See
    /// <see cref="SoftmaxCrossEntropy"/>.</para>
    /// </summary>
    void Gradient(ReadOnlySpan<float> outputs, ReadOnlySpan<float> targets, Span<float> into);

    /// <summary>
    /// Checks that the architecture is compatible, and throws with an explanation if not.
    /// Called once when a network is built or loaded, so a mismatch is a startup error rather
    /// than a quietly wrong gradient.
    /// </summary>
    void Validate(ILayer outputLayer);
}

/// <summary>
/// Mean squared error: the average of the squared differences. The default, and the right choice
/// for regression — predicting numbers rather than choosing a category.
///
/// <para>For classification it is a poor fit, and §22's MNIST run measures the cost. See
/// <see cref="SoftmaxCrossEntropy"/>.</para>
/// </summary>
public sealed class MeanSquaredError : ILoss
{
    /// <summary>The shared instance. The type is stateless, so one is enough.</summary>
    public static readonly MeanSquaredError Instance = new();

    private MeanSquaredError() { }

    public string Name => "mse";

    /// <summary>Nothing to do: the last layer's activation is already the answer.</summary>
    public void Transform(Span<float> outputs) { }

    /// <summary>L = (1/m) · Σ (a - y)²</summary>
    public float Evaluate(ReadOnlySpan<float> outputs, ReadOnlySpan<float> targets)
    {
        float total = 0f;

        for (int j = 0; j < outputs.Length; j++)
        {
            float e = outputs[j] - targets[j];
            total += e * e;
        }

        return total / outputs.Length;
    }

    /// <summary>dL/da = 2(a - y) / m</summary>
    public void Gradient(ReadOnlySpan<float> outputs, ReadOnlySpan<float> targets, Span<float> into)
    {
        for (int j = 0; j < outputs.Length; j++)
            into[j] = 2f * (outputs[j] - targets[j]) / outputs.Length;
    }

    /// <summary>Works with any output layer.</summary>
    public void Validate(ILayer outputLayer) { }
}

/// <summary>
/// Softmax output with cross-entropy loss: the standard way to make a network choose among
/// mutually exclusive categories.
///
/// <para><b>Softmax</b> turns the last layer's raw scores — <i>logits</i> — into a probability
/// distribution:</para>
/// <code>
///   p_j = exp(z_j) / Σ_k exp(z_k)
/// </code>
/// <para>Every output is positive and they sum to 1. Ten independent sigmoids cannot do this:
/// they can all say 0.9, which is incoherent when the digit is exactly one of them. Softmax makes
/// the outputs <i>compete</i> — raising one necessarily lowers the others.</para>
///
/// <para><b>Cross-entropy</b> then scores that distribution by asking a single question: what
/// probability did you assign to the right answer?</para>
/// <code>
///   L = -Σ_j y_j · log(p_j)     which for a one-hot target is just  -log(p_correct)
/// </code>
/// <para>Confidently right costs nothing; confidently wrong costs unboundedly much. MSE, by
/// contrast, caps the penalty at 1 per output no matter how wrong the network is.</para>
///
/// <para><b>The fusion, and why it is the entire point.</b> Differentiating softmax on its own
/// gives a full Jacobian — every output depends on every logit. Differentiating cross-entropy on
/// its own gives a <c>1/p</c> that explodes as p approaches 0. Composed, almost everything
/// cancels and what remains is:</para>
/// <code>
///   dL/dz_j = p_j - y_j
/// </code>
/// <para>Prediction minus target. No Jacobian, no division, nothing to overflow. Compare MSE
/// through a sigmoid, whose gradient carries a <c>σ'(z) = a(1-a)</c> factor that collapses toward
/// zero exactly when the network is confidently wrong — the moment it most needs to learn. That
/// vanishing factor is why the MSE version of the MNIST demo needs a learning rate of 1.0, and
/// removing it is why this one does not.</para>
///
/// <para><b>Requires a linear output layer.</b> The fused gradient above is only correct if the
/// last layer hands over raw logits, so it must be <c>Dense&lt;Identity&gt;</c>. Anything else is
/// rejected by <see cref="Validate"/> rather than silently producing a wrong gradient — squashing
/// the logits through a sigmoid first would make <c>p - y</c> simply false.</para>
/// </summary>
public sealed class SoftmaxCrossEntropy : ILoss
{
    /// <summary>The shared instance. The type is stateless, so one is enough.</summary>
    public static readonly SoftmaxCrossEntropy Instance = new();

    /// <summary>
    /// Floor for probabilities inside the logarithm. A softmax output can underflow to exactly 0
    /// in float32, and <c>log(0)</c> is negative infinity, which would poison the reported loss
    /// (though not the gradient, which never takes a logarithm).
    /// </summary>
    private const float Epsilon = 1e-7f;

    private SoftmaxCrossEntropy() { }

    public string Name => "softmax-cross-entropy";

    /// <summary>
    /// Softmax, in place and numerically stable.
    ///
    /// <para>The naive formula overflows: <c>exp(z)</c> is infinity for z above about 88 in
    /// float32, and logits of that size are ordinary in a trained network. Subtracting the
    /// largest logit first fixes it exactly, because softmax is shift-invariant — the same
    /// constant appears in every numerator and in the denominator, and cancels. After the shift
    /// the largest exponent is <c>exp(0) = 1</c>, so nothing can overflow.</para>
    /// </summary>
    public void Transform(Span<float> outputs)
    {
        float max = float.NegativeInfinity;
        for (int j = 0; j < outputs.Length; j++)
            if (outputs[j] > max) max = outputs[j];

        float sum = 0f;
        for (int j = 0; j < outputs.Length; j++)
        {
            outputs[j] = MathF.Exp(outputs[j] - max);
            sum += outputs[j];
        }

        for (int j = 0; j < outputs.Length; j++)
            outputs[j] /= sum;
    }

    /// <summary>L = -Σ y · log(p), the negative log-probability assigned to the true class.</summary>
    public float Evaluate(ReadOnlySpan<float> outputs, ReadOnlySpan<float> targets)
    {
        float total = 0f;

        for (int j = 0; j < outputs.Length; j++)
            if (targets[j] != 0f)
                total -= targets[j] * MathF.Log(MathF.Max(outputs[j], Epsilon));

        return total;
    }

    /// <summary>
    /// The fused gradient: <c>dL/dz = p - y</c>. See the class summary for why softmax's Jacobian
    /// and cross-entropy's reciprocal cancel each other out completely.
    /// </summary>
    public void Gradient(ReadOnlySpan<float> outputs, ReadOnlySpan<float> targets, Span<float> into)
    {
        for (int j = 0; j < outputs.Length; j++)
            into[j] = outputs[j] - targets[j];
    }

    /// <summary>The fused gradient assumes raw logits, so the output layer must be linear.</summary>
    public void Validate(ILayer outputLayer)
    {
        ArgumentNullException.ThrowIfNull(outputLayer);

        if (outputLayer.Descriptor != "Dense<Identity>")
            throw new ArgumentException(
                $"Softmax cross-entropy needs a linear output layer producing logits, but the last " +
                $"layer is {outputLayer.Descriptor}. Use Dense<Identity> (or Sequential.SoftmaxOutput), " +
                $"because the fused gradient p - y is only correct for raw logits — squashing them " +
                $"first would make it quietly wrong rather than obviously broken.");
    }
}

/// <summary>
/// The losses a model file can name, and the lookup that turns a saved name back into an
/// instance. Mirrors the layer factory table in <see cref="ModelIO"/>, and for the same reason:
/// a file names things as strings, and an explicit table is safer than reflecting over the string.
/// </summary>
public static class Losses
{
    private static readonly Dictionary<string, ILoss> ByName = new()
    {
        [MeanSquaredError.Instance.Name] = MeanSquaredError.Instance,
        [SoftmaxCrossEntropy.Instance.Name] = SoftmaxCrossEntropy.Instance,
    };

    /// <summary>The default when a model file does not say — and what version 1 files used.</summary>
    public static ILoss Default => MeanSquaredError.Instance;

    /// <summary>Looks up a loss by its <see cref="ILoss.Name"/>.</summary>
    /// <exception cref="InvalidDataException">No loss is registered under that name.</exception>
    public static ILoss ByNameOrThrow(string name)
    {
        if (!ByName.TryGetValue(name, out ILoss? loss))
            throw new InvalidDataException(
                $"Unknown loss '{name}' in model file. Known losses: {string.Join(", ", ByName.Keys)}.");

        return loss;
    }
}
