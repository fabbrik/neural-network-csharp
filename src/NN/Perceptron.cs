namespace NN;

/// <summary>
/// Classic single-unit perceptron: a <see cref="Dense{TActivation}"/> of one unit with a step
/// activation, trained by the perceptron rule <c>w += lr * (y - ŷ) * x</c>. Converges in a
/// finite number of updates iff the data is linearly separable.
/// </summary>
public sealed class Perceptron
{
    private readonly Dense<Step> _layer;
    private readonly float[] _out = new float[1];

    public Perceptron(int inputs) => _layer = new Dense<Step>(inputs, units: 1);

    public int Inputs => _layer.Inputs;
    public Span<float> Weights => _layer.UnitWeights(0);
    public float Bias { get => _layer.Bias[0]; set => _layer.Bias[0] = value; }

    public float Predict(ReadOnlySpan<float> x)
    {
        _layer.Forward(x, _out);
        return _out[0];
    }

    /// <summary>
    /// Trains on a contiguous row-major batch (example i at <c>x[i * Inputs ..]</c>).
    /// Returns the number of epochs actually run — stops early once an epoch makes no update.
    /// </summary>
    public int Train(ReadOnlySpan<float> x, ReadOnlySpan<float> y, int epochs = 100, float learningRate = 0.1f)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(epochs);
        if (float.IsNaN(learningRate) || float.IsInfinity(learningRate) || learningRate < 0f)
            throw new ArgumentOutOfRangeException(nameof(learningRate), learningRate, "Learning rate must be finite and non-negative.");

        int samples = y.Length;
        if (samples == 0) throw new ArgumentException("No samples.", nameof(y));

        if (x.Length != samples * Inputs)
            throw new ArgumentException(
                $"Targets describe {samples} examples, so inputs should be {samples * Inputs} floats " +
                $"({samples} × {Inputs}), but got {x.Length}.", nameof(x));

        Span<float> w = Weights;

        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            bool converged = true;

            for (int i = 0; i < samples; i++)
            {
                ReadOnlySpan<float> xi = x.Slice(i * Inputs, Inputs);
                float error = y[i] - Predict(xi);
                if (error == 0f) continue;

                converged = false;
                float delta = learningRate * error;

                for (int k = 0; k < Inputs; k++)
                    w[k] += delta * xi[k];

                Bias += delta;
            }

            if (converged) return epoch;
        }

        return epochs;
    }
}
