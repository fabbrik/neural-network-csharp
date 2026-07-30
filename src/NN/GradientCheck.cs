namespace NN;

/// <summary>
/// Verifies a network's backward pass against finite differences.
///
/// <para>
/// For every parameter w, backprop's analytic dL/dw must match the numerical estimate
/// <c>(L(w + ε) - L(w - ε)) / 2ε</c>. The central difference is used rather than the one-sided
/// version because its error is O(ε²) instead of O(ε), which matters at float precision.
/// </para>
///
/// <para>
/// This is far too slow for training — two full forward passes per parameter — but it is the
/// only way to be sure a backward pass is correct. A subtly wrong gradient still trains
/// <i>somewhat</i>, which makes such bugs extremely hard to find any other way.
/// </para>
/// </summary>
public static class GradientCheck
{
    /// <summary>
    /// Compares analytic and numerical gradients for one example across every parameter.
    /// </summary>
    /// <param name="network">Network to check. Its gradients are cleared as a side effect.</param>
    /// <param name="x">One input example.</param>
    /// <param name="y">The matching target.</param>
    /// <param name="epsilon">
    /// Step size. The default sits at the bottom of the accuracy U-curve: larger ε is dominated
    /// by truncation error, smaller ε by float32 roundoff in <c>L(w + ε) - L(w - ε)</c>. Expect a
    /// relative error around 1e-4; anything near 1e-1 means the backward pass is wrong.
    /// </param>
    /// <returns>The largest relative error found across all parameters of all layers.</returns>
    public static float MaxRelativeError(
        Network network, ReadOnlySpan<float> x, ReadOnlySpan<float> y, float epsilon = 1e-2f)
    {
        ArgumentNullException.ThrowIfNull(network);

        // Analytic gradients for this one example.
        network.ZeroGradients();
        network.AccumulateGradients(x, y);

        // Snapshot them: the numerical probing below runs forward passes, and while those don't
        // touch gradients today, comparing against a live buffer would be fragile.
        var analytic = new float[network.Layers.Count][];
        for (int l = 0; l < network.Layers.Count; l++)
        {
            ILayer layer = network.Layers[l];
            analytic[l] = new float[layer.ParameterCount];

            for (int p = 0; p < layer.ParameterCount; p++)
                analytic[l][p] = layer.GetParameterGradient(p);
        }

        float worst = 0f;

        for (int l = 0; l < network.Layers.Count; l++)
        {
            ILayer layer = network.Layers[l];

            for (int p = 0; p < layer.ParameterCount; p++)
            {
                float original = layer.GetParameter(p);

                layer.SetParameter(p, original + epsilon);
                float lossPlus = network.Loss(x, y);

                layer.SetParameter(p, original - epsilon);
                float lossMinus = network.Loss(x, y);

                layer.SetParameter(p, original);

                float numeric = (lossPlus - lossMinus) / (2f * epsilon);
                float denominator = MathF.Max(1e-6f, MathF.Abs(numeric) + MathF.Abs(analytic[l][p]));

                worst = MathF.Max(worst, MathF.Abs(numeric - analytic[l][p]) / denominator);
            }
        }

        return worst;
    }
}
