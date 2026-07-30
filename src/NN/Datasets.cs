namespace NN;

/// <summary>
/// Synthetic datasets for demonstrations and tests.
///
/// <para>XOR has four examples, which is enough to show that a hidden layer defeats the
/// Minsky–Papert wall and nothing more. It cannot show mini-batching (four examples is one full
/// batch), generalization (there is no held-out data — the four points <i>are</i> the problem),
/// or overfitting. Those need a dataset with more points than the network has parameters, and
/// with noise the network can memorize instead of learning.</para>
///
/// <para>Generated rather than downloaded on purpose: no data files, no network access, and
/// identical output on every machine for a given seed.</para>
/// </summary>
public static class Datasets
{
    /// <summary>
    /// The classic "two moons": two interleaving crescents that no straight line can separate,
    /// but a small hidden layer can. Real enough to need a train/test split, small enough to
    /// train in a fraction of a second.
    ///
    /// <para>Points are emitted row-major, matching what <see cref="Network.Train"/> expects:
    /// example i is <c>x[i * 2 ..]</c> with label <c>y[i]</c> (0 = lower moon, 1 = upper moon).
    /// Classes alternate, so any prefix or suffix of the arrays is balanced — which is what makes
    /// a simple positional train/test split legitimate.</para>
    /// </summary>
    /// <param name="count">Number of points. Split evenly between the two moons.</param>
    /// <param name="noise">Standard deviation of the Gaussian jitter added to each coordinate.
    /// At 0 the moons are clean arcs; around 0.2 they overlap enough that perfect accuracy is
    /// not achievable, which is what makes overfitting visible.</param>
    /// <param name="seed">Fixed so the dataset is identical on every run and every machine.</param>
    public static (float[] X, float[] Y) Moons(int count, float noise = 0.2f, int seed = 0)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(count);
        ArgumentOutOfRangeException.ThrowIfNegative(noise);

        var rng = new Random(seed);
        var x = new float[count * 2];
        var y = new float[count];

        for (int i = 0; i < count; i++)
        {
            // Alternate classes so that any contiguous slice stays roughly balanced.
            bool upper = i % 2 == 0;

            // Each moon is a half-circle; the upper one is shifted right and down so the two
            // crescents interlock rather than merely sitting side by side.
            double angle = Math.PI * rng.NextDouble();

            double cx = upper ? Math.Cos(angle) : 1.0 - Math.Cos(angle);
            double cy = upper ? Math.Sin(angle) : 0.5 - Math.Sin(angle);

            x[i * 2] = (float)(cx + Gaussian(rng) * noise);
            x[i * 2 + 1] = (float)(cy + Gaussian(rng) * noise);
            y[i] = upper ? 1f : 0f;
        }

        return (x, y);
    }

    /// <summary>
    /// One standard normal sample by the Box–Muller transform: two uniforms in, one Gaussian out.
    /// <c>Random</c> offers no Gaussian of its own, and uniform noise would give the moons hard
    /// edges rather than the soft overlap that makes the classification problem interesting.
    /// </summary>
    private static double Gaussian(Random rng)
    {
        // NextDouble() can return exactly 0, and log(0) is -infinity.
        double u1 = 1.0 - rng.NextDouble();
        double u2 = rng.NextDouble();

        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
