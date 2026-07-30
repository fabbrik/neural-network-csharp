using NN.Mnist;
using Xunit;

namespace NN.Tests;

/// <summary>
/// The end-to-end property the digit reader depends on: a digit that goes out through a PNG and
/// comes back through <see cref="ImageFile"/> and <see cref="DigitPreprocessor"/> must be the same
/// digit.
///
/// <para>The individual pieces are tested elsewhere. This tests them <i>composed</i>, which is
/// where the interesting failures live: decoding can be right and preprocessing can be right while
/// the pair still loses the digit to an off-by-one in the crop, a polarity flip, or a resample
/// that erodes thin strokes. Nothing here needs the MNIST dataset or a network connection.</para>
/// </summary>
public class ImageRoundTripTests : IDisposable
{
    private readonly string _directory = Directory.CreateTempSubdirectory("nn-roundtrip").FullName;

    public void Dispose() => Directory.Delete(_directory, recursive: true);

    /// <summary>
    /// Draws a crude 7 — a top bar and a descending diagonal — into a 28x28 canvas in MNIST's own
    /// convention: white ink on black. Asymmetric on both axes on purpose, so a flip or transpose
    /// anywhere in the pipeline cannot pass unnoticed.
    /// </summary>
    private static float[] SyntheticSeven()
    {
        var canvas = new float[Idx.PixelCount];

        void Ink(int x, int y)
        {
            if (x is >= 0 and < Idx.ImageSize && y is >= 0 and < Idx.ImageSize)
                canvas[y * Idx.ImageSize + x] = 1f;
        }

        for (int x = 7; x < 20; x++) { Ink(x, 6); Ink(x, 7); }              // the bar
        for (int i = 0; i < 13; i++) { Ink(19 - i, 7 + i); Ink(18 - i, 7 + i); }   // the diagonal

        return canvas;
    }

    /// <summary>
    /// Writes a 28x28 MNIST-convention image out the way a person would produce one: enlarged,
    /// inverted to dark ink on white paper, and surrounded by margin. Everything the preprocessor
    /// then has to undo.
    /// </summary>
    private string WriteAsPhotoStylePng(string name, float[] mnist, int scale, int margin)
    {
        int side = Idx.ImageSize * scale + 2 * margin;
        var pixels = new byte[side * side];
        Array.Fill(pixels, (byte)255);   // white paper

        for (int y = 0; y < Idx.ImageSize; y++)
            for (int x = 0; x < Idx.ImageSize; x++)
            {
                byte value = (byte)(255 - (int)(mnist[y * Idx.ImageSize + x] * 255));

                for (int dy = 0; dy < scale; dy++)
                    for (int dx = 0; dx < scale; dx++)
                        pixels[(margin + y * scale + dy) * side + margin + x * scale + dx] = value;
            }

        string path = Path.Combine(_directory, name);
        File.WriteAllBytes(path, PngWriter.Write(side, side, colorType: 0, pixels));

        return path;
    }

    private static (float X, float Y) Centroid(float[] p)
    {
        double mass = 0, mx = 0, my = 0;

        for (int y = 0; y < Idx.ImageSize; y++)
            for (int x = 0; x < Idx.ImageSize; x++)
            {
                float m = p[y * Idx.ImageSize + x];
                mass += m; mx += m * x; my += m * y;
            }

        return mass <= 0 ? (0, 0) : ((float)(mx / mass), (float)(my / mass));
    }

    /// <summary>Pearson correlation — a shape comparison that tolerates the softening a resample causes.</summary>
    private static float Correlation(float[] a, float[] b)
    {
        float meanA = a.Average(), meanB = b.Average();
        double covariance = 0, varianceA = 0, varianceB = 0;

        for (int i = 0; i < a.Length; i++)
        {
            double da = a[i] - meanA, db = b[i] - meanB;
            covariance += da * db;
            varianceA += da * da;
            varianceB += db * db;
        }

        return (float)(covariance / Math.Sqrt(varianceA * varianceB));
    }

    /// <summary>
    /// The headline: enlarge a digit 6x, invert it, pad it with 40 pixels of margin, save it as a
    /// PNG — and get the same digit back.
    /// </summary>
    [Fact]
    public void A_digit_survives_the_trip_out_to_a_png_and_back()
    {
        float[] original = DigitPreprocessor.ToMnist(
            new GreyImage(Idx.ImageSize, Idx.ImageSize, SyntheticSeven()), invert: false);

        string path = WriteAsPhotoStylePng("seven.png", original, scale: 6, margin: 40);
        float[] recovered = DigitPreprocessor.ToMnist(ImageFile.Load(path));

        Assert.True(Correlation(original, recovered) > 0.9f,
            $"the recovered digit should match the original, correlation was {Correlation(original, recovered):F3}");
    }

    /// <summary>
    /// The same digit photographed at different sizes and offsets must normalize to the same
    /// thing. This is what lets a model trained on MNIST read an image that was never MNIST-shaped.
    /// </summary>
    [Theory]
    [InlineData(3, 10)]
    [InlineData(6, 40)]
    [InlineData(12, 5)]
    [InlineData(4, 120)]
    public void Scale_and_margin_do_not_change_the_normalized_result(int scale, int margin)
    {
        float[] original = DigitPreprocessor.ToMnist(
            new GreyImage(Idx.ImageSize, Idx.ImageSize, SyntheticSeven()), invert: false);

        string path = WriteAsPhotoStylePng($"seven-{scale}-{margin}.png", original, scale, margin);
        float[] recovered = DigitPreprocessor.ToMnist(ImageFile.Load(path));

        Assert.True(Correlation(original, recovered) > 0.85f,
            $"scale {scale}, margin {margin}: correlation {Correlation(original, recovered):F3}");

        (float cx, float cy) = Centroid(recovered);
        Assert.InRange(cx, 12.5f, 15.5f);
        Assert.InRange(cy, 12.5f, 15.5f);
    }

    /// <summary>
    /// Preprocessing must be idempotent: normalizing an already-normalized digit changes nothing
    /// of substance. If it drifted, every extra pass would shrink or shift the digit a little more.
    /// </summary>
    [Fact]
    public void Normalizing_an_already_normalized_digit_is_stable()
    {
        float[] once = DigitPreprocessor.ToMnist(
            new GreyImage(Idx.ImageSize, Idx.ImageSize, SyntheticSeven()), invert: false);

        float[] twice = DigitPreprocessor.ToMnist(
            new GreyImage(Idx.ImageSize, Idx.ImageSize, once), invert: false);

        Assert.True(Correlation(once, twice) > 0.95f,
            $"a second pass should be near-identity, correlation was {Correlation(once, twice):F3}");
    }

    /// <summary>
    /// Automatic polarity detection must produce the same answer as being told explicitly — the
    /// image is dark-on-light, and the border is what gives that away.
    /// </summary>
    [Fact]
    public void Polarity_detection_agrees_with_an_explicit_instruction()
    {
        float[] original = DigitPreprocessor.ToMnist(
            new GreyImage(Idx.ImageSize, Idx.ImageSize, SyntheticSeven()), invert: false);

        GreyImage onPaper = ImageFile.Load(WriteAsPhotoStylePng("polarity.png", original, 6, 30));

        float[] detected = DigitPreprocessor.ToMnist(onPaper);
        float[] told = DigitPreprocessor.ToMnist(onPaper, invert: true);

        Assert.Equal(told, detected);
    }

    /// <summary>
    /// And the whole point: the recovered pixels feed a real network without complaint, and a
    /// softmax classifier gives back a probability distribution over the ten digits.
    /// </summary>
    [Fact]
    public void The_recovered_image_feeds_straight_into_a_classifier()
    {
        float[] original = DigitPreprocessor.ToMnist(
            new GreyImage(Idx.ImageSize, Idx.ImageSize, SyntheticSeven()), invert: false);

        float[] recovered = DigitPreprocessor.ToMnist(
            ImageFile.Load(WriteAsPhotoStylePng("feed.png", original, 6, 40)));

        var net = new Sequential(inputs: Idx.PixelCount)
            .Dense<Tanh>(16)
            .SoftmaxOutput(10)
            .Build(seed: 1);

        ReadOnlySpan<float> p = net.Predict(recovered);

        Assert.Equal(10, p.Length);
        Assert.Equal(1f, p.ToArray().Sum(), 4);
    }
}
