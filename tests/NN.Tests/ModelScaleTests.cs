using NN.Mnist;
using Xunit;

namespace NN.Tests;

/// <summary>
/// Model persistence at the scale the MNIST demo actually uses: 101,770 parameters across a
/// 784-128-10 stack, rather than XOR's 17.
///
/// <para>The existing round-trip tests prove the format works. These prove it still works when
/// there is enough data for an off-by-one in the layout to hide — with 17 parameters a
/// mis-serialized weight is almost certain to change a prediction visibly, while with 101,770 a
/// single misplaced value can shift accuracy by a fraction of a percent and look like noise.</para>
/// </summary>
public class ModelScaleTests
{
    private static Network MnistShaped(int seed = 42) =>
        new Sequential(inputs: Idx.PixelCount)
            .Dense<Tanh>(128)
            .Dense<Sigmoid>(10)
            .Build(seed);

    private static float[] SyntheticImage(int seed)
    {
        var rng = new Random(seed);
        var image = new float[Idx.PixelCount];

        for (int i = 0; i < image.Length; i++) image[i] = rng.NextSingle();

        return image;
    }

    [Fact]
    public void A_mnist_sized_model_round_trips_every_parameter()
    {
        Network original = MnistShaped();

        using var stream = new MemoryStream();
        ModelIO.Save(original, stream);
        stream.Position = 0;
        Network loaded = ModelIO.Load(stream);

        Assert.Equal(101_770, original.ParameterCount);
        Assert.Equal(original.ParameterCount, loaded.ParameterCount);

        for (int l = 0; l < original.Layers.Count; l++)
            for (int p = 0; p < original.Layers[l].ParameterCount; p++)
                Assert.Equal(original.Layers[l].GetParameter(p), loaded.Layers[l].GetParameter(p));
    }

    /// <summary>
    /// What the demo asserts at runtime after saving: predictions must be bit-for-bit identical,
    /// not merely close. "Close" is what a subtly broken serializer produces.
    /// </summary>
    [Fact]
    public void Predictions_are_bit_for_bit_identical_across_a_round_trip()
    {
        Network original = MnistShaped();

        using var stream = new MemoryStream();
        ModelIO.Save(original, stream);
        stream.Position = 0;
        Network loaded = ModelIO.Load(stream);

        for (int i = 0; i < 50; i++)
        {
            float[] image = SyntheticImage(i);

            // Predict lends out a buffer the next call overwrites, so copy before comparing.
            float[] before = original.Predict(image).ToArray();
            float[] after = loaded.Predict(image).ToArray();

            Assert.Equal(before, after);
        }
    }

    /// <summary>
    /// The file should be float32 per parameter plus a small header — the property the demo
    /// reports. A format that silently widened to double would still round-trip correctly and
    /// quietly double every model file on disk.
    /// </summary>
    [Fact]
    public void File_size_is_four_bytes_per_parameter_plus_a_small_header()
    {
        using var stream = new MemoryStream();
        ModelIO.Save(MnistShaped(), stream);

        long expected = 101_770 * 4L;
        long overhead = stream.Length - expected;

        Assert.True(overhead is > 0 and < 256,
            $"expected a small header over {expected:N0} bytes of float32, got {overhead} bytes");
    }

    /// <summary>
    /// Loading reconstructs the architecture from the file alone. This is what lets the demo skip
    /// training without being told the shape it is loading.
    /// </summary>
    [Fact]
    public void The_architecture_is_recovered_from_the_file_without_being_told()
    {
        using var stream = new MemoryStream();
        ModelIO.Save(MnistShaped(), stream);
        stream.Position = 0;

        Network loaded = ModelIO.Load(stream);

        Assert.Equal(Idx.PixelCount, loaded.Inputs);
        Assert.Equal(10, loaded.Outputs);
        Assert.Equal(2, loaded.Layers.Count);
        Assert.Equal("Dense<Tanh>", loaded.Layers[0].Descriptor);
        Assert.Equal(128, loaded.Layers[0].Units);
        Assert.Equal("Dense<Sigmoid>", loaded.Layers[1].Descriptor);
    }

    /// <summary>A reloaded model must be usable for further training, not just inference.</summary>
    [Fact]
    public void A_reloaded_mnist_model_can_still_be_trained()
    {
        using var stream = new MemoryStream();
        ModelIO.Save(MnistShaped(), stream);
        stream.Position = 0;

        Network loaded = ModelIO.Load(stream);

        var x = new float[4 * Idx.PixelCount];
        var y = new float[4 * 10];

        for (int i = 0; i < 4; i++)
        {
            SyntheticImage(i).CopyTo(x, i * Idx.PixelCount);
            y[i * 10 + i] = 1f;
        }

        float before = loaded.Loss(x.AsSpan(0, Idx.PixelCount), y.AsSpan(0, 10));
        loaded.Train(x, y, epochs: 50, learningRate: 0.5f, batchSize: 2);
        float after = loaded.Loss(x.AsSpan(0, Idx.PixelCount), y.AsSpan(0, 10));

        Assert.True(after < before, $"further training should reduce loss: {before:F6} -> {after:F6}");
    }
}
