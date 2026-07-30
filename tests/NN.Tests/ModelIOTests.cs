using Xunit;

namespace NN.Tests;

public class ModelIOTests
{
    private static readonly float[] Inputs = [0, 0, 0, 1, 1, 0, 1, 1];
    private static readonly float[] Targets = [0, 1, 1, 0];

    private static Network TrainedXor()
    {
        var net = new Sequential(inputs: 2).Dense<Tanh>(4).Dense<Sigmoid>(1).Build(seed: 42);
        net.Train(Inputs, Targets, epochs: 500, learningRate: 0.5f);
        return net;
    }

    /// <summary>
    /// The regression test for the bug where loading ran weight initialization and silently
    /// discarded everything it had just read. It would not crash — predictions would just be
    /// those of an untrained network.
    /// </summary>
    [Fact]
    public void Round_trip_preserves_predictions_exactly()
    {
        Network original = TrainedXor();

        using var stream = new MemoryStream();
        ModelIO.Save(original, stream);
        stream.Position = 0;
        Network loaded = ModelIO.Load(stream);

        for (int i = 0; i < Targets.Length; i++)
        {
            var x = Inputs.AsSpan(i * 2, 2);
            Assert.Equal(original.Predict(x)[0], loaded.Predict(x)[0]);
        }
    }

    [Fact]
    public void Round_trip_preserves_every_parameter_and_the_architecture()
    {
        Network original = TrainedXor();

        using var stream = new MemoryStream();
        ModelIO.Save(original, stream);
        stream.Position = 0;
        Network loaded = ModelIO.Load(stream);

        Assert.Equal(original.Layers.Count, loaded.Layers.Count);
        Assert.Equal(original.ParameterCount, loaded.ParameterCount);

        for (int l = 0; l < original.Layers.Count; l++)
        {
            ILayer a = original.Layers[l], b = loaded.Layers[l];

            Assert.Equal(a.Descriptor, b.Descriptor);
            Assert.Equal(a.Inputs, b.Inputs);
            Assert.Equal(a.Units, b.Units);

            for (int p = 0; p < a.ParameterCount; p++)
                Assert.Equal(a.GetParameter(p), b.GetParameter(p));
        }
    }

    [Fact]
    public void A_loaded_model_can_be_trained_further()
    {
        using var stream = new MemoryStream();
        ModelIO.Save(TrainedXor(), stream);
        stream.Position = 0;

        Network loaded = ModelIO.Load(stream);
        float loss = loaded.Train(Inputs, Targets, epochs: 2000, learningRate: 0.5f);

        Assert.True(loss < 0.01f, $"expected further training to converge, got {loss:F6}");
    }

    [Fact]
    public void Save_and_load_via_a_file_path_works()
    {
        string path = Path.Combine(Path.GetTempPath(), $"nn-test-{Guid.NewGuid():N}.nnm");

        try
        {
            Network original = TrainedXor();
            ModelIO.Save(original, path);

            Assert.True(File.Exists(path));

            Network loaded = ModelIO.Load(path);
            var x = Inputs.AsSpan(0, 2);
            Assert.Equal(original.Predict(x)[0], loaded.Predict(x)[0]);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Loading_a_non_model_file_fails_with_a_clear_message()
    {
        using var stream = new MemoryStream("this is not a model, it is a text file"u8.ToArray());

        var ex = Assert.Throws<InvalidDataException>(() => ModelIO.Load(stream));
        Assert.Contains("magic", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Loading_an_unsupported_version_fails_with_a_clear_message()
    {
        using var stream = new MemoryStream();
        ModelIO.Save(TrainedXor(), stream);

        byte[] bytes = stream.ToArray();
        bytes[8] = 99;   // the version field, straight after the 8 magic bytes

        var ex = Assert.Throws<InvalidDataException>(() => ModelIO.Load(new MemoryStream(bytes)));
        Assert.Contains("version", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Loading_a_truncated_file_fails_with_a_clear_message()
    {
        using var stream = new MemoryStream();
        ModelIO.Save(TrainedXor(), stream);

        byte[] truncated = stream.ToArray()[..40];

        var ex = Assert.Throws<InvalidDataException>(() => ModelIO.Load(new MemoryStream(truncated)));
        Assert.Contains("truncated", ex.Message, StringComparison.OrdinalIgnoreCase);
    }
}
