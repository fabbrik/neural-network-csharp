namespace NN;

/// <summary>
/// Saves and loads trained networks.
///
/// <para><b>File format</b> (little-endian, the .NET <see cref="BinaryWriter"/> default):</para>
/// <code>
///   magic     8 bytes   "NNMODEL\0"
///   version   int32     format version (currently 1)
///   layers    int32     number of layers
///   per layer:
///     descriptor  string   length-prefixed UTF-8, e.g. "Dense&lt;Tanh&gt;"
///     inputs      int32
///     units       int32
///     weights     float32 × (inputs × units)
///     biases      float32 × units
/// </code>
///
/// <para>
/// The architecture is stored alongside the weights, so loading needs no prior knowledge of the
/// shape — unlike a bare weight dump, which silently corrupts if the code's architecture drifts.
/// The magic bytes and version let a wrong or outdated file fail with a clear message instead of
/// being read as garbage floats.
/// </para>
///
/// <para>
/// Only trainable parameters are written. Gradient accumulators and the forward-pass activation
/// cache are training scratch space and are rebuilt empty on load.
/// </para>
/// </summary>
public static class ModelIO
{
    private static readonly byte[] Magic = "NNMODEL\0"u8.ToArray();
    private const int CurrentVersion = 1;
    private static readonly object FactoriesLock = new();

    /// <summary>
    /// Maps a layer <see cref="ILayer.Descriptor"/> back to a constructor. A saved file names its
    /// layer types as strings, and this is what turns those names back into real types — C# can't
    /// construct <c>Dense&lt;Tanh&gt;</c> from a string without either this table or reflection,
    /// and an explicit table is both faster and safe against loading arbitrary types from a file.
    /// </summary>
    private static readonly Dictionary<string, Func<int, int, ILayer>> Factories = new()
    {
        ["Dense<Sigmoid>"] = (i, u) => new Dense<Sigmoid>(i, u),
        ["Dense<Tanh>"] = (i, u) => new Dense<Tanh>(i, u),
        ["Dense<ReLU>"] = (i, u) => new Dense<ReLU>(i, u),
        ["Dense<Identity>"] = (i, u) => new Dense<Identity>(i, u),
        ["Dense<Step>"] = (i, u) => new Dense<Step>(i, u),
    };

    /// <summary>
    /// Registers a custom layer type so <see cref="Load(string)"/> can reconstruct it.
    ///
    /// <para>Mutates process-wide state. Access to the registration table is synchronized, but
    /// registering custom layer types during start-up is still the clearest policy: it avoids
    /// load behavior depending on timing.</para>
    /// </summary>
    /// <param name="descriptor">Must match the layer's <see cref="ILayer.Descriptor"/>.</param>
    /// <param name="factory">Builds the layer given (inputs, units).</param>
    public static void Register(string descriptor, Func<int, int, ILayer> factory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(descriptor);
        ArgumentNullException.ThrowIfNull(factory);

        lock (FactoriesLock)
            Factories[descriptor] = factory;
    }

    /// <summary>Writes <paramref name="network"/> to <paramref name="path"/>, overwriting it.</summary>
    public static void Save(Network network, string path)
    {
        ArgumentNullException.ThrowIfNull(network);

        using var stream = File.Create(path);
        Save(network, stream);
    }

    /// <summary>Writes to an open stream — for embedding a model in a larger file or sending it over a network.</summary>
    public static void Save(Network network, Stream stream)
    {
        ArgumentNullException.ThrowIfNull(network);

        using var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true);

        writer.Write(Magic);
        writer.Write(CurrentVersion);
        writer.Write(network.Layers.Count);

        foreach (ILayer layer in network.Layers)
        {
            writer.Write(layer.Descriptor);
            writer.Write(layer.Inputs);
            writer.Write(layer.Units);
            layer.WriteParameters(writer);
        }
    }

    /// <summary>Reads a network previously written by <see cref="Save(Network, string)"/>.</summary>
    public static Network Load(string path)
    {
        using var stream = File.OpenRead(path);
        return Load(stream);
    }

    /// <summary>Reads a network from an open stream.</summary>
    public static Network Load(Stream stream)
    {
        using var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true);

        byte[] magic = reader.ReadBytes(Magic.Length);
        if (!magic.AsSpan().SequenceEqual(Magic))
            throw new InvalidDataException("Not a model file (bad magic bytes).");

        int version = reader.ReadInt32();
        if (version != CurrentVersion)
            throw new InvalidDataException($"Model format version {version} is not supported (expected {CurrentVersion}).");

        int count = reader.ReadInt32();
        if (count <= 0)
            throw new InvalidDataException($"Model declares {count} layers.");

        var layers = new ILayer[count];

        for (int i = 0; i < count; i++)
        {
            string descriptor;
            int inputs, units;

            try
            {
                descriptor = reader.ReadString();
                inputs = reader.ReadInt32();
                units = reader.ReadInt32();
            }
            catch (EndOfStreamException e)
            {
                throw new InvalidDataException($"Model file is truncated: ran out of data at layer {i} of {count}.", e);
            }

            Func<int, int, ILayer>? factory;
            lock (FactoriesLock)
                Factories.TryGetValue(descriptor, out factory);

            if (factory is null)
                throw new InvalidDataException(
                    $"Unknown layer type '{descriptor}' in layer {i}. Call ModelIO.Register to add it.");

            ILayer layer = factory(inputs, units);

            try
            {
                layer.ReadParameters(reader);
            }
            catch (EndOfStreamException e)
            {
                throw new InvalidDataException(
                    $"Model file is truncated: layer {i} ({descriptor}) needs {layer.ParameterCount} parameters.", e);
            }

            layers[i] = layer;
        }

        // Must skip initialization — the public constructor randomizes weights, which would
        // silently discard everything just read from the file.
        return Network.FromTrainedLayers(layers);
    }
}
