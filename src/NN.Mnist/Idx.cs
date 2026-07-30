namespace NN.Mnist;

/// <summary>
/// Reader for the IDX file format that MNIST ships in.
///
/// <para>The format is about as simple as a binary format gets, which is why it's worth parsing by
/// hand rather than taking a dependency:</para>
/// <code>
///   magic       int32     0x00000803 for images (3 dimensions), 0x00000801 for labels (1)
///   dim 0       int32     count
///   dim 1..n    int32     one per remaining dimension (28, 28 for images)
///   data        uint8[]   row-major, one byte per pixel or per label
/// </code>
///
/// <para><b>Every integer is big-endian</b> — the format predates x86's dominance. .NET runs
/// little-endian on every platform this project targets, so each header field needs its bytes
/// reversed. Reading them raw is the classic way to get a "60000-image" file that claims to
/// contain 1,745,946,112 images.</para>
///
/// <para>Parsing is separated from downloading on purpose: these methods take a
/// <see cref="Stream"/> and never touch the network or the disk, so the test suite can exercise
/// the format — including its failure modes — against a handful of bytes in memory.</para>
/// </summary>
public static class Idx
{
    private const int ImageMagic = 0x0803;
    private const int LabelMagic = 0x0801;

    /// <summary>Pixels per MNIST image edge. The format carries this, but it is checked against it.</summary>
    public const int ImageSize = 28;

    /// <summary>Length of one flattened image, and the input width of a network that reads them.</summary>
    public const int PixelCount = ImageSize * ImageSize;

    /// <summary>
    /// Reads an IDX image file into one flat row-major array: image <c>i</c> occupies
    /// <c>result[i * PixelCount ..]</c>, exactly the layout <see cref="Network.Train"/> expects.
    /// </summary>
    /// <param name="stream">An IDX3 stream, already decompressed.</param>
    /// <param name="limit">Read at most this many images; 0 or negative reads all of them.</param>
    /// <returns>Pixels scaled from 0–255 to 0–1, and the number of images read.</returns>
    public static (float[] Pixels, int Count) ReadImages(Stream stream, int limit = 0)
    {
        ArgumentNullException.ThrowIfNull(stream);

        using var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true);

        int magic = ReadBigEndianInt32(reader);
        if (magic != ImageMagic)
            throw new InvalidDataException(
                $"Not an IDX image file: expected magic 0x{ImageMagic:X4}, got 0x{magic:X4}.");

        int count = ReadBigEndianInt32(reader);
        int rows = ReadBigEndianInt32(reader);
        int columns = ReadBigEndianInt32(reader);

        if (rows != ImageSize || columns != ImageSize)
            throw new InvalidDataException($"Expected {ImageSize}x{ImageSize} images, got {rows}x{columns}.");

        if (limit > 0) count = Math.Min(count, limit);

        var pixels = new float[count * PixelCount];
        byte[] raw = reader.ReadBytes(count * PixelCount);

        if (raw.Length != pixels.Length)
            throw new InvalidDataException(
                $"File is truncated: needed {pixels.Length} pixel bytes for {count} images, got {raw.Length}.");

        // Scale to 0–1. Left at 0–255, the first layer's pre-activations would be ~255x larger
        // than Xavier initialization assumes, saturating tanh immediately and killing the gradient.
        for (int i = 0; i < raw.Length; i++)
            pixels[i] = raw[i] / 255f;

        return (pixels, count);
    }

    /// <summary>
    /// Reads an IDX label file and one-hot encodes it: label <c>i</c> becomes ten floats at
    /// <c>result[i * 10 ..]</c>, all zero but for a 1 at the digit's index.
    /// </summary>
    /// <param name="stream">An IDX1 stream, already decompressed.</param>
    /// <param name="limit">Read at most this many labels; 0 or negative reads all of them.</param>
    /// <returns>The one-hot targets and the raw digits, which are handier for reporting.</returns>
    public static (float[] OneHot, byte[] Digits) ReadLabels(Stream stream, int limit = 0)
    {
        ArgumentNullException.ThrowIfNull(stream);

        using var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true);

        int magic = ReadBigEndianInt32(reader);
        if (magic != LabelMagic)
            throw new InvalidDataException(
                $"Not an IDX label file: expected magic 0x{LabelMagic:X4}, got 0x{magic:X4}.");

        int count = ReadBigEndianInt32(reader);
        if (limit > 0) count = Math.Min(count, limit);

        byte[] digits = reader.ReadBytes(count);

        if (digits.Length != count)
            throw new InvalidDataException(
                $"File is truncated: needed {count} label bytes, got {digits.Length}.");

        // One-hot, because the network has ten independent outputs and MSE compares each against
        // its target. "The answer is 7" becomes "output 7 should be 1, the other nine should be 0."
        var oneHot = new float[count * 10];

        for (int i = 0; i < count; i++)
        {
            if (digits[i] > 9)
                throw new InvalidDataException($"Label {i} is {digits[i]}, which is not a digit.");

            oneHot[i * 10 + digits[i]] = 1f;
        }

        return (oneHot, digits);
    }

    /// <summary>
    /// Reads a big-endian int32. <see cref="BinaryReader.ReadInt32"/> is little-endian, so the
    /// bytes are reversed rather than reinterpreted.
    /// </summary>
    private static int ReadBigEndianInt32(BinaryReader reader)
    {
        byte[] b = reader.ReadBytes(4);

        if (b.Length != 4)
            throw new InvalidDataException("File is truncated: ran out of data inside the header.");

        return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3];
    }
}
