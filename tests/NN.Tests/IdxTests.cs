using NN.Mnist;
using Xunit;

namespace NN.Tests;

/// <summary>
/// The IDX reader, exercised against bytes built in memory — no network, no cached dataset, no
/// 11 MB download. That is the reason <see cref="Idx"/> takes a <see cref="Stream"/> and knows
/// nothing about where the data came from.
///
/// <para>The failure modes matter as much as the happy path here. A misparsed header does not
/// throw on its own: it produces a plausible-looking array of the wrong shape, and the first
/// symptom is a network that trains to 10% accuracy for no visible reason.</para>
/// </summary>
public class IdxTests
{
    /// <summary>Builds an IDX image file: magic 0x0803, count, rows, columns, then raw pixels.</summary>
    private static MemoryStream ImageFile(int count, params byte[] pixels)
    {
        var bytes = new List<byte>();

        bytes.AddRange(BigEndian(0x0803));
        bytes.AddRange(BigEndian(count));
        bytes.AddRange(BigEndian(Idx.ImageSize));
        bytes.AddRange(BigEndian(Idx.ImageSize));
        bytes.AddRange(pixels);

        return new MemoryStream([.. bytes]);
    }

    /// <summary>Builds an IDX label file: magic 0x0801, count, then one byte per label.</summary>
    private static MemoryStream LabelFile(params byte[] labels)
    {
        var bytes = new List<byte>();

        bytes.AddRange(BigEndian(0x0801));
        bytes.AddRange(BigEndian(labels.Length));
        bytes.AddRange(labels);

        return new MemoryStream([.. bytes]);
    }

    /// <summary>IDX headers are big-endian, which is the whole point of these tests.</summary>
    private static byte[] BigEndian(int value) =>
        [(byte)(value >> 24), (byte)(value >> 16), (byte)(value >> 8), (byte)value];

    private static byte[] Pixels(int images, byte fill) =>
        [.. Enumerable.Repeat(fill, images * Idx.PixelCount)];

    [Fact]
    public void Reads_images_into_one_flat_row_major_array()
    {
        var (pixels, count) = Idx.ReadImages(ImageFile(3, Pixels(3, 255)));

        Assert.Equal(3, count);
        Assert.Equal(3 * Idx.PixelCount, pixels.Length);
    }

    /// <summary>
    /// Pixels must arrive in 0–1. Left at 0–255, the first layer's pre-activations would be two
    /// orders of magnitude larger than Xavier initialization assumes, saturating tanh on the very
    /// first forward pass and flattening the gradient to nothing.
    /// </summary>
    [Fact]
    public void Scales_pixels_from_bytes_to_the_unit_interval()
    {
        byte[] raw = Pixels(1, 0);
        raw[0] = 255;
        raw[1] = 128;

        var (pixels, _) = Idx.ReadImages(ImageFile(1, raw));

        Assert.Equal(1f, pixels[0]);
        Assert.Equal(128f / 255f, pixels[1], 5);
        Assert.Equal(0f, pixels[2]);
    }

    /// <summary>
    /// The header is big-endian. Read little-endian, a count of 3 becomes 50,331,648 — so this
    /// asserts the count is exactly what was written, not merely that parsing succeeded.
    /// </summary>
    [Fact]
    public void Interprets_the_header_as_big_endian()
    {
        var (_, imageCount) = Idx.ReadImages(ImageFile(2, Pixels(2, 10)));
        var (_, digits) = Idx.ReadLabels(LabelFile(7, 7));

        Assert.Equal(2, imageCount);
        Assert.Equal(2, digits.Length);
    }

    [Fact]
    public void One_hot_encodes_labels_into_ten_outputs_each()
    {
        var (oneHot, digits) = Idx.ReadLabels(LabelFile(0, 3, 9));

        Assert.Equal([0, 3, 9], digits);
        Assert.Equal(30, oneHot.Length);

        Assert.Equal(1f, oneHot[0]);        // label 0 -> index 0
        Assert.Equal(1f, oneHot[10 + 3]);   // label 3 -> index 3 of the second block
        Assert.Equal(1f, oneHot[20 + 9]);   // label 9 -> index 9 of the third

        Assert.Equal(3, oneHot.Count(v => v == 1f));
        Assert.Equal(27, oneHot.Count(v => v == 0f));
    }

    [Fact]
    public void Respects_a_limit_on_how_much_to_read()
    {
        var (pixels, count) = Idx.ReadImages(ImageFile(5, Pixels(5, 1)), limit: 2);
        var (oneHot, digits) = Idx.ReadLabels(LabelFile(1, 2, 3, 4, 5), limit: 2);

        Assert.Equal(2, count);
        Assert.Equal(2 * Idx.PixelCount, pixels.Length);
        Assert.Equal(2, digits.Length);
        Assert.Equal(20, oneHot.Length);
    }

    [Fact]
    public void A_limit_larger_than_the_file_reads_only_what_is_there()
    {
        var (_, count) = Idx.ReadImages(ImageFile(2, Pixels(2, 1)), limit: 500);

        Assert.Equal(2, count);
    }

    [Fact]
    public void Rejects_a_label_file_passed_where_images_were_expected()
    {
        var ex = Assert.Throws<InvalidDataException>(() => Idx.ReadImages(LabelFile(1, 2, 3)));

        Assert.Contains("IDX image", ex.Message);
    }

    [Fact]
    public void Rejects_an_image_file_passed_where_labels_were_expected()
    {
        var ex = Assert.Throws<InvalidDataException>(() => Idx.ReadLabels(ImageFile(1, Pixels(1, 0))));

        Assert.Contains("IDX label", ex.Message);
    }

    [Fact]
    public void Rejects_a_file_that_is_not_idx_at_all()
    {
        using var stream = new MemoryStream("this is not a dataset"u8.ToArray());

        Assert.Throws<InvalidDataException>(() => Idx.ReadLabels(stream));
    }

    /// <summary>A truncated file must be named as such, not silently yield fewer images.</summary>
    [Fact]
    public void Rejects_a_truncated_image_file()
    {
        // Header claims 4 images; only 2 images' worth of pixels follow.
        var ex = Assert.Throws<InvalidDataException>(
            () => Idx.ReadImages(ImageFile(4, Pixels(2, 200))));

        Assert.Contains("truncated", ex.Message);
    }

    [Fact]
    public void Rejects_a_truncated_header()
    {
        using var stream = new MemoryStream([0x00, 0x00, 0x08]);

        var ex = Assert.Throws<InvalidDataException>(() => Idx.ReadImages(stream));

        Assert.Contains("truncated", ex.Message);
    }

    [Fact]
    public void Rejects_images_that_are_not_28_by_28()
    {
        var bytes = new List<byte>();
        bytes.AddRange(BigEndian(0x0803));
        bytes.AddRange(BigEndian(1));
        bytes.AddRange(BigEndian(16));   // wrong dimensions
        bytes.AddRange(BigEndian(16));
        bytes.AddRange(Enumerable.Repeat((byte)0, 256));

        var ex = Assert.Throws<InvalidDataException>(() => Idx.ReadImages(new MemoryStream([.. bytes])));

        Assert.Contains("28x28", ex.Message);
    }

    [Fact]
    public void Rejects_a_label_that_is_not_a_digit()
    {
        var ex = Assert.Throws<InvalidDataException>(() => Idx.ReadLabels(LabelFile(3, 42)));

        Assert.Contains("42", ex.Message);
    }

    /// <summary>
    /// The shapes the reader produces must be exactly what the network consumes: 784 inputs per
    /// image, 10 targets per label. This is the contract that makes the demo work at all.
    /// </summary>
    [Fact]
    public void Output_shapes_match_what_the_network_expects()
    {
        var (pixels, count) = Idx.ReadImages(ImageFile(4, Pixels(4, 128)));
        var (oneHot, _) = Idx.ReadLabels(LabelFile(1, 2, 3, 4));

        var net = new Sequential(inputs: Idx.PixelCount).Dense<Tanh>(8).Dense<Sigmoid>(10).Build(seed: 1);

        Assert.Equal(net.Inputs * count, pixels.Length);
        Assert.Equal(net.Outputs * count, oneHot.Length);

        // The real check: Train validates lengths against the architecture and would reject a
        // mismatch, so this passing means the reader and the network genuinely agree.
        net.Train(pixels, oneHot, epochs: 1, learningRate: 0.1f, batchSize: 2);
    }
}
