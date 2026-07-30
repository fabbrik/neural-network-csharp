namespace NN.Mnist;

/// <summary>
/// Converts an arbitrary image into the 784 floats an MNIST-trained network expects.
///
/// <para><b>This class matters more than it looks, and skipping it is the single most common
/// reason a hand-rolled digit recognizer "works on MNIST but not on my own images."</b> The
/// network never learned "what a 7 looks like" in general. It learned what a 7 looks like <i>under
/// MNIST's very specific conventions</i>, and an image that violates them is, as far as the
/// network is concerned, not a digit at all.</para>
///
/// <para>Those conventions, from the dataset's own description, are:</para>
/// <list type="number">
///   <item><description><b>White ink on a black background</b> — the opposite of pen on paper.</description></item>
///   <item><description><b>The digit fits in a 20×20 box</b>, scaled with its aspect ratio kept.</description></item>
///   <item><description><b>Centred in 28×28 by centre of mass</b>, not by bounding box.</description></item>
/// </list>
///
/// <para>Get any one wrong and accuracy collapses in a way that looks like a broken model. This
/// is what people mean by "most of machine learning is data preparation": the 101,770 trained
/// parameters are worthless without the hundred lines below that put the input in the shape they
/// were trained on.</para>
/// </summary>
public static class DigitPreprocessor
{
    /// <summary>Side of the box the digit is scaled to fit, inside the 28×28 canvas.</summary>
    private const int DigitBox = 20;

    /// <summary>Ink below this intensity is treated as background when finding the digit.</summary>
    private const float InkThreshold = 0.15f;

    /// <summary>
    /// Produces a 784-float MNIST-convention image from an arbitrary greyscale one.
    /// </summary>
    /// <param name="image">The source image, any size.</param>
    /// <param name="invert">
    /// Whether the source is dark ink on a light background and must be inverted. Null
    /// auto-detects from the border, which is right for essentially every photo or drawing.
    /// </param>
    /// <returns>Pixels in the same layout and range as <see cref="Idx.ReadImages"/> produces.</returns>
    public static float[] ToMnist(GreyImage image, bool? invert = null)
    {
        ArgumentNullException.ThrowIfNull(image);

        // ── 1. Polarity. MNIST is white-on-black; a scan or screenshot is almost always the
        //       reverse. Sampling the border rather than the whole image is deliberate: the mean
        //       of a thick digit on white paper can be darker than you would guess, but the edge
        //       of the frame is background nearly by definition.
        bool shouldInvert = invert ?? BorderIsLight(image);

        var ink = new float[image.Width * image.Height];
        for (int i = 0; i < ink.Length; i++)
            ink[i] = shouldInvert ? 1f - image.Pixels[i] : image.Pixels[i];

        // ── 2. Find the digit. Everything outside its bounding box is padding that would
        //       otherwise shrink the digit when scaled — a 28×28 digit centred in a 1000×1000
        //       photo becomes a smudge two pixels across.
        (int minX, int minY, int maxX, int maxY) = InkBounds(ink, image.Width, image.Height);

        if (maxX < minX || maxY < minY)
            return new float[Idx.PixelCount];   // no ink at all: a blank image, not an error

        int boxWidth = maxX - minX + 1;
        int boxHeight = maxY - minY + 1;

        // ── 3. Scale into a 20×20 box, preserving the aspect ratio. Stretching a 1 to fill the
        //       square would make it look like an 8's worth of ink in the wrong places.
        float scale = (float)DigitBox / Math.Max(boxWidth, boxHeight);
        int scaledWidth = Math.Max(1, (int)MathF.Round(boxWidth * scale));
        int scaledHeight = Math.Max(1, (int)MathF.Round(boxHeight * scale));

        float[] scaled = Resample(ink, image.Width, minX, minY, boxWidth, boxHeight, scaledWidth, scaledHeight);

        // ── 4. Centre by centre of mass in the 28×28 canvas. MNIST centres by the ink's centroid
        //       rather than its bounding box, and the difference is real: a 7 with a long
        //       descending stroke has its mass high and its box centre low.
        var canvas = new float[Idx.PixelCount];

        (float centroidX, float centroidY) = Centroid(scaled, scaledWidth, scaledHeight);

        int offsetX = (int)MathF.Round(Idx.ImageSize / 2f - centroidX);
        int offsetY = (int)MathF.Round(Idx.ImageSize / 2f - centroidY);

        for (int y = 0; y < scaledHeight; y++)
        {
            int targetY = y + offsetY;
            if (targetY < 0 || targetY >= Idx.ImageSize) continue;

            for (int x = 0; x < scaledWidth; x++)
            {
                int targetX = x + offsetX;
                if (targetX < 0 || targetX >= Idx.ImageSize) continue;

                canvas[targetY * Idx.ImageSize + targetX] = scaled[y * scaledWidth + x];
            }
        }

        return canvas;
    }

    /// <summary>
    /// True when the image's border is lighter than mid-grey, meaning dark ink on light paper and
    /// therefore in need of inversion.
    /// </summary>
    private static bool BorderIsLight(GreyImage image)
    {
        double total = 0;
        int count = 0;

        for (int x = 0; x < image.Width; x++)
        {
            total += image[x, 0] + image[x, image.Height - 1];
            count += 2;
        }

        for (int y = 0; y < image.Height; y++)
        {
            total += image[0, y] + image[image.Width - 1, y];
            count += 2;
        }

        return total / count > 0.5;
    }

    /// <summary>Tightest box containing pixels above <see cref="InkThreshold"/>.</summary>
    private static (int MinX, int MinY, int MaxX, int MaxY) InkBounds(float[] ink, int width, int height)
    {
        int minX = width, minY = height, maxX = -1, maxY = -1;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                if (ink[y * width + x] < InkThreshold) continue;

                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }

        return (minX, minY, maxX, maxY);
    }

    /// <summary>
    /// Box-filter resampling: each destination pixel averages every source pixel that falls inside
    /// it. Nearest-neighbour would be simpler and is wrong here — downscaling a 400-pixel-wide
    /// photo by picking one pixel in fourteen drops most of the stroke and leaves a dotted,
    /// broken digit. Averaging preserves the stroke as the soft grey edges MNIST itself has.
    /// </summary>
    private static float[] Resample(
        float[] source, int sourceStride, int originX, int originY,
        int sourceWidth, int sourceHeight, int targetWidth, int targetHeight)
    {
        var target = new float[targetWidth * targetHeight];

        for (int y = 0; y < targetHeight; y++)
        {
            // The source rows this destination row covers.
            int y0 = originY + y * sourceHeight / targetHeight;
            int y1 = Math.Max(y0 + 1, originY + (y + 1) * sourceHeight / targetHeight);

            for (int x = 0; x < targetWidth; x++)
            {
                int x0 = originX + x * sourceWidth / targetWidth;
                int x1 = Math.Max(x0 + 1, originX + (x + 1) * sourceWidth / targetWidth);

                float total = 0;
                int count = 0;

                for (int sy = y0; sy < y1; sy++)
                {
                    for (int sx = x0; sx < x1; sx++)
                    {
                        total += source[sy * sourceStride + sx];
                        count++;
                    }
                }

                target[y * targetWidth + x] = count == 0 ? 0f : total / count;
            }
        }

        return target;
    }

    /// <summary>
    /// Intensity-weighted centre of the ink — the average position of every pixel, each counted
    /// in proportion to how bright it is.
    /// </summary>
    private static (float X, float Y) Centroid(float[] pixels, int width, int height)
    {
        double totalMass = 0, weightedX = 0, weightedY = 0;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float mass = pixels[y * width + x];

                totalMass += mass;
                weightedX += mass * x;
                weightedY += mass * y;
            }
        }

        // No ink: fall back to the geometric centre so the caller still gets something sensible.
        if (totalMass <= 0) return (width / 2f, height / 2f);

        return ((float)(weightedX / totalMass), (float)(weightedY / totalMass));
    }
}
