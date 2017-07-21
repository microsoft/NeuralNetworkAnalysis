// --------------------------------------------------------------------------------------------------
// Neural Network Analysis Framework
//
// Copyright(c) Microsoft Corporation
// All rights reserved.
//
// MIT License
//  
//  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
//  associated documentation files (the "Software"), to deal in the Software without restriction,
//  including without limitation the rights to use, copy, modify, merge, publish, distribute,
//  sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included in all copies or
//  substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
//  NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
//  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
//  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// --------------------------------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;

using System.Drawing;
using System.Drawing.Imaging;
using System.Drawing.Printing;
using System.IO;

using System.Diagnostics;

using Mono.Options;

namespace NNAnalysis
{
    using M = Math;

    namespace Utils
    {
        public static class UArray
        {
            public static double[] ToDoubleArray(float[] point)
            {
                double[] result = new double[point.Length];
                for (int i = 0; i < point.Length; i++)
                {
                    result[i] = point[i];
                }
                return result;
            }

            public static float[] ToFloatArray(double[] point)
            {
                float[] result = new float[point.Length];
                for (int i = 0; i < point.Length; i++)
                {
                    result[i] = (float)point[i];
                }
                return result;
            }

            public static double[] ToDoubleArray(int[] point)
            {
                double[] result = new double[point.Length];
                for (int i = 0; i < point.Length; i++)
                {
                    result[i] = point[i];
                }
                return result;
            }

            // NB: Python and our script treat 'float' arrays as C# doubles!
            // Hence 8 byte offets!
            public static double[] ToDoubleArrayFromDoubleBytes(byte[] point)
            {

                double[] result = new double[point.Length / 8];

                for (int n = 0; n < point.Length; n += 8)
                {
                    result[n/8] = (double) BitConverter.ToDouble(point, n);

                }
                return result;
            }

            public static double[] ToDoubleArrayFromInt8Bytes(byte[] point)
            {

                double[] result = new double[point.Length];
                for (int i = 0; i < point.Length; i++)
                {
                    result[i] = point[i];
                }
                return result;
            }

            public static double[] ToDoubleArray(byte[] point, int sourceIndex, int length)
            {
                double[] result = new double[length];
                for (int i = 0; i < length; i++)
                {
                    result[i] = point[sourceIndex + i];
                }
                return result;
            }

            public static byte[] ToByteArray(double[] point)
            {
                byte[] bytes = new byte[point.Length];
                for (int i = 0; i < point.Length; i++)
                {
                    bytes[i] = point[i] > 255f ? (byte)255 : (point[i] < 0f ? (byte)0 : (byte)M.Round(point[i]));
                }
                return bytes;
            }

            public static int[] ToIntArray(double[] array)
            {
                int[] intArray = new int[array.Length];
                for (int i = 0; i < array.Length; i++)
                {
                    intArray[i] = (int)array[i];
                }
                return intArray;
            }


            /// <summary>
            /// output[i] = input[i]*scale + offset
            /// </summary>
            /// <param name="array"></param>
            /// <param name="scale"></param>
            /// <param name="offset"></param>
            /// <returns></returns>
            public static int[] ToRGBArray(double[] array, double scale, double offset)
            {
                int[] result = new int[array.Length];
                for (int i=0; i < array.Length; i++)
                {
                    result[i] = (int)UMath.Clamp(array[i] * scale + offset, 0.0, 255.0);
                }
                return result;
            }


            public static int[] ToIntArray(byte[] array)
            {
                int[] intArray = new int[array.Length];
                for (int i = 0; i < array.Length; i++)
                {
                    intArray[i] = array[i];
                }
                return intArray;
            }

            public static void InPlaceRoundDoubleArray(double[] array)
            {
                for (int i = 0; i < array.Length; i++)
                {
                    array[i] = (int) Math.Round(array[i]);
                }
            }

            public static int ComputeRoundIdenticals(double[] oldarr, double[] newarr)
            {
                int samcount = 0;
                for (int i = 0; i < oldarr.Length; i++)
                {
                    if ((int) Math.Round(oldarr[i]) == (int) Math.Round(newarr[i]))
                    {
                        samcount++;
                    }
                }
                return samcount;
            }



        }
        public static class UMath
        {

            /// <summary>
            /// In-place soft max
            /// </summary>
            /// <param name="input"></param>
            public static void SoftMax(double[] input)
            {
                double max = input[0];
                double min = input[0];
                for (int i = 0; i < input.Length; i++)
                {
                    if (input[i] > max)
                    {
                        max = input[i];
                    }
                }

                double k = max - 4;

                for (int i = 0; i < input.Length; i++)
                {
                    input[i] = M.Exp(input[i] - k);
                }
                double sum = 0;
                for (int i = 0; i < input.Length; i++)
                {
                    sum += input[i];
                }
                for (int i = 0; i < input.Length; i++)
                {
                    input[i] /= sum;
                }
            }

            /// <summary>
            /// Rounds a double and ensures it was an integer
            /// </summary>
            /// <param name="value">The double to be converted</param>
            /// <returns>The integer represented by the double</returns>
            public static int EnsureInt(double value)
            {
                int intValue = (int)M.Round(value);
                if (value != (double)intValue)
                {
                    throw new SystemException("Invalid integer: " + value);
                }
                return intValue;

            }

            /// <summary>
            /// Converts an entire array to integers, ensuring their format
            /// </summary>        
            public static int[] EnsureIntArray(double[] array)
            {
                int[] integerArray = new int[array.Length];
                for (int i = 0; i < array.Length; i++)
                {
                    integerArray[i] = EnsureInt(array[i]);
                }
                return integerArray;
            }

            public static Tuple<int, int> Max(int[] output)
            {
                int max = output[0];
                int maxIndex = 0;
                for (int i = 1; i < output.Length; i++)
                {
                    if (output[i] > max)
                    {
                        max = output[i];
                        maxIndex = i;
                    }
                }
                return new Tuple<int, int>(max, maxIndex);
            }

            public static Tuple<float, int> Max(float[] output)
            {
                float max = output[0];
                int maxIndex = 0;
                for (int i = 1; i < output.Length; i++)
                {
                    if (output[i] > max)
                    {
                        max = output[i];
                        maxIndex = i;
                    }
                }
                return new Tuple<float, int>(max, maxIndex);
            }

            public static Tuple<double, int> Max(double[] output)
            {
                double max = output[0];
                int maxIndex = 0;
                for (int i = 1; i < output.Length; i++)
                {
                    if (output[i] > max)
                    {
                        max = output[i];
                        maxIndex = i;
                    }
                }
                return new Tuple<double, int>(max, maxIndex);
            }

            public static Tuple<double, int> MaxExcluding(int idx, double[] output)
            {
                double max = (idx == 0) ? output[1] : output[0];
                int maxIndex = (idx == 0) ? 1 : 0;
                for (int i = (maxIndex + 1); i < output.Length; i++)
                {
                    if (i == idx) continue; // excluded index

                    if (output[i] > max)
                    {
                        max = output[i];
                        maxIndex = i;
                    }
                }
                return new Tuple<double, int>(max, maxIndex);
            }


            public static double Clamp(double value, double min, double max)
            {
                return (value < min) ? min : ((value > max) ? max : value);
            }

            public static int Clamp(int value, int min, int max)
            {
                return (value < min) ? min : ((value > max) ? max : value);
            }

            public static double[] ClampArray(double[] values, double min, double max)
            {
                double[] newValues = new double[values.Length];
                for (int i = 0; i < values.Length; i++)
                {
                    newValues[i] = Clamp(values[i], min, max);
                }
                return newValues;
            }

            public static int[] ClampArray(int[] values, int min, int max)
            {
                int[] newValues = new int[values.Length];
                for (int i = 0; i < values.Length; i++)
                {
                    newValues[i] = Clamp(values[i], min, max);
                }
                return newValues;
            }

            /// <summary>
            /// Calculates the LInfinity distance between two points in Rn
            /// </summary>
            public static double LInfinityDistance(double[] point1, double[] point2)
            {
                if (point1.Length != point2.Length)
                {
                    throw new SystemException("Invalid inputs!");
                }
                double max = M.Abs(point1[0] - point2[0]);
                for (int i = 1; i < point1.Length; i++)
                {
                    double cur = M.Abs(point1[i] - point2[i]);
                    if (cur > max)
                    {
                        max = cur;
                    }
                }
                return max;
            }

            public static double L1Distance(double[] point1, double[] point2)
            {
                if (point1.Length != point2.Length)
                {
                    throw new SystemException("Invalid inputs!");
                }
                double curr = M.Abs(point1[0] - point2[0]);
                for (int i = 1; i < point1.Length; i++)
                {
                    curr += M.Abs(point1[i] - point2[i]);
                }
                return (curr / (double)point1.Length);
            }

        }

        /// <summary>
        /// Various functions that utilize randomness
        /// </summary>
        public static class URand
        {
            /// <summary>
            /// Returns a double drawn from a Gaussian distribution(0,1)
            /// </summary>
            public static double NextGaussian(Random random)
            {
                double u1 = random.NextDouble();
                double u2 = random.NextDouble();
                return M.Sqrt(-2.0 * M.Log(u1)) * M.Sin(2.0 * M.PI * u2);
            }

            public static double[] NextRandomImage(Random random, int size)
            {
                double[] result = new double[size];
                for (int i = 0; i < size; i++)
                {
                    result[i] = M.Round(255.0 * random.NextDouble());
                }
                return result;
            }


            /// <summary>
            ///  Draws a double from a Gaussian distribution weith a specific mean and deviation
            /// </summary>
            /// <returns></returns>
            public static double NextGaussian(double mean, double sd, Random random)
            {
                return sd * NextGaussian(random) + mean;
            }



            /// <summary>
            /// Standard Fisher-Yates random permutation
            /// </summary>
            /// <param name="random"></param>
            /// <param name="length"></param>
            /// <returns></returns>
            /// 
            public static int[] NextPermutation(Random random, int length)
            {
                int[] list = new int[length];
                for (int i = 0; i < length; i++)
                {
                    list[i] = i;
                }
                int n = length;
                for (int i = length - 1; i > 0; i--)
                {
                    // swap randomly with element in (i, length]
                    int k = random.Next(i, length);
                    int bucket = list[k];
                    list[k] = list[i];
                    list[i] = bucket;
                }
                return list;
            }
            public static double[] GetNoisyPoint(double[] point, double addedNoiseSD, Random random)
            {
                double[] newPoint = new double[point.Length];
                double addedNoise = Utils.URand.NextGaussian(0.0, addedNoiseSD, random);
                for (int j = 0; j < point.Length; j++)
                {
                    newPoint[j] = M.Min(255.0, M.Max(0.0, point[j] + addedNoise));
                }
                return newPoint;
            }
        }

        /// <summary>
        /// Displaying images
        /// </summary>
        public static class UDraw
        {
            public static Bitmap DrawGrayscalePixels(int[] pixels, int numRows, int numCols, bool isRowOrder = true)
            {
                Bitmap image = new Bitmap(numRows, numCols);
                for (int i = 0; i < numRows; i++)
                {
                    for (int j = 0; j < numCols; j++)
                    {
                        int greyScale = isRowOrder ? pixels[numRows * j + i] : pixels[numCols * i + j];
                        Color c = Color.FromArgb(255, greyScale, greyScale, greyScale);
                        image.SetPixel(i, j, c);
                    }
                }
                return image;
            }

            public static Bitmap DrawRGBPixels(int[] pixels, int numRows, int numCols, bool isRowOrder = true)
            {
                Bitmap image = new Bitmap(numRows, numCols);
                int colorOffset = numRows * numCols;
                for (int i = 0; i < numRows; i++)
                {
                    for (int j = 0; j < numCols; j++)
                    {
                        int pixelOffset = isRowOrder ? numRows * j + i : numCols * i + j;
                        Color c = Color.FromArgb(255, pixels[pixelOffset], pixels[pixelOffset + colorOffset], pixels[pixelOffset + 2 * colorOffset]);
                        image.SetPixel(i, j, c);
                    }
                }
                return image;
            }
            public static void DisplayImageAndPause(int[] imagePixels, int numRows, int numCols, bool isColor, bool isRowOrder = true)
            {
                Bitmap image = isColor ? DrawRGBPixels(imagePixels, numRows, numCols, isRowOrder) : DrawGrayscalePixels(imagePixels, numRows, numCols, isRowOrder);
                string temporaryPath = Path.Combine(System.IO.Path.GetTempPath(), "visualization.png");
                image.Save(temporaryPath);
                Thread.Sleep(800);
                System.Diagnostics.Process.Start(temporaryPath);
                Console.WriteLine("Hit enter to continue...");
                Console.ReadLine();
            }


            public static int[] Rotate(int[] imagePixels, int numRows, int numCols, bool isColor, float degrees, bool isRowOrder = true)
            {


                Bitmap image_0 = isColor ? DrawRGBPixels(imagePixels, numRows, numCols, isRowOrder) : DrawGrayscalePixels(imagePixels, numRows, numCols, isRowOrder);
                Graphics g_0 = Graphics.FromImage(image_0);
                g_0.RotateTransform(degrees);
                g_0.DrawImage(image_0, new Point(0, 0));
                
                Bitmap image = isColor ? DrawRGBPixels(imagePixels, numRows, numCols, isRowOrder) : DrawGrayscalePixels(imagePixels, numRows, numCols, isRowOrder);

                Graphics g = Graphics.FromImage(image);
                Bitmap curr = image;

                for (int i = 0; i < degrees; i+=2)
                {
                    g.RotateTransform(i);
                    g.DrawImage(curr, new Point(0, 0));

                    MemoryStream ms = new MemoryStream();
                    curr.Save(ms, ImageFormat.Png);
                    curr = new Bitmap(ms);
                    g = Graphics.FromImage(curr);
                }

                g.DrawImage(image_0, new Point(0, 0));

                //MemoryStream ms = new MemoryStream();

                //image.Save(ms, ImageFormat.Png);

                //Bitmap rotated = new Bitmap(ms); 
                return FromBitmap(curr, numRows, numCols, isColor, isRowOrder);

            }


            public static int[] FromBitmap(Bitmap m, int numRows, int numCols, bool isColor, bool isRowOrder = true)
            {
                int[] newImagePixels = new int[numRows*numCols*(isColor?3:1)];

                for (int x = 0; x < numRows; x++)
                {
                    for (int y = 0; y < numCols; y++)
                    {

                        Color pixel = m.GetPixel(y, x);

                        if (isColor)
                        {
                            newImagePixels[0 * numRows * numCols + x * numCols + y] = pixel.R;
                            newImagePixels[1 * numRows * numCols + x * numCols + y] = pixel.G;
                            newImagePixels[2 * numRows * numCols + x * numCols + y] = pixel.B;
                        }
                        else
                        {
                            newImagePixels[x * numCols + y] = (pixel.R + pixel.G + pixel.B)  / 3;
                        }
                    }
                }

                return newImagePixels;
            }


            // NB: photoquality 0 - 50 
            public static int[] LossyJPGAndBack(int[] imagePixels, int numRows, int numCols, bool isColor, Int64 photoquality, bool isRowOrder = true)
            {
                
                Bitmap image = isColor ? DrawRGBPixels(imagePixels, numRows, numCols, isRowOrder) : DrawGrayscalePixels(imagePixels, numRows, numCols, isRowOrder);

                ImageCodecInfo[] encoders = ImageCodecInfo.GetImageEncoders();

                ImageCodecInfo jpgEncoder = null;
                for (int i=0; i < encoders.Length; i++)
                {
                    if (encoders[i].FormatID == ImageFormat.Jpeg.Guid)
                    {
                        jpgEncoder = encoders[i];
                        break;
                    }
                }

                Trace.Assert(jpgEncoder != null);

                System.Drawing.Imaging.Encoder myEncoder = System.Drawing.Imaging.Encoder.Quality;           
                EncoderParameters myEncoderParameters = new EncoderParameters(1);

                EncoderParameter myEncoderParameter = new EncoderParameter(myEncoder, photoquality);
                myEncoderParameters.Param[0] = myEncoderParameter;

                MemoryStream mstream = new MemoryStream();
                image.Save(mstream,jpgEncoder, myEncoderParameters); 

                System.Drawing.Image reload_image = System.Drawing.Image.FromStream(mstream);

                Bitmap m = new Bitmap(reload_image);

                int[] newImagePixels = FromBitmap(m, numRows, numCols, isColor, isRowOrder);


                //for (int x = 0; x < numRows; x++)
                //{
                //    for (int y = 0; y < numCols; y++)
                //    {
                //        Color pixel = m.GetPixel(y,x);
                //        if (isColor)
                //        {
                //            newImagePixels[0* numRows * numCols + x * numCols + y] = pixel.R;
                //            newImagePixels[1* numRows * numCols + x * numCols + y] = pixel.G;
                //            newImagePixels[2* numRows * numCols + x * numCols + y] = pixel.B;
                //        }
                //        else
                //        {
                //            newImagePixels[x * numCols + y] = pixel.R + pixel.G + pixel.B;
                //        }
                //    }
                //}
                // Utils.UDraw.DisplayImageAndPause(newImagePixels, numRows, numCols, isColor);
                // Console.WriteLine("Linf distance = {0}", Utils.UMath.LInfinityDistance(UArray.ToDoubleArray(newImagePixels), UArray.ToDoubleArray(imagePixels)));

              
                return newImagePixels;



            }



        }


        public static class Cmd
        {
            public static void RunOptionSet(OptionSet opt, string[] args)
            {
                bool show_help = false;

                var p = opt.Add("help", "Show this message and exit", x => show_help = x != null);

                List<string> extra;
                try
                {
                    extra = p.Parse(args);
                }
                catch (OptionException e)
                {
                    Console.WriteLine(e.Message);
                    Console.WriteLine("Try `--help' for more information.");
                    Environment.Exit(0);
                }

                if (show_help)
                {
                    Console.WriteLine("Options:");
                    p.WriteOptionDescriptions(Console.Out);
                    Environment.Exit(0);
                }
            }
        }
    }
}
