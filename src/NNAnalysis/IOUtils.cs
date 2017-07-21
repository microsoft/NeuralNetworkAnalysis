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
using System.IO;


namespace NNAnalysis
{
    namespace Utils
    {
        // TODO: what is the role of ALL_IMAGES?

        public static class MNIST
        {
            public static int ALL_IMAGES = -1;
            public static int InputDimensions = 28 * 28;
            public static ImageCoordinates InputCoordinates = new ImageCoordinates(1, 28, 28);

            public static NeuralNet GetNN(string file)
            {
                return NNetReader.ReadBinNeuralNet(file, InputDimensions, InputCoordinates);
            }

            public static ImageDataset ReadDirectoryData(string dir)
            {
                Dataset ds = new Dataset(dir, 10);
                return new ImageDataset(ds, 1, 28, 28, false);
            }

            public static ImageDataset ReadData(string labelFile, string imageFile, int maxImageCount, int startImage)
            {
                Console.WriteLine("Reading labels from: " + labelFile);
                Console.WriteLine("Reading images from: " + imageFile);
                using (FileStream fsL = File.Open(labelFile, FileMode.Open))
                using (FileStream fsI = File.Open(imageFile, FileMode.Open))
                    return ReadData(fsL, fsI, maxImageCount, startImage);
            }
            public static ImageDataset ReadData(FileStream fsL, FileStream fsI, int maxImageCount, int startImage)
            {
                // Step 0: Read the label file header
                byte[] labelHeader = new byte[8];
                fsL.Read(labelHeader, 0, labelHeader.Length);
                if (BitConverter.IsLittleEndian)
                {
                    Array.Reverse(labelHeader, 0, 4);
                    Array.Reverse(labelHeader, 4, 4);
                }
                if (BitConverter.ToUInt32(labelHeader, 0) != 2049)
                {
                    throw new Exception("Invalid label file magic number!");
                }
                int labelCount = (int)BitConverter.ToUInt32(labelHeader, 4);

                // Step 1: Read the image file header
                byte[] imageHeader = new byte[16];
                fsI.Read(imageHeader, 0, imageHeader.Length);
                if (BitConverter.IsLittleEndian)
                {
                    Array.Reverse(imageHeader, 0, 4);
                    Array.Reverse(imageHeader, 4, 4);
                    Array.Reverse(imageHeader, 8, 4);
                    Array.Reverse(imageHeader, 12, 4);
                }
                if (BitConverter.ToUInt32(imageHeader, 0) != 2051)
                {
                    throw new Exception("Invalid image file magic number!");
                }
                int imageCount = (int)BitConverter.ToUInt32(imageHeader, 4);
                int rowCount = (int)BitConverter.ToUInt32(imageHeader, 8);
                int columnCount = (int)BitConverter.ToUInt32(imageHeader, 12);
                int pixelCount = rowCount * columnCount;

                // Step 2: Do some validation
                if (labelCount != imageCount)
                {
                    throw new SystemException("Inconsistent number of labels vs images: " + labelCount + " labels vs. " + imageCount + " images");
                }
                int readImageCount = maxImageCount == ALL_IMAGES ? imageCount : Math.Min(imageCount, maxImageCount);
                Console.WriteLine("Reading " + readImageCount + " images with " + pixelCount + " pixels each");

                // Step 3: Read in the labels
                byte[] tempByteLabels = new byte[startImage];
                fsL.Read(tempByteLabels, 0, startImage);
                byte[] byteLabels = new byte[readImageCount];
                fsL.Read(byteLabels, 0, readImageCount);
                List<int> labels = Utils.UArray.ToIntArray(byteLabels).ToList();

                // Step 4: Read in the images one by one and write to the memory stream
                List<double[]> images = new List<double[]>();
                byte[] image = new byte[pixelCount];
                for (int i = 0; i < startImage; i++)
                {
                    fsI.Read(image, 0, image.Length);
                }
                for (int i = 0; i < readImageCount; i++)
                {
                    // Step 4a: Read the image
                    fsI.Read(image, 0, image.Length);
                    images.Add(Utils.UArray.ToDoubleArrayFromInt8Bytes(image));
                }

                // Step 5: Build the data point collection
                Console.WriteLine("Done reading images");
                return new ImageDataset(new Dataset(images, labels, 10), 1, rowCount, columnCount, false);

            }
            public static void WriteData(string labelFile, string imageFile, ImageDataset data)
            {
                UTF8Encoding encoding = new UTF8Encoding(true);
                Console.WriteLine("Writing labels to: " + labelFile);
                Console.WriteLine("Writing images to: " + imageFile);
                using (FileStream fsL = File.Open(labelFile, FileMode.Create))
                using (FileStream fsI = File.Open(imageFile, FileMode.Create))
                    WriteData(fsL, fsI, data);
            }
            private static void WriteData(FileStream fsL, FileStream fsI, ImageDataset data)
            {
                {
                    int numPixels = data.RowCount * data.ColumnCount;
                    Console.WriteLine("Writing " + data.Dataset.Count() + " images with " + numPixels + " pixels each");

                    // Step 0: Write the label file header
                    byte[] labelHeader = new byte[8];
                    Array.Copy(BitConverter.GetBytes(2049), 0, labelHeader, 0, 4);
                    Array.Copy(BitConverter.GetBytes(data.Dataset.Count()), 0, labelHeader, 4, 4);
                    if (BitConverter.IsLittleEndian)
                    {
                        Array.Reverse(labelHeader, 0, 4);
                        Array.Reverse(labelHeader, 4, 4);
                    }
                    fsL.Write(labelHeader, 0, 8);

                    // Step 1: Write the image file header
                    byte[] imageHeader = new byte[16];
                    Array.Copy(BitConverter.GetBytes(2051), 0, imageHeader, 0, 4);
                    Array.Copy(BitConverter.GetBytes(data.Dataset.Count()), 0, imageHeader, 4, 4);
                    Array.Copy(BitConverter.GetBytes(data.RowCount), 0, imageHeader, 8, 4);
                    Array.Copy(BitConverter.GetBytes(data.ColumnCount), 0, imageHeader, 12, 4);
                    if (BitConverter.IsLittleEndian)
                    {
                        Array.Reverse(imageHeader, 0, 4);
                        Array.Reverse(imageHeader, 4, 4);
                        Array.Reverse(imageHeader, 8, 4);
                        Array.Reverse(imageHeader, 12, 4);
                    }
                    fsI.Write(imageHeader, 0, imageHeader.Length);

                    // Step 3: Write the labels
                    byte[] labels = new byte[data.Dataset.Count()];
                    for (int i = 0; i < data.Dataset.Count(); i++)
                    {
                        labels[i] = Convert.ToByte(data.Dataset.GetLabel(i));
                    }
                    fsL.Write(labels.ToArray(), 0, labels.Length);

                    // Step 4: Write in the images one by one and write to the memory stream
                    for (int i = 0; i < labels.Length; i++)
                    {
                        fsI.Write(Utils.UArray.ToByteArray(data.Dataset.GetDatum(i)), 0, data.Dataset.GetDatum(i).Length);
                    }
                    fsL.Flush();
                    fsI.Flush();
                    Console.WriteLine("Done writing");
                }
            }

        }        

        public static class CIFAR
        {
            public static int ALL_IMAGES = -1;

            public static int InputDimensions = 32 * 32 * 3;
            public static ImageCoordinates InputCoordinates = new ImageCoordinates(3, 32, 32);

            //  Reading and writing CIFAR networks and data
            public static NeuralNet GetNN(String file)
            {
                return NNetReader.ReadBinNeuralNet(file, InputDimensions, InputCoordinates);
            }

            public static ImageDataset ReadDirectoryData(string dir)
            {
                Dataset ds = new Dataset(dir, 10); 
                return new ImageDataset(ds, 3, 32, 32, true);
            }


            public static ImageDataset ReadData(string file, int maxImageCount, int startImage)
            {
                Console.WriteLine("Reading data from: " + file);
                using (FileStream fs = File.Open(file, FileMode.Open))
                {
                    // Read in the images one by one and write to the memory stream
                    List<double[]> images = new List<double[]>();
                    List<int> labels = new List<int>();
                    byte[] image = new byte[1 + 3072]; // first byte is label, next 3072 are image
                    int bytesRead;
                    int counter = 0;
                    while ((bytesRead = fs.Read(image, 0, image.Length)) > 0)
                    {
                        if (counter < startImage)
                        {
                            counter++;
                            continue;
                        }
                        if (maxImageCount != -1 && counter >= startImage + maxImageCount)
                        {
                            break;
                        }
                        if (bytesRead != image.Length)
                        {
                            throw new Exception("Invalid CIFAR data file!");
                        }
                        // Step 4a: Read the label
                        labels.Add(image[0]);
                        // Step 4b: Read the image
                        images.Add(Utils.UArray.ToDoubleArray(image, 1, image.Length - 1));
                        counter++;
                    }
                    Console.WriteLine("Read {0} images", counter);

                    // Step 5: Build the data point collection
                    return new ImageDataset(new Dataset(images, labels, 10), 3, 32, 32, true);
                }
            }
            public static void WriteData(string file, ImageDataset images)
            {
                UTF8Encoding encoding = new UTF8Encoding(true);
                Console.WriteLine("Writing data to: " + file);
                using (FileStream fw = File.Open(file, FileMode.Create))
                {
                    // Write in the images one by one
                    for (int i = 0; i < images.Dataset.Count(); i++)
                    {
                        // Step 4a: Write the label
                        fw.WriteByte(Convert.ToByte(images.Dataset.GetLabel(i)));
                        // Step 4b: Write the image
                        fw.Write(Utils.UArray.ToByteArray(images.Dataset.GetDatum(i)), 0, images.Dataset.GetDatum(i).Length);
                    }
                    fw.Flush();
                    Console.WriteLine("Done writing");
                }
            }
        }
    }
}
