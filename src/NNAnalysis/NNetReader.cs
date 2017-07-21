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
using System.Runtime.CompilerServices;

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

using MathNet.Numerics;

using System.Diagnostics;

namespace NNAnalysis
{

        public class NNetReader
        {
            public static byte[] ReadBytes(BinaryReader input, int byteCount)
            {
                byte[] bytes = input.ReadBytes(byteCount);
                Array.Reverse(bytes);
                return bytes;
            }
            public static int ReadInt32(BinaryReader input)
            {
                byte[] bytes = ReadBytes(input, 4);
                return BitConverter.ToInt32(bytes, 0);
            }
            public static double ReadSingle(BinaryReader input)
            {
                byte[] bytes = ReadBytes(input, 4);
                return (double)BitConverter.ToSingle(bytes, 0);
            }
            public static int ReadInt32_TextNewLine(TextReader input)
            {
                var str = input.ReadLine();
                return Int32.Parse(str);
            }
            public static double ReadSingle_TextNewLine(TextReader input)
            {
                var str = input.ReadLine();
                return Double.Parse(str);
            }
            static Tuple<CropTransform,Layer> ReadDataLayer(
                int index, 
                Func<int> readInt32, Func<double> readSingle, 
                ImageCoordinates inputCoordinates,
                int inputDim)
            {
                int has_transform_param = readInt32();
                if (has_transform_param == 0) { 
                    DataLayer layer = new DataLayer(index,inputDim,inputCoordinates);
                    return new Tuple<CropTransform, Layer>(null, layer);
                }

                // Scale 
                double scale = 1.0;
                int has_scale = readInt32();
                if (has_scale != 0)
                    scale = readSingle();
                
                // Mirror 
                int has_mirror = readInt32(); // ignore
                
                // Crop size
                CropTransform cropT_ = null;
                int has_crop_siz = readInt32();
                if (has_crop_siz != 0)
                {
                    int crop_siz = readInt32();
                    cropT_ = new CropTransform(inputCoordinates, inputDim, crop_siz, true);
                }
                // Mean value
                List<double> mean_val = new List<double>();
                int mean_val_cnt = readInt32();
                for (int x = 0; x < mean_val_cnt; x++)
                {
                    mean_val.Add(readSingle());
                }
                // Mean file
                int has_mean_file = readInt32();
                double[] mean_image = null;
                if (has_mean_file != 0)
                {
                    int mean_image_siz = readInt32();
                    if (mean_image_siz > 0) mean_image = new double[mean_image_siz];
                    for (int x = 0; x < mean_image_siz; x++)
                    {
                        mean_image[x] = readSingle();
                    }
                }

                ImageCoordinates dataLayerInputCoordinates = inputCoordinates;
                int dataLayerInputDim = inputDim;
                double[] dataLayerMeanImage = mean_image;
                if (cropT_ != null)
                {
                    dataLayerInputCoordinates = cropT_.TransformedCoordinates();
                    dataLayerInputDim = cropT_.TransformedDimension();
                    if (mean_image != null)
                    {
                        dataLayerMeanImage = cropT_.Transform(DenseVector.OfArray(mean_image)).ToArray();
                    }
                }

                Layer l = new DataLayer(
                    index, 
                    dataLayerInputDim, 
                    dataLayerInputCoordinates, 
                    scale, 
                    dataLayerMeanImage, 
                    mean_val);
                return new Tuple<CropTransform, Layer>(cropT_, l);

            }

            static Layer ReadInnerProductLayer(int index, Func<int> readInt32, Func<double> readSingle, ImageCoordinates inputCoordinates, int inputDim)
            {
                // Caffe format:
                // K : input dimension
                // N : output dimension
                // A : N * K dimensional matrix (row major order)
                // B : N dimensional vector
                // Matrix formula: output = A * input + B
                // Array formula: output[i] = (\sum_j A[i][j] * x[j]) + B[i]
                int inputDimension = readInt32();
                int outputDimension = readInt32();
                double[][] weights = new double[outputDimension][];
                for (int i = 0; i < outputDimension; i++)
                {
                    weights[i] = new double[inputDimension];
                    for (int j = 0; j < inputDimension; j++)
                    {
                        weights[i][j] = readSingle();
                    }
                }
                double[] intercept = new double[outputDimension];
                for (int i = 0; i < outputDimension; i++)
                {
                    intercept[i] = readSingle();
                }
                Console.WriteLine("Dimensions: " + inputDimension + " * " + outputDimension);
                return new InnerProductLayer(index, weights, intercept, inputCoordinates);
            }
            static Layer ReadRectifiedLinearLayer(int index, Func<int> readInt32, Func<double> readSingle, ImageCoordinates inputCoordinates, int inputDim)
            {
                return new ReLULayer(index, inputDim, inputCoordinates);
            }
            static Layer ReadSoftmaxLayer(Func<int> readInt32, Func<double> readSingle)
            {
                return null;
            }
            static Layer ReadConvolutional(int index, Func<int> readInt32, Func<double> readSingle, ImageCoordinates inputCoordinates, int inputDim)
            {
                int kernelCount = readInt32();
                int kernelDimension = readInt32();
                int padding = readInt32();

                // read kernels
                int kernelTotalDataCount = readInt32();
                int kernelDataCount = kernelTotalDataCount / kernelCount;
                double[][] kernels = new double[kernelCount][];
                for (int i = 0; i < kernelCount; i++)
                {
                    kernels[i] = new double[kernelDataCount];
                    for (int j = 0; j < kernelDataCount; j++)
                    {
                        kernels[i][j] = readSingle();
                    }
                }

                // read intercepts
                int interceptTotalDataCount = readInt32();
                if (interceptTotalDataCount != kernelCount)
                {
                    throw new Exception("Invalid parameters!");
                }
                double[] intercepts = new double[kernelCount];
                for (int i = 0; i < kernelCount; i++)
                {
                    intercepts[i] = readSingle();
                }

                int channelCount = (kernelDataCount / (kernelCount * kernelDimension * kernelDimension));
                ImageCoordinates kernelCoordinates = new ImageCoordinates(channelCount, kernelDimension, kernelDimension);

                return new ConvolutionLayer(index, inputCoordinates, kernels, intercepts, kernelDimension, padding);
            }
            static Layer ReadPooling(int index, Func<int> readInt32, Func<double> readSingle, ImageCoordinates inputCoordinates)
            {
                int kernelDimension = readInt32();
                int stride = readInt32();
                int padding = readInt32();

                int poolMeth = readInt32();

                if (kernelDimension == 0)
                {
                    Console.WriteLine("Kernel dimension = 0, treating this as global pooling!");
                    Debug.Assert(inputCoordinates.ColumnCount == inputCoordinates.RowCount);
                    Debug.Assert(padding == 0);

                    kernelDimension = inputCoordinates.ColumnCount;
                }


                if (poolMeth == 0)
                {   // MAX
                    Console.WriteLine("Pool method = MAX");
                    return new MaxPoolingLayer(index, inputCoordinates, kernelDimension, padding, stride);
                }
                else
                {   // AVG 
                    Console.WriteLine("Pool method = AVG");
                    return new AvgPoolingLayer(index, inputCoordinates, kernelDimension, padding, stride);
                }
            }
            public static Layer ReadLayer(int index, Func<int> readInt32, Func<double> readSingle, 
                ImageCoordinates inputCoordinates, 
                int inputDim,
                out CropTransform cropT
                )
            {
                int typeID = readInt32();
                cropT = null;
                Console.Write("Reading layer with index {0,2}, of type {1}, input dimension {2}:", index, typeID, inputDim);
                switch (typeID)
                {
                    case 0: // "Data"
                        Console.WriteLine("Data");
                        var res = ReadDataLayer(index, readInt32, readSingle, inputCoordinates, inputDim);
                        cropT = res.Item1;
                        return res.Item2;
                    case 1: // "InnerProduct"
                        Console.Write("InnerProduct");
                        return ReadInnerProductLayer(index, readInt32, readSingle, inputCoordinates, inputDim);
                    case 2: // "ReLU"
                        Console.WriteLine("ReLU");
                        return ReadRectifiedLinearLayer(index, readInt32, readSingle, inputCoordinates, inputDim);
                    case 3: // "SoftmaxWithLoss"
                        Console.WriteLine("SoftMax");
                        return ReadSoftmaxLayer(readInt32, readSingle);
                    case 4: // "Convolution"
                        Console.WriteLine("Convolution");
                        return ReadConvolutional(index, readInt32, readSingle, inputCoordinates, inputDim);
                    case 5: // "Pooling"
                        Console.Write("Pooling, ");
                        return ReadPooling(index, readInt32, readSingle, inputCoordinates);
                    case 6: // "Dropout"
                        Console.Write("Dropout, ");
                        return null;
                    default:
                        throw new Exception("Layer type ID not recognized: " + typeID);
                }
            }
            public static NeuralNet ReadNeuralNet(Func<int> readInt32, Func<double> readSingle, int inputDimension, ImageCoordinates inputCoordinates)
            {
                NeuralNet nn = new NeuralNet();
                int layerCount = readInt32();

                int curDimension = inputDimension;
                ImageCoordinates curCoordinates = inputCoordinates;

                CropTransform cropT = null;

                for (int i = 0; i < layerCount; i++)
                {
                    Layer layer = ReadLayer(i, readInt32, readSingle, curCoordinates, curDimension, out cropT);
                    if (layer != null)
                    {
                        nn.AddLayer(layer);
                        curDimension = layer.OutputDimension;
                        curCoordinates = layer.OutputCoordinates;

                        if (layer.LayerType == LayerType.DATA_LAYER && cropT != null)
                        {
                            nn.AddCropTransform(cropT);
                        }

                    }
                }

                // Currently disabled because perf gains are not enough, in fact things seem somewhat slower ... 
                // Console.Write("Linearizing sequences of linear layers ...");
                // Coalesce affine layers for running the network more optimally
                // nn.CoalesceToVirtual();
                // GC.Collect(2);
                // Console.WriteLine("Done.");
                return nn;
            }
            /// <summary>
            /// Reading from our own binary format. These are the networks we get as output of our protobuf utilities from Caffe.
            /// </summary>
            /// <param name="file"></param>
            /// <param name="inputDimension"></param>
            /// <param name="inputCoordinates"></param>
            /// <returns></returns>
            public static NeuralNet ReadBinNeuralNet(string file, int inputDimension, ImageCoordinates inputCoordinates)
            {
                Console.WriteLine("Reading neural net from file: " + file);
                using (FileStream fs = File.Open(file, FileMode.Open))
                {
                    var br = new BinaryReader(fs);
                    return ReadNeuralNet(() => ReadInt32(br),
                                         () => ReadSingle(br), inputDimension, inputCoordinates);
                }
            }
            /// <summary>
            /// Reading from our own text format. These are the networks we we get as output from our Torch/LUA experiments.
            /// </summary>
            /// <param name="file"></param>
            /// <param name="inputDimension"></param>
            /// <param name="inputCoordinates"></param>
            /// <returns></returns>
            public static NeuralNet ReadTxtNeuralNet(string file, int inputDimension, ImageCoordinates inputCoordinates)
            {
                Console.WriteLine("Reading neural net from file: " + file);
                using (FileStream fs = File.Open(file, FileMode.Open))
                {
                    TextReader br = new StreamReader(fs);
                    return ReadNeuralNet(() => ReadInt32_TextNewLine(br),
                                         () => ReadSingle_TextNewLine(br), inputDimension, inputCoordinates);
                }
            }
        }
}
