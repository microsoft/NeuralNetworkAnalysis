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

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics;


namespace NNAnalysis
{

    interface ITransform
    {
        ImageCoordinates TransformedCoordinates();
        ImageCoordinates OriginalCoordinates();

        int TransformedDimension();
        int OriginalDimension();

        Vector<double> Transform(Vector<double> input);

        Vector<double> UnTransform(Vector<double> original, Vector<double> image);


        LPSTerm[] Transform(LPSTerm[] input);

    }


    public class CropTransform : ITransform
    {
        ImageCoordinates inputCoordinates_;
        int inputDimension_;
        int cropSize_;
        int outputDimension_;
        ImageCoordinates outputCoordinates_;
        bool fromCenter_ = true;

        public CropTransform(ImageCoordinates inputCoordinates, int inputDimension, int cropSize, bool fromCenter = true)
        {
            cropSize_ = cropSize;
            inputDimension_ = inputDimension;
            inputCoordinates_ = inputCoordinates;

            outputDimension_ = cropSize * cropSize * inputCoordinates.ChannelCount;
            outputCoordinates_ = new ImageCoordinates(inputCoordinates.ChannelCount, cropSize, cropSize);
        }
        public ImageCoordinates OriginalCoordinates()
        {
            return inputCoordinates_;
        }
        public int OriginalDimension()
        {
            return inputDimension_;
        }

        public ImageCoordinates TransformedCoordinates()
        {
            return outputCoordinates_;
        }
        public int TransformedDimension()
        {
            return outputDimension_;
        }

        /// <summary>
        /// Given an original image and a transformed one, we get back one that
        /// has the same dimensions the original and if we transform it we get back image, i.e:
        /// <code> UnTransform(orig, Transform(orig)) = orig </code>
        /// </summary>
        /// <param name="orig">The original image.</param>
        /// <param name="image">The transformed image.</param>
        /// <returns>An image satisfying the above equation.</returns>
        public Vector<double> UnTransform(Vector<double> orig, Vector<double> image)
        {
            Vector<double> ret = DenseVector.Create(orig.Count(), 0.0);
            orig.CopyTo(ret);
            // Now ret is a copy of orig, so simply overwrite the relevant pixels

            int center_row = inputCoordinates_.RowCount / 2;
            int center_col = inputCoordinates_.ColumnCount / 2;

            int topleft_row = center_row - cropSize_ / 2;
            int topleft_col = center_col - cropSize_ / 2;

            for (int channel = 0; channel < inputCoordinates_.ChannelCount; channel++)
            {
                for (int i = 0; i < cropSize_; i++)
                {
                    for (int j = 0; j < cropSize_; j++)
                    {
                        int input_idx = inputCoordinates_.GetIndex(channel, topleft_row + i, topleft_col + j);
                        int output_idx = outputCoordinates_.GetIndex(channel, i, j);
                        if (input_idx >= 0 && input_idx < inputDimension_ && output_idx >= 0 && output_idx < outputDimension_)
                        {
                            ret[input_idx] = image[output_idx];
                        }
                    }
                }
            }
            return ret;

        }



        V TransformGeneric<NumT,T,V>(V input)
            where NumT : struct, Num<T, V>
            where V : IList<T>
        {
            if (!fromCenter_) throw new NotImplementedException("Non-center image cropping not supported yet!");

            int center_row = inputCoordinates_.RowCount / 2;
            int center_col = inputCoordinates_.ColumnCount / 2;

            int topleft_row = center_row - cropSize_ / 2;
            int topleft_col = center_col - cropSize_ / 2;

            V output = default(NumT).CreateVector(
                outputCoordinates_.ChannelCount * 
                outputCoordinates_.RowCount * 
                outputCoordinates_.ColumnCount);

            for (int channel = 0; channel < inputCoordinates_.ChannelCount; channel++)
            {
                for (int i = 0; i < cropSize_; i++)
                {
                    for (int j=0; j < cropSize_; j++)
                    {
                        int input_idx = inputCoordinates_.GetIndex(channel,topleft_row + i,topleft_col + j);
                        int output_idx = outputCoordinates_.GetIndex(channel,i,j);
                        if (input_idx >= 0 && input_idx < inputDimension_ && output_idx >= 0 && output_idx < outputDimension_)
                        {
                            var z = default(NumT).Const(0.0);
                            default(NumT).Add(ref z, input[input_idx]);
                            output[output_idx] = z;
                            
                        }
                    }
                }
            }
            
            return output;
        }

        public Vector<double> Transform(Vector<double> input)
        {

//             Utils.UDraw.DisplayImageAndPause(Utils.UArray.ToRGBArray(input.ToArray(), 128, 127), 32, 32, true);


            var output = TransformGeneric<NumInstDouble, double, Vector<double>>(input);


//            Utils.UDraw.DisplayImageAndPause(Utils.UArray.ToRGBArray(output.ToArray(), 128, 127), 30, 30, true);


           return output;

        }

        public LPSTerm[] Transform(LPSTerm[] input)
        {
            return TransformGeneric<NumInstLPSTermArr, LPSTerm, LPSTerm[]>(input);
        }

    }

    class DataLayer : Layer
    {
        // Scaling
        double scale_ = 1.0;

        // Mean image, NB: only one of the two can be non-null. 
        // Invariant: their coordinates must be the coordinates of this layer
        double[] meanImage_ = null;
        double[] meanChannel_ = null; 

        // NB: Mirror not really implemented ... 
        // Nullable<bool> mirror_ = null;

        public DataLayer(int index, int inputDimension, ImageCoordinates inputCoordinates)
        {
            scale_ = 1.0;
            meanImage_ = null;
            meanChannel_ = null;
            InitLayer(index, LayerType.DATA_LAYER, inputDimension, inputDimension, inputCoordinates, inputCoordinates);
        }

        public DataLayer(int index, int inputDimension, 
            ImageCoordinates inputCoordinates, 
            double scale, 
            double[] meanImage, 
            List<double> meanChannel
            )
        {
            scale_ = scale;
            meanImage_ = meanImage;
            if (meanChannel != null && meanChannel.Count > 0)
            {
                int channel_count = inputCoordinates.ChannelCount;
                meanChannel_ = new double[channel_count];
                for (int i=0; i < channel_count; i++)
                {
                    meanChannel_[i] = meanChannel[i % meanChannel.Count()];
                }
            }

            InitLayer(index, LayerType.DATA_LAYER, inputDimension, inputDimension, inputCoordinates, inputCoordinates);
        }


        public override void Instrument(NNInstrumentation instr, Vector<double> input, Vector<double> output)
        {
            instr[Index] = Instrumentation.NoInstrumentation();
        }

        public override bool IsAffine()
        {
            return false; // Oh well just just don't coalesce across data layer although technically it's entirely possible

        }
        public override Vector<double> EvaluateConcrete(Vector<double> input)
        {

            // If we have a meanImage ...
            if (meanImage_ != null)
            {
                return (input - DenseVector.OfArray(meanImage_)) * scale_;
            }

            // If we have a meanChannel ... 
            if (meanChannel_ != null && meanChannel_.Count() > 0)
            {
                Vector<double> cur = input;
                for (int channel = 0; channel < InputCoordinates.ChannelCount; channel++)
                {
                    for (int r = 0; r < InputCoordinates.RowCount; r++)
                    {
                        for (int c = 0; c < InputCoordinates.ColumnCount; c++)
                        {
                            int index = InputCoordinates.GetIndex(channel, r, c);
                            cur[index] = (input[index] - meanChannel_[channel]) * scale_;
                        }
                    }
                }
                return cur;
            }
            // If we are only doing scaling ... 
            return (input * (float) scale_);
        }



        public override LPSTerm[] EvaluateSymbolic(LPSState state, LPSTerm[] input)
        {

            LPSTerm[] cur = default(NumInstLPSTermArr).CreateVector(input.Length);

            // If we have a meanImage ...
            if (meanImage_ != null)
            {
                for (int i = 0; i < input.Length; i++)
                {
                    cur[i].Sub(LPSTerm.Const(meanImage_[i]));  // - mean
                    cur[i].Add(input[i]);                      // + input
                    cur[i].Mul(scale_);
                }

                return cur;
            }

            // If we have a meanChannel ... 
            if (meanChannel_ != null && meanChannel_.Count() > 0)
            {
                for (int channel = 0; channel < InputCoordinates.ChannelCount; channel++)
                {
                    for (int r = 0; r < InputCoordinates.RowCount; r++)
                    {
                        for (int c = 0; c < InputCoordinates.ColumnCount; c++)
                        {
                            int index = InputCoordinates.GetIndex(channel, r, c);
                            cur[index].Sub(LPSTerm.Const(meanChannel_[channel]));
                            cur[index].Add(input[index]);
                            cur[index].Mul(scale_);
                        }
                    }
                }
                return cur;
            }

            // Finally, if we are only doing scaling ...
            for (int i = 0; i < input.Length; i++)
            {
                cur[i].Add(input[i]);
                cur[i].Mul(scale_);
            }
            return cur;
        }

    }
}
