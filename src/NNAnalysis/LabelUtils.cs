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

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics;


namespace NNAnalysis
{
    namespace Utils
    {
        /* The crop parameter indicates whether we are giving the network a cropped image or not */
        /* Essentially we need to use NNet.CropMaybe() if the image is not cropped already.      */
        public struct LabelWithConfidence
        {

            public double[] datum;
            public int actualLabel;
            public int secBestLabel;
            public double softMaxValue;
            public double diffFromSecondBest;

            public LabelWithConfidence(double[] dat, int lab, int seclab, double val, double diff)
            {
                datum = dat;
                actualLabel = lab;
                secBestLabel = seclab;
                softMaxValue = val;
                diffFromSecondBest = diff;
            }
        }


        public class ULabel
        {

            public static double[] RunWithSoftmax(NeuralNet model, double[] datum, bool crop)
            {
                Vector<double> datum_v = DenseVector.OfArray(datum);
                if (crop)
                {
                    datum_v = model.CropMaybe(datum_v);
                }

                double[] outs = model.EvaluateNNConcretePostCrop(datum_v, null);

                UMath.SoftMax(outs);

                return outs;

            }

            public static LabelWithConfidence LabelWithConfidence(NeuralNet model, NNInstrumentation instr, double[] datum, bool crop)
            {
                Vector<double> datum_v = DenseVector.OfArray(datum);
                if (crop)
                {
                    datum_v = model.CropMaybe(datum_v);
                }

                double[] outs = model.EvaluateNNConcretePostCrop(datum_v,instr);

//                Console.WriteLine("Outs = {0}", DenseVector.OfArray(outs));


                Tuple<double, int> max = UMath.Max(outs);
                Tuple<double, int> secmax = UMath.MaxExcluding(max.Item2, outs);

                UMath.SoftMax(outs);

                var result = new LabelWithConfidence
                {
                    datum = datum,
                    actualLabel = max.Item2,
                    secBestLabel = secmax.Item2,
                    softMaxValue = outs[max.Item2],
                    diffFromSecondBest = Math.Abs(outs[max.Item2] - outs[secmax.Item2])
                };

                return result;

            }
            public static LabelWithConfidence LabelWithConfidence(NeuralNet model, double[] datum, bool crop)
            {
                return LabelWithConfidence(model, null, datum, crop);
            }

            public static int Label(NeuralNet model, double[] datum, bool crop)
            {
                return LabelWithConfidence(model, datum, crop).actualLabel;
            }

            public static LabelWithConfidence[] LabelWithConfidence(NeuralNet model, Dataset input)
            {
                LabelWithConfidence[] result = new LabelWithConfidence[input.Count()];
                for (int i = 0; i < input.Count(); i++)
                {
                    result[i] = LabelWithConfidence(model, input.GetDatum(i), true);
                }
                return result;
            }
            public static int[] Label(NeuralNet model, Dataset input)
            {
                int[] result = new int[input.Count()];
                for (int i = 0; i < input.Count(); i++)
                {
                    result[i] = Label(model, input.GetDatum(i),true);
                }
                return result;
            }

        }
    }
}
