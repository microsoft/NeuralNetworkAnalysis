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

namespace NNAnalysis
{

    public abstract class PoolingLayer : Layer
    {
        ImageCoordinates inputCoordinates_;
        int kernelDimension_;
        int padding_;
        int stride_;
        public int KernelDimension
        {
            get { return kernelDimension_; }
        }
        public int Padding
        {
            get { return padding_; }
        }
        public int Stride
        {
            get { return stride_; }
        }

        public PoolingLayer(int index, ImageCoordinates inputCoordinates, int kernelDimension, int padding, int stride) : base()
        {
            inputCoordinates_ = inputCoordinates;
            kernelDimension_ = kernelDimension;
            padding_ = padding;
            stride_ = stride;

            int inputDimension = inputCoordinates_.ChannelCount * inputCoordinates_.RowCount * inputCoordinates_.ColumnCount;

            int rowCount = Utils.UImageCoordinate.ComputeOutputCounts(kernelDimension, inputCoordinates.RowCount, stride, padding, true);
            int columnCount = Utils.UImageCoordinate.ComputeOutputCounts(kernelDimension, inputCoordinates.ColumnCount, stride, padding, true);

            int outputDimension = inputCoordinates.ChannelCount * rowCount * columnCount;

            ImageCoordinates ouputCoordinates = new ImageCoordinates(inputCoordinates.ChannelCount, rowCount, columnCount);

            InitLayer(index, LayerType.POOLING_LAYER, inputDimension, outputDimension, inputCoordinates, ouputCoordinates);

        }


        public abstract double ApplyKernelConcrete(NNInstrumentation instr, Vector<double> input, int outIndex, int channel, int row, int column);
        public abstract LPSTerm ApplyKernelSymbolic(LPSState state, LPSTerm[] input, int outIndex, int channel, int row, int column);


        protected V ApplyKernels<NumT,T,V,S>(S state, Func<S,V,int,int,int,int,T> applyKernel, V input) where NumT : struct, Num<T,V> where V : IList<T>
        {
            V output = default(NumT).CreateVector(OutputDimension);
            int stride = Stride;

            int jbound = Utils.UImageCoordinate.ComputeOutputCounts(KernelDimension, InputCoordinates.RowCount, Stride, Padding, true);
            int kbound = Utils.UImageCoordinate.ComputeOutputCounts(KernelDimension, InputCoordinates.ColumnCount, Stride, Padding, true);

            for (int i = 0; i < InputCoordinates.ChannelCount; i++)
            {
                for (int j = 0; j < jbound; j++)
                {
                    for (int k = 0; k < kbound; k++)
                    {
                        int index = OutputCoordinates.GetIndex(i, j, k);
                        T value = applyKernel(state, input, index, i, j * stride, k * stride);
                        output[index] = value;
                    }
                }
            }
            return output;
        }


        public override LPSTerm[] EvaluateSymbolic(LPSState state, LPSTerm[] input)
        {
            return ApplyKernels<NumInstLPSTermArr, LPSTerm, LPSTerm[], LPSState>(state, ApplyKernelSymbolic, input);
        }

        public override Vector<double> EvaluateConcrete(Vector<double> input)
        {
            return ApplyKernels<NumInstDouble, double, Vector<double>, NNInstrumentation>(null, ApplyKernelConcrete, input);
        }

    }

}
