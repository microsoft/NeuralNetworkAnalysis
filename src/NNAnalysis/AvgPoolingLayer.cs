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


    public class AvgPoolingLayer : PoolingLayer
    {
        public AvgPoolingLayer(int index, ImageCoordinates inputCoordinates, int kernelDimension, int padding, int stride) :
            base(index, inputCoordinates, kernelDimension, padding, stride) { }

        public override void Instrument(NNInstrumentation instr, Vector<double> input, Vector<double> output)
        {
            instr[Index] = Instrumentation.NoInstrumentation();
        }
        public override double ApplyKernelConcrete(NNInstrumentation instr, Vector<double> input, int outIndex, int channel, int row, int column)
        {
            return ApplyKernel<NumInstDouble, double, Vector<double>>(input, channel, row, column);
        }
        public override LPSTerm ApplyKernelSymbolic(LPSState state, LPSTerm[] input, int outIndex, int channel, int row, int column)
        {
            return ApplyKernel<NumInstLPSTermArr, LPSTerm, LPSTerm[]>(input, channel, row, column);
        }
        public T ApplyKernel<NumT,T,V>(V input, int channel, int row, int column) where NumT: struct, Num<T,V> where V : IList<T>
        {
            T sum = default(NumT).Const(0.0);
            int count = 1;

            for (int i = 0; i < KernelDimension; i++)
            {
                for (int j = 0; j < KernelDimension; j++)
                {
                    int x = row - Padding + i;
                    int y = column - Padding + j;
                    if (x >= InputCoordinates.RowCount || y >= InputCoordinates.ColumnCount) continue;

                    int index = InputCoordinates.GetIndex(channel, x, y);
                    if (index < 0 || index >= input.Count) continue;

                    default(NumT).Add(ref sum, input[index]);
                    count++;

                }
            }

            default(NumT).Mul(ref sum, 1.0/(double)count);
            return sum;

        }

        public override bool IsAffine()
        {
            return true;
        }


    }

}
