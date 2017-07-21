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

    public class MaxPoolingLayer : PoolingLayer
    {
        public MaxPoolingLayer(int index, ImageCoordinates inputCoordinates, int kernelDimension, int padding, int stride) :
            base(index, inputCoordinates, kernelDimension, padding, stride) { }

        public override void Instrument(NNInstrumentation instrumentation, Vector<double> input, Vector<double> output)
        {
            instrumentation[Index] = Instrumentation.MaxPoolingInstrumentation(new int[OutputDimension]);
            base.ApplyKernels<NumInstDouble, double, Vector<double>, NNInstrumentation>(instrumentation, ApplyKernelConcrete, input);
        }

        public override double ApplyKernelConcrete(NNInstrumentation instr, Vector<double> input, int outIndex, int channel, int row, int column)
        {
            int argMax = InputCoordinates.GetIndex(channel, row, column);
            double max = input[argMax];

            for (int i = 0; i < KernelDimension; i++)
            {
                for (int j = 0; j < KernelDimension; j++)
                {
                    if (i == 0 && j == 0) continue;

                    int x = row - Padding + i;
                    int y = column - Padding + j;
                    if (x >= InputCoordinates.RowCount || y >= InputCoordinates.ColumnCount) continue;

                    int index = InputCoordinates.GetIndex(channel, x, y);
                    if (index < 0 || index >= input.Count) continue;

                    if (max < input[index])
                    {
                        argMax = index;
                        max = input[index];
                    }
                }
            }

            if (instr != null)
            {
                instr[Index].Selections[outIndex] = argMax;
            }

            return max;
        }
        public override LPSTerm ApplyKernelSymbolic(LPSState state, LPSTerm[] input, int outIndex, int channel, int row, int column)
        {

            int[] selections = state.Instrumentation[Index].Selections;
            int maxIndex = selections[outIndex];

            LPSTerm maxInput = input[maxIndex];
            for (int i = 0; i < KernelDimension; i++)
            {
                for (int j = 0; j < KernelDimension; j++)
                {
                    int x = row - Padding + i;
                    int y = column - Padding + j;
                    if (x >= InputCoordinates.RowCount || y >= InputCoordinates.ColumnCount) continue;

                    int curIndex = InputCoordinates.GetIndex(channel, x, y);
                    if (curIndex == maxIndex) continue;
                    if (curIndex < 0 || curIndex >= input.Length) continue;

                    // maxInput - input[curIndex] >= 0 
                    LPSTerm t = LPSTerm.Const(0.0);
                    t.Add(maxInput);
                    t.AddMul(input[curIndex], -1.0);
                    state.DeferredCts.And(t, InequalityType.GE);
                }
            }
            return maxInput;
        }

        public override bool IsAffine()
        {
            return false;
        }


    }

}
