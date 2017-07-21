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

using System.Threading;

using System.Diagnostics;


namespace NNAnalysis
{

    public class ConvolutionLayer : Layer
    {
        // let V = channels * kernelrows * kernelcolumns
        Matrix<double> kernelMatrix_;             // kernelCount * V
        Vector<double> interceptVector_;
        int kernelDimension_;
        int padding_;

        int outputRowCount_;
        int outputColumnCount_;
        int outputChannelCount_;

        ImageCoordinates kernelCoordinates_;

        ThreadLocal<LPSTerm[]> symbolic_output_storage;

        // Scratchpad stuff
        Matrix<double> _intercept_scratch; // #kernels x kernelpositions copies of the intercept vector.


        ThreadLocal<Matrix<double>> _input_scratch;  // kernelmatrix.columncount x kernelpositions
        ThreadLocal<Vector<double>> _output_result;  // kernelmatrix.rowCount * kernelpositions
        ThreadLocal<Matrix<double>> _output_scratch;

        public void InputToScratch(Vector<double> input)
        {

            // Must populate _input_scratch :: kernelMatrix.ColumnCount x kernel-positions

            var jBound = Utils.UImageCoordinate.ComputeOutputCounts(KernelDimension, InputCoordinates.RowCount, 1, Padding, false);
            var kBound = Utils.UImageCoordinate.ComputeOutputCounts(KernelDimension, InputCoordinates.ColumnCount, 1, Padding, false);

            for (int row = 0; row < jBound; row++)
            {
                for (int col = 0; col < kBound; col++) 
                {
                    for (int c = 0; c < InputCoordinates.ChannelCount; c++)
                    {
                        for (int i = 0; i < KernelDimension; i++)
                        {
                            for (int j = 0; j < KernelDimension; j++)
                            {

                                int x = row - padding_ + i;
                                int y = col - padding_ + j;

                                int output_x = c * KernelDimension * KernelDimension + i * KernelDimension + j;
                                int output_y = row * kBound + col;

                                if (x < 0 || y < 0 || x >= InputCoordinates.RowCount || y >= InputCoordinates.ColumnCount)
                                {
                                    _input_scratch.Value[output_x, output_y] = 0;
                                    continue;
                                }

                                int index = InputCoordinates.GetIndex(c, x, y);

                                if (index < 0 || index >= input.Count)
                                {
                                    _input_scratch.Value[output_x, output_y] = 0;
                                    continue;
                                }

                                _input_scratch.Value[output_x, output_y] = input[index];
                            }
                        }
                    }


                }
            }
        }



        public Vector<double> OutputScratchToRes(Matrix<double> output_scratch)
        {
            return DenseVector.OfArray(output_scratch.ToRowWiseArray());
        }

        public Vector<double> DoConvolution(Vector<double> input)
        {
            InputToScratch(input);

            kernelMatrix_.Multiply(_input_scratch.Value, _output_scratch.Value);
            _output_scratch.Value.Add(_intercept_scratch, _output_scratch.Value);

            /* 
            var res = kernelMatrix_ * _input_scratch + _intercept_scratch;
            return OutputScratchToRes(res);
             */
            return OutputScratchToRes(_output_scratch.Value);

        }

        private void CheckInitThreadLocalScratch(int kernelpositions)
        {
            _input_scratch  = new ThreadLocal<Matrix<double>>(() => DenseMatrix.Create(kernelMatrix_.ColumnCount, kernelpositions,0.0));
            _output_scratch = new ThreadLocal<Matrix<double>>(() => DenseMatrix.Create(kernelMatrix_.RowCount, kernelpositions, 0.0));
            _output_result =  new ThreadLocal<Vector<double>>(() => DenseVector.Create(kernelMatrix_.RowCount * kernelpositions, 0.0));
        }

        public ConvolutionLayer(int index, ImageCoordinates inputCoordinates, double[][] kernels, double[] intercepts, int kernelDimension, int padding)
        {
            kernelMatrix_ = DenseMatrix.OfRowArrays(kernels);
            interceptVector_ = DenseVector.OfArray(intercepts);
            kernelDimension_ = kernelDimension;
            padding_ = padding;

            int inputDimension = inputCoordinates.ChannelCount * inputCoordinates.RowCount * inputCoordinates.ColumnCount;

            kernelCoordinates_ = new ImageCoordinates(kernels.Length, kernelDimension_, kernelDimension_);

            outputRowCount_ = Utils.UImageCoordinate.ComputeOutputCounts(kernelDimension, inputCoordinates.RowCount, 1, padding, false);
            outputColumnCount_ = Utils.UImageCoordinate.ComputeOutputCounts(kernelDimension, inputCoordinates.ColumnCount, 1, padding, false);
            outputChannelCount_ = kernels.Length;

            ImageCoordinates outputCoordinates = new ImageCoordinates(outputChannelCount_, outputRowCount_, outputColumnCount_);

            int outputDimension = outputChannelCount_ * outputRowCount_ * outputColumnCount_;

            InitLayer(index, LayerType.CONVOLUTION_LAYER, inputDimension, outputDimension, inputCoordinates, outputCoordinates);

            symbolic_output_storage = new ThreadLocal<LPSTerm[]>();

            // Fast convolution stuff:

            var jBound = Utils.UImageCoordinate.ComputeOutputCounts(KernelDimension, InputCoordinates.RowCount, 1, Padding, false);
            var kBound = Utils.UImageCoordinate.ComputeOutputCounts(KernelDimension, InputCoordinates.ColumnCount, 1, Padding, false);

            int kernelpositions = jBound * kBound;

            CheckInitThreadLocalScratch(kernelpositions);

            _intercept_scratch = DenseMatrix.Create(kernelMatrix_.RowCount, kernelpositions, 0.0);
            for (int k = 0; k < kernelMatrix_.RowCount; k++)
            {
                for (int x = 0; x < kernelpositions; x++)
                {
                    _intercept_scratch[k, x] = interceptVector_[k];
                }
            }

        }

        public int OutputRowCount
        {
            get { return outputRowCount_; }
        }
        public int OutputColumnCount
        {
            get { return outputColumnCount_; }
        }
        public int OutputChannelCount
        {
            get { return outputChannelCount_; }
        }
        public int KernelCount
        {
            get { return kernelMatrix_.RowCount; }
        }
        public Matrix<double> Kernels
        {
            get { return kernelMatrix_; }
        }
        public Vector<double> Intercepts
        {
            get { return interceptVector_; }
        }
        public int Padding
        {
            get { return padding_; }
        }
        public int KernelDimension
        {
            get { return kernelDimension_; }
        }
        public ImageCoordinates KernelCoordinates
        {
            get { return kernelCoordinates_; }
        }
        public override void Instrument(NNInstrumentation instr, Vector<double> input, Vector<double> output)
        {
            instr[Index] = Instrumentation.NoInstrumentation();
        }
        public void ApplyKernel<NumT,T,V>(ref T output, V input, int padding, int kernel, int row, int column)
            where NumT : struct, Num<T,V> 
            where V : IList<T>
        {

            for (int i = 0; i < InputCoordinates.ChannelCount; i++)
            {
                for (int j = 0; j < KernelDimension; j++)
                {
                    for (int k = 0; k < KernelDimension; k++)
                    {
                        int x = row - padding + j;
                        int y = column - padding + k;

                        if (x < 0 || y < 0 || x >= InputCoordinates.RowCount || y >= InputCoordinates.ColumnCount) continue;

                        int index = InputCoordinates.GetIndex(i, x, y);
                        if (index >= 0 && index < input.Count)
                        {
                            default(NumT).AddMul(ref output,input[index],Kernels[kernel, KernelCoordinates.GetIndex(i, j, k)]);
                        }
                    }
                }
            }
            
            default(NumT).Add(ref output,Intercepts[kernel]);
            return;
        }
        public V Evaluate<NumT,T,V>(V input) where NumT : struct, Num<T,V> where V : IList<T>
        {
            V output = default(NumT).CreateVector(OutputDimension); // Has initialized all values to 0.0
            Evaluate<NumT,T,V>(input, output);
            return output;
        }

        public void Evaluate<NumT, T, V>(V input, V output)
            where NumT : struct, Num<T, V>
            where V : IList<T>
        {
            var jBound = Utils.UImageCoordinate.ComputeOutputCounts(KernelDimension, InputCoordinates.RowCount, 1, Padding, false);
            var kBound = Utils.UImageCoordinate.ComputeOutputCounts(KernelDimension, InputCoordinates.ColumnCount, 1, Padding, false);

            Parallel.For(0, KernelCount,  new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, i =>
            {
                    
                for (int j = 0; j < jBound; j++)
                {
                    for (int k = 0; k < kBound; k++)
                    {       
                        int outIndex = OutputCoordinates.GetIndex(i, j, k);
                        var tmp = default(NumT).Const(0.0);
                        ApplyKernel<NumT, T, V>(ref tmp, input, Padding, i, j, k);
                        output[outIndex] = tmp;
                    }
               }
            });
        }


        public override Vector<double> EvaluateConcrete(Vector<double> input)
        {
 
            var x = DoConvolution(input);
            return x;

            // Old code, slightly slower:
            // return Evaluate<NumInstDouble,double,Vector<double>>(input);
        }
        public override LPSTerm[] EvaluateSymbolic(LPSState state, LPSTerm[] input)
        {
            if (!symbolic_output_storage.IsValueCreated)
            {
                symbolic_output_storage.Value = default(NumInstLPSTermArr).CreateVector(OutputDimension);
            }

            Evaluate<NumInstLPSTermArr, LPSTerm, LPSTerm[]>(input, symbolic_output_storage.Value);
            return symbolic_output_storage.Value;

        }

        public override bool IsAffine()
        {
            return true;
        }

    }
}
