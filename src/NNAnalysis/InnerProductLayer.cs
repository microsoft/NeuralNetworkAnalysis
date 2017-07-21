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

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

using MathNet.Numerics;

namespace NNAnalysis
{

    public class InnerProductLayer : Layer
    {

        Matrix<double> weightMatrix_;
        Vector<double>[] weightMatrixRows_;
        Vector<double> interceptVector_;
        ImageCoordinates inputCoordinates_;
        public Matrix<double> WeightMatrix
        {
            get { return weightMatrix_; }
        }
        public Vector<double> InterceptVector
        {
            get { return interceptVector_; }
        }


        public InnerProductLayer(int index, Tuple<Matrix<double>, Vector<double>> ws_and_is, ImageCoordinates inputCoordinates)
            : this(index,ws_and_is.Item1,ws_and_is.Item2,inputCoordinates) { }

        public InnerProductLayer(int index, Matrix<double> weights, Vector<double> intercepts, ImageCoordinates inputCoordinates)
        {
            weightMatrix_ = weights;

            weightMatrixRows_ = new Vector<double>[weightMatrix_.RowCount];
            for (int i = 0; i < weightMatrix_.RowCount; i++)
            {
                weightMatrixRows_[i] = weightMatrix_.Row(i);
            }

            interceptVector_ = intercepts;
            inputCoordinates_ = inputCoordinates;
            InitLayer(index, LayerType.INNER_PRODUCT, weights.ColumnCount, weights.RowCount, inputCoordinates, null);

        }


        public InnerProductLayer(int index, double[][] weights, double[] intercepts, ImageCoordinates inputCoordinates)
            : this(index,SparseMatrix.OfRowArrays(weights), DenseVector.OfArray(intercepts),inputCoordinates) {}


        public override Vector<double> EvaluateConcrete(Vector<double> v)
        {
            return (weightMatrix_ * v + interceptVector_);
        }
        public override void Instrument(NNInstrumentation instr, Vector<double> input, Vector<double> output)
        {
            instr[Index] = Instrumentation.NoInstrumentation();
        }
        public override LPSTerm[] EvaluateSymbolic(LPSState state, LPSTerm[] input)
        {
            LPSTerm[] output = new LPSTerm[OutputDimension];
            for (int i = 0; i < OutputDimension; i++)
            {
                output[i] = doInnerProduct(state, input, weightMatrixRows_[i]);
                output[i].Add(interceptVector_[i]);
            }
            return output;
        }
        private LPSTerm doInnerProduct(LPSState state, LPSTerm[] vs, Vector<double> ds)
        {
            LPSTerm result = LPSTerm.Const(0.0);
            for (int i = 0; i < vs.Length; i++)
            {
                result.AddMul(vs[i], ds[i]);
            }
            return result;
        }


        public override bool IsAffine()
        {
            return true;
        }


    }

}
