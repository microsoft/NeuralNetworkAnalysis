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

using System.Diagnostics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace NNAnalysis
{

    public class ReLULayer : Layer
    {
        int dimension_;

        public ReLULayer(int index, int dimension, ImageCoordinates coordinates)
        {
            dimension_ = dimension;
            InitLayer(index, LayerType.RECTIFIED_LINEAR, dimension, dimension, coordinates, coordinates);
        }
        public int Dimension
        {
            get { return dimension_; }
        }
        private bool Active(double x)
        {
            return (x >= 0.0);
        }

        public override Vector<double> EvaluateConcrete(Vector<double> v)
        {
            var res = v.Map(x => Active(x)? x : 0.0 );

            if (Utils.RobustnessOptions.ReLULogFile != "")
            {
                var disjunctionChoices = CreateDisjunctionChoices(v);
                Instrumentation.LogDisjunctionChoices(Utils.RobustnessOptions.ReLULogFile, Index, disjunctionChoices);
            }

            return res;
        }

        public DisjunctionChoice[] CreateDisjunctionChoices(Vector<double> input)
        {
            DisjunctionChoice[] disjunctionChoices = new DisjunctionChoice[input.Count];
            for (int i = 0; i < input.Count; i++)
            {
                disjunctionChoices[i] = Active(input[i]) ? DisjunctionChoice.ACTIVE : DisjunctionChoice.INACTIVE;
            }
            return disjunctionChoices;
        }

        public override void Instrument(NNInstrumentation instr, Vector<double> input, Vector<double> output)
        {
            var disjunctionChoices = CreateDisjunctionChoices(input);

            instr[Index] = Instrumentation.ReLUInstrumentation(disjunctionChoices);

        }

        public bool IsActivationWobbly(LPSTerm input, double[] image)
        {
            double icpt = input.Intercept;
            Vector<double> imagecoeffs = input.GetCoefficients().SubVector(0, image.Length);
            double innerprod = imagecoeffs * DenseVector.OfArray(image);

            double shouldIncrease = (innerprod + icpt < 0) ? 1.0 : -1.0;
            Vector<double> signVec = imagecoeffs.Map(x => (x >= 0) ? 1.0 : -1.0);

            // Adversarial image:
            Vector<double> adversarial_image = DenseVector.OfArray(image);
            for (int i=0; i < image.Length; i++)
            {
                adversarial_image[i] += shouldIncrease * signVec[i] * 0.5 * Utils.RobustnessOptions.Epsilon;
            }

            //Console.WriteLine("Original activation:    {0}", innerprod + icpt);
            //Console.WriteLine("Adversarial activation: {0}", imagecoeffs * adversarial_image + icpt);
            //Console.Read();

            return (Math.Sign(innerprod + icpt) != Math.Sign(imagecoeffs * adversarial_image + icpt));
        }

        public override LPSTerm[] EvaluateSymbolic(LPSState state, LPSTerm[] input)
        {
            DisjunctionChoice[] disjunctionChoices = state.Instrumentation[Index].DisjunctionConstraints;
            Debug.Assert(InputDimension == disjunctionChoices.Length);
            LPSTerm[] output = new LPSTerm[OutputDimension];

            Random r = new Random(System.DateTime.Now.Millisecond);
            // int uncertain = 0;

            for (int i = 0; i < OutputDimension; i++)
            {
                switch (disjunctionChoices[i])
                {
                    case DisjunctionChoice.ACTIVE:
                        output[i] = input[i];
                        // If we are supposed to do sampling
                        if (Utils.RobustnessOptions.LiveConstraintSamplingRatio != 1.0)
                        {
                            // Console.WriteLine("Sampling!");
                            // if we are above threshold defer
                            if (r.Next(0, 100) > (int) Utils.RobustnessOptions.LiveConstraintSamplingRatio * 100)
                            {
                                state.DeferredCts.And(input[i], InequalityType.GE);
                            }
                            else
                            {
                                state.CurrentCts.And(input[i], InequalityType.GE);
                            }
                        }
                        else
                        {
                            state.CurrentCts.And(input[i], InequalityType.GE);
                        }
                        break;
                    case DisjunctionChoice.INACTIVE:
                        output[i] = LPSTerm.Const(0.0);
                        // CEGAR version: defer 'dead' constraint
                        state.DeferredCts.And(input[i], InequalityType.LT);
                        // Original version
                        // state.CurrentCts.And(input[i],InequalityType.LT);
                        break;
                    default:
                        throw new Exception("Invalid disjunction choice type!");
                }

                /* This is more of an experiment really ...
                if (IsActivationWobbly(input[i], state.Origin))
                {
                    uncertain++;
                    // Mutate state to have new disjunction choices to explore in the future
                    disjunctionChoices[i] = Instrumentation.FlipDisjunctionChoice(disjunctionChoices[i]);
                }
                */

            }

            // Console.WriteLine("** Ultra-sensitive ReLU activations {0}/{1}", uncertain, OutputDimension);
 
            return output;
        }

        public override bool IsAffine()
        {
            return false;
        }

    }
}