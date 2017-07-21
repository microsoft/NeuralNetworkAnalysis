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

using System.Diagnostics;

namespace NNAnalysis
{
    public enum LPSObjectiveType
    {
        Min, Max
    }

    public enum LPSObjectiveKind
    {
        MinLinf, MaxConf
    }

    public struct LPSObjective
    {
        public LPSTerm term;
        public LPSObjectiveType type;
    }


    public class NNETObjectives
    {
        /// <summary>
        /// Create formulae of the form:  <code> -epsilon &lt input[i] - origin[i] &lt epsilon </code>
        /// </summary>
        /// 

        public static void AddEpsilonBounds(LPSConstraints cts, LPSTerm[] input, LPSTerm epsilon, double[] origin)
        {
            

            for (int i = 0; i < origin.Length; i++)
            {
                var curr = input[i];

                // i.e: origin[i] - epsilon < input[i]
                var tmp = LPSTerm.Const(origin[i]);
                tmp.Sub(epsilon);
                cts.And(tmp, InequalityType.LE, curr);

                // and: input[i] < epsilon + origin[i]
                tmp = LPSTerm.Const(origin[i]);
                tmp.Add(epsilon);
                cts.And(curr, InequalityType.LE, tmp);
            }

            cts.And(epsilon, InequalityType.GT, LPSTerm.Const(0.0)); // Quantization error!
            cts.And(epsilon, InequalityType.LE, LPSTerm.Const(Utils.RobustnessOptions.Epsilon));

        }

        public static void AddQuantizationSafety(LPSConstraints cts, LPSTerm[] input, double[] origin)
        {
            Random r = new Random();
            int i = r.Next(0,origin.Length - 1);

            LPSTerm curr = input[i];

            // i.e: origin[i] - epsilon < input[i]
            var tmp = LPSTerm.Const(origin[i] + 1.0);
            cts.And(tmp, InequalityType.LE, curr);
        }



        public static LPSObjective MinLInf(LPSConstraints cts, LPSTerm[] input, LPSTerm epsilon, double[] origin)
        {
            return (new LPSObjective { term = epsilon, type = LPSObjectiveType.Min });

        }
        public static LPSObjective MaxConf(LPSTerm[] output, int origLabel, int newLabel)
        {
            var tmp = LPSTerm.Const(0.0);
            tmp.Add(output[newLabel]);
            tmp.Sub(output[origLabel]);

            return (new LPSObjective { term = tmp, type = LPSObjectiveType.Max });

        }

    }


    public class NNetFormulas
    {
        /// <summary>
        /// LabelFormula(output,label,confidence) gives back a formula expressing
        /// that: for all i s.t. i != label, output[label] - output[i] >= confidence
        /// </summary>
        /// <param name="output">Output of neural network (before softmax, as given by our evaluator).</param>
        /// <param name="label">The label we wish to win.</param>
        /// <param name="confidence">A confidence interval for all comparisons (e.g. for quantization etc).</param>
        /// <returns>The constraint expressing that our label is indeed the winning one. </returns>
        public static LPSConstraints LabelFormula(LPSTerm[] output, int label, double confidence = 0f)
        {
            LPSConstraints ct = new LPSConstraints();

            for (int i = 0; i < output.Length; i++)
            {
                if (i != label)
                {
                    // Need: output[label] - output[i] >= confidence 
                    // i.e.: output[label] - output[i] - confidence >= 0
                    var tmp = LPSTerm.Const(0.0);   // tmp := 0
                    tmp.Add(output[label]);         // tmp := output[label]
                    tmp.AddMul(output[i], -1.0);    // tmp := output[label] - output[i]
                    tmp.Add(-1.0 * confidence);     // tmp := output[label] - output[i] - confidence

                    ct.And(tmp, InequalityType.GE);
                }
            }
            return ct;
        }

        /// <summary>
        /// Ensures that the input is within an originBound ball of origin, or within 0.0f - 255f, 
        /// whichever is tightest.
        /// </summary>
        /// <returns></returns>
        public static LPSConstraints OriginBoundFormula(LPSTerm[] input, double[] origin, double originBound)
        {
            Debug.Assert(input.Length == origin.Length);

            LPSConstraints ct = new LPSConstraints();

            for (int i = 0; i < input.Length; i++)
            {

                double ub = Math.Min(Utils.RobustnessOptions.MaxValue, origin[i] + originBound);
                double lb = Math.Max(Utils.RobustnessOptions.MinValue, origin[i] - originBound);


                if (lb <= ub)
                {
                    var tmp = LPSTerm.Const(ub);
                    ct.And(input[i], InequalityType.LE, tmp);
                    tmp = LPSTerm.Const(lb);
                    ct.And(input[i], InequalityType.GE, tmp);
                }
                else
                {
                    var tmp = LPSTerm.Const(origin[i] + originBound);
                    ct.And(input[i], InequalityType.LE, tmp);
                    tmp = LPSTerm.Const(origin[i] - originBound);
                    ct.And(input[i], InequalityType.GE, tmp);
                }
            }

                return ct;
        }

    }
}
