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
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Diagnostics;

namespace NNAnalysis
{
    /// <summary>
    /// Constraint operator
    /// </summary>
    public enum InequalityType
    {
        EQ, GE, GT, LE, LT
    }

    /// <summary>
    /// If x = Term, op = Inequality, represents  x `op` 0
    /// </summary>
    public struct LPSConstraint
    {
        public LPSTerm Term;
        public InequalityType Inequality;

        public bool Added; // default = false. Have we added this constraint to the solver? No. Yikes.

    }

    /// <summary>
    /// A simple (list-implemented) set of constraints and inequalities that
    /// encode an LP instance. The list-based implementation exists because 
    /// we only iterate/insert. If we need more, make sure to optimize this!
    /// </summary>
    public class LPSConstraints : IEnumerable
    {
        // To support O(1) extension
        List<LPSConstraint> constraints_ = new List<LPSConstraint>();
        // To support O(1) union 
        List<LPSConstraints> unioned_ = new List<LPSConstraints>();

        int constraintCount_ = 0;
        public void And(LPSConstraints s)
        {
            unioned_.Add(s);
            constraintCount_ += s.constraintCount_;
        }
        public void And(LPSTerm term, InequalityType inequality)
        {
            constraints_.Add(new LPSConstraint { Inequality = inequality, Term = term });
            constraintCount_++;
        }

        // (left `binop` right), equivalently: (left - right `binop` 0)
        // NB: Allocates, does not overwrite the left or right term.
        public void And(LPSTerm left, InequalityType inequality, LPSTerm right)
        {
            LPSTerm t = LPSTerm.Const(0.0);
            t.Add(left);
            t.AddMul(right, -1.0);
            And(t, inequality);
        }

        public int Count
        {
            get { return constraintCount_; }
        }

        public IEnumerator GetEnumerator()
        {

            foreach (var o in constraints_) yield return o;

            foreach (var s in unioned_)
            {
                foreach (var o in s.constraints_) yield return o;
            }

        }

        public List<LPSConstraint> ToList()
        {
            List<LPSConstraint> ret = new List<LPSConstraint>(constraints_);
            foreach (var s in unioned_)
            {
                foreach (var o in s.constraints_)
                {
                    ret.Add(o);
                }
            }
            return ret;
        }


    }

}
