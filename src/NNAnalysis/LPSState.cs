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
using System.IO;

namespace NNAnalysis
{

    public enum DisjunctionChoice
    {
        ACTIVE, INACTIVE, EITHER
    }

    public class DCSComparer : IEqualityComparer<DisjunctionChoice[]>
    {
        public bool Equals(DisjunctionChoice[] a, DisjunctionChoice[] b)
        {
            return Instrumentation.EqualDisjunctionChoices(a, b);
        }

        public int GetHashCode(DisjunctionChoice[] a)
        {
            int ret = 0;
            for (int i = 0; i < a.Length; i++)
            {
                if (a[i] == DisjunctionChoice.ACTIVE)
                {
                    ret++;
                }
            }
            return ret;
        }
    }




    public class Instrumentation
    {
        public LayerType LayerType;
        public DisjunctionChoice[] DisjunctionConstraints = null; // RELU
        public int[] Selections = null;                           // MAXPOOLING

        private Instrumentation() { }

        public static bool EqualDisjunctionChoices(DisjunctionChoice[] a, DisjunctionChoice[] b)
        {
            for (int i=0; i< Math.Min(a.Length,b.Length); i++)
            {
                if (a[i] != b[i])
                {
                    return false;
                }
            }
            return true;
        }



        public static string DisjunctionChoiceStr(DisjunctionChoice dc)
        {
            switch (dc)
            {
                case DisjunctionChoice.ACTIVE: return "A";
                case DisjunctionChoice.INACTIVE: return "I";
                case DisjunctionChoice.EITHER: return "E";
                default:
                    throw new Exception("FlipDisjunctionChoice: can't happen");
            }
        }


        public static Dictionary<KeyValuePair<int,int>,List<DisjunctionChoice[]>> DCLog = new Dictionary<KeyValuePair<int, int>, List<DisjunctionChoice[]>>();
        public static int Collisions = 0;

        public static void InitReLULogging()
        {
                    DCLog = new Dictionary<KeyValuePair<int, int>, List<DisjunctionChoice[]>>();
                    Collisions = 0;
        }

    public static void LogDisjunctionChoices(string fn, int layeridx, DisjunctionChoice[] dcs)
        {

            int hash = new DCSComparer().GetHashCode(dcs);
            var idx = new KeyValuePair<int, int>(layeridx, hash);

            if (DCLog.ContainsKey(idx))
            {
                var entries = DCLog[idx];
                foreach (var entry in entries)
                {
                    if (EqualDisjunctionChoices(dcs, entry))
                    {
                        Collisions++;
                        return;
                    }
                }
                entries.Add(dcs);
            }
            else
            {
                DCLog[idx] = new List<DisjunctionChoice[]>();
                DCLog[idx].Add(dcs);
            }
        }



        public static DisjunctionChoice FlipDisjunctionChoice(DisjunctionChoice dc)
        {
            switch(dc)
            {
                case DisjunctionChoice.ACTIVE: return DisjunctionChoice.INACTIVE;
                case DisjunctionChoice.INACTIVE: return DisjunctionChoice.ACTIVE;
                case DisjunctionChoice.EITHER: return DisjunctionChoice.EITHER;
                default:
                    throw new Exception("FlipDisjunctionChoice: can't happen");
            }
        }
        public static Instrumentation ReLUInstrumentation(DisjunctionChoice[] choices)
        {
            Instrumentation ret = new Instrumentation();
            ret.LayerType = LayerType.RECTIFIED_LINEAR;
            ret.DisjunctionConstraints = choices;
            ret.Selections = null;
            return ret;
        }
        public static Instrumentation MaxPoolingInstrumentation(int[] choices)
        {
            Instrumentation ret = new Instrumentation();
            ret.LayerType = LayerType.POOLING_LAYER;
            ret.DisjunctionConstraints = null;
            ret.Selections = choices;
            return ret;
        }
        public static Instrumentation NoInstrumentation()
        {
            Instrumentation ret = new Instrumentation();
            return ret;
        }
    }

    public class NNInstrumentation : Dictionary<int, Instrumentation> { };
    
    public class LPSState
    {
        double[] origin_;
        // The ones we cache for a different round of CEGAR
        LPSConstraints deferredConstraints_;
        // The ones we collect as we interpret
        LPSConstraints currentConstraints_;
        // NB: null for the layers where we have no instrumentation
        NNInstrumentation instrumentation_;
        public NNInstrumentation Instrumentation
        {
            get { return instrumentation_; }
        }

        public double[] Origin
        {
            get { return origin_; }
        }

        public void ClearConstraints()
        {
            deferredConstraints_ = new LPSConstraints();
            currentConstraints_ = new LPSConstraints();
        }

        public LPSConstraints CurrentCts
        {
            get { return currentConstraints_;  }
        }
        public LPSConstraints DeferredCts
        {
            get { return deferredConstraints_;  }
        }
        public LPSState(NNInstrumentation instrumentation, double[] origin)
        {
            deferredConstraints_ = new LPSConstraints();
            currentConstraints_ = new LPSConstraints();
            instrumentation_ = instrumentation;
            origin_ = origin;
        }
    }

}