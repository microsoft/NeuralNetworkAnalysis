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

using NNAnalysis;
using NNAnalysis.Utils;

using System.Diagnostics;


namespace NNCIFARDataTests
{
    class Program
    {
        static int SplitSize = 100;

        static void Main(string[] args)
        {

            string CifarDataBatch = args[0];

            /* Initialize parameters */
            Options.InitializeNNAnalysis();

            // Plain old CIFAR binary format        
            ImageDataset data = CIFAR.ReadData(CifarDataBatch, CIFAR.ALL_IMAGES, 0);
            Trace.Assert(!data.Dataset.HasUninitialized());


            for (SplitSize = 100; SplitSize < 10000; SplitSize += 115)
            {
                Console.WriteLine("Splitting, SplitSize = {0}", SplitSize);

                // Split
                var splits = data.ShuffleSplitMany(SplitSize);

                Console.WriteLine("Shuffle-split to {0} splits", splits.Count());

                foreach (var s in splits)
                {
                    Trace.Assert(!s.Dataset.HasUninitialized());
                }

                Console.WriteLine("Joining back");
                // Union 
                data = Data.UnionMany(splits);

                Trace.Assert(!data.Dataset.HasUninitialized());

            }

        }
    }
}
