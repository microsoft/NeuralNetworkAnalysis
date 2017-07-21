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
using Mono.Options;

class Program
{
        static void Main(string[] args)
        {

        string CifarDataBatch = null;

        int split_size = 0;

        var p = new OptionSet();
        p.Add<string>("dataset=", "CIFAR dataset file name",    (x => CifarDataBatch = x));
        p.Add<int>("split-size=", "Number of images per split", (x => split_size = x));

        Cmd.RunOptionSet(p, args);

        if (CifarDataBatch == null || split_size <= 0)
        {
            Console.WriteLine("Invalid arguments, use --help");
            Environment.Exit(1);
        }

        /* Initialize parameters */
        Options.InitializeNNAnalysis();

        // Plain old CIFAR binary format        
        ImageDataset data = CIFAR.ReadData(CifarDataBatch, CIFAR.ALL_IMAGES, 0);

        // Split
        var splits = data.ShuffleSplitMany(split_size);
        int count = 0;
        foreach (var s in splits)
        {
            CIFAR.WriteData(CifarDataBatch + ".split_" + count.ToString(), s);
            count++;
        }

    }

}

