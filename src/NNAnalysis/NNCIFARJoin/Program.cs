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
        string[] split_files = null;
        
        var p = new OptionSet();
        p.Add<string>("dataset=", "CIFAR dataset file name to store result",    (x => CifarDataBatch = x));
        p.Add<string>("files=", "CIFAR dataset files to join", (x => split_files = x.Split()));
        Cmd.RunOptionSet(p, args);

        if (CifarDataBatch == null || split_files == null)
        {
            Console.WriteLine("Invalid arguments, use --help");
            Environment.Exit(1);
        }

        List<ImageDataset> dss = new List<ImageDataset>();
        Console.WriteLine("Joining files ...");
        for (int i=0; i < split_files.Length; i++)
        {
            Console.WriteLine(split_files[i]);
            dss.Add(CIFAR.ReadData(split_files[i],CIFAR.ALL_IMAGES, 0));
        }

        var data = Data.UnionMany(dss);

        var sd = data.Dataset.CreateShuffle(new Random());

        data.Dataset = sd;


        Console.WriteLine("Output file ...");
        Console.WriteLine(CifarDataBatch);

        CIFAR.WriteData(CifarDataBatch,data);

    }

}


