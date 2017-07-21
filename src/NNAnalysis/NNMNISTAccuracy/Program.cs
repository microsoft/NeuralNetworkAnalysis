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
        
        string MNISTFile = null;
        string MNISTData = null;
        string MNISTLabels = null;
        var p = new OptionSet();


        bool just_accuracy = false;
        bool just_loss = false;

        p.Add("nnet=",      "MNIST neural network file name", x => MNISTFile = x);
        p.Add("datafile=",  "MNIST data file name",           x => MNISTData = x);
        p.Add("labelfile=", "MNIST label file name",          x => MNISTLabels = x);
        p.Add<bool>("optimization=", "Do optimization (Default: true)", (x => RobustnessOptions.DoOptimization = x));
        p.Add<double>("bound=", "Linfinity-ball to search", (x => RobustnessOptions.Epsilon = x));
        p.Add<double>("sub=", "Subsample from 'live' constraints (0.0-1.0)", (x => RobustnessOptions.LiveConstraintSamplingRatio = x));
        p.Add<string>("registry=", "Unique name to store output examples and statistics", (x => RobustnessOptions.Registry = x));
        p.Add<bool>("cegar=", "Do CEGAR (default: true)", (x => RobustnessOptions.CEGAR = x));
        p.Add<string>("only-accuracy", "Only evaluate accuracy", (x => just_accuracy = (x != null)));
        p.Add<string>("only-loss", "Only evaluate loss", (x => just_loss = (x != null)));

        p.Add<string>("no-quant-safety", "Quantization integrality safety off", (x => RobustnessOptions.QuantizationSafety = (x == null)));


        p.Add<string>("max-conf", "Use max-conf objective", (x => { 
            if (x != null) { RobustnessOptions.ObjectiveKind = LPSObjectiveKind.MaxConf;}
        }));

        p.Add<double>("winner-diff=", "Winning label should be that much different than second best", (x => RobustnessOptions.LabelConfidenceDiff = x));


        p.Add<string>("log-png", "Log png files", (x => RobustnessOptions.SavePNGCounterexamples = (x != null)));

        bool only_misclass = false;
        p.Add("only-filter-misclass", "Only keep the misclassifications", (x => only_misclass = (x != null)));



        Cmd.RunOptionSet(p, args);

        if (MNISTFile == null || MNISTData == null || MNISTLabels == null)
        {
            Console.WriteLine("Invalid arguments, use --help");
            Environment.Exit(1);
        }

        RobustnessOptions.Dump();

        Options.InitializeNNAnalysis();

        NeuralNet nn = MNIST.GetNN(MNISTFile);
        ImageDataset data = MNIST.ReadData(MNISTLabels, MNISTData, MNIST.ALL_IMAGES, 0);


        if (just_accuracy)
        {
            NNAccuracy.GetAccuracy(nn, data.Dataset);
            return;
        }

        if (just_loss)
        {
            NNAccuracy.GetLoss(nn, data.Dataset);
            return;
        }


        if (only_misclass)
        {
            string filtered = RobustnessOptions.Registry + "-misclass";

            Console.WriteLine("Orig {0} data", data.Dataset.Count());

            var ds = NNAccuracy.KeepMisclass(nn, data.Dataset);

            Console.WriteLine("Kept {0} data", ds.Count());

            ImageDataset ret = new ImageDataset(ds,
                MNIST.InputCoordinates.ChannelCount,
                MNIST.InputCoordinates.RowCount,
                MNIST.InputCoordinates.ColumnCount, true);

            MNIST.WriteData(filtered + "-labels", filtered + "-images", ret);
            return;

        }

        // NB: No snapshotting for MNIST since it never crashes ... 
        ImageDataset synth = Robustness.SynthesizeCounterexamplesAndStore(nn, data, x => { return; } );
        MNIST.WriteData(RobustnessOptions.Registry + "-synth-labels", 
                        RobustnessOptions.Registry + "-synth-images", synth);


    }
}
