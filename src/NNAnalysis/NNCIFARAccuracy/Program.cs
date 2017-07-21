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

        string CifarNNFile = null;
        string CifarDataBatch = null;

        bool just_accuracy = false;
        bool just_loss = false;

        bool raw_directory = false;

        var p = new OptionSet();
        p.Add<string>("nnet=",           "CIFAR neural network file name",   (x => CifarNNFile = x));
        p.Add<string>("dataset=",        "CIFAR dataset file name",          (x => CifarDataBatch = x));
        p.Add<string>("rawdir", "If set then --dataset value should be a directory in raw directory format", (x => raw_directory = (x != null)));
        p.Add<bool>  ("optimization=",   "Do optimization (Default: true)" , (x => RobustnessOptions.DoOptimization = x));
        p.Add<double>("sub=",            "Subsample from 'live' constraints (0.0-1.0)", (x => RobustnessOptions.LiveConstraintSamplingRatio = x));
        p.Add<string>("registry=",       "Unique name to store output examples and statistics", (x => RobustnessOptions.Registry = x));
        p.Add<bool>  ("cegar=",          "Do CEGAR (default: true)",         (x => RobustnessOptions.CEGAR = x));
        p.Add<string>("only-accuracy",   "Only evaluate accuracy",           (x => just_accuracy = (x != null)));
        p.Add<string>("only-loss",       "Only evaluate loss",               (x => just_loss = (x != null)));

        p.Add<double>("bound=", "Linfinity-ball to search", (x => RobustnessOptions.Epsilon = x));
        p.Add<double>("minval=", "Minimum value of each entry", (x => RobustnessOptions.MinValue = x));
        p.Add<double>("maxval=", "Maximum value of each entry", (x => RobustnessOptions.MaxValue = x));
        p.Add<string>("no-quant-safety", "Quantization integrality safety off", (x => RobustnessOptions.QuantizationSafety = (x == null)));

        p.Add<double>("scale-preprocessed=", "If image data is preprocessed, scale before dumping to registry", (x => RobustnessOptions.ScalePreProcessed = x));
        p.Add<double>("offset-preprocessed=", "If image data is preprocessed, offset scaled before dumping to registry", (x => RobustnessOptions.OffsetPreProcessed = x));

        p.Add<string>("max-conf", "Use max-conf objective", (x =>
        {
            if (x != null) { RobustnessOptions.ObjectiveKind = LPSObjectiveKind.MaxConf; }
        }));

        p.Add<double>("winner-diff=", "Winning label should be that much different than second best", (x => RobustnessOptions.LabelConfidenceDiff = x));
        p.Add<string>("log-png", "Log png files", (x => RobustnessOptions.SavePNGCounterexamples = (x != null)));

        bool only_filter = false;
        double filter_conf = 0.98;
        p.Add("only-filter", "Only filter by confidence", (x => only_filter = (x != null)));
        p.Add<double>("filter-conf=", "Filter confidence", (x => filter_conf = x));



        Cmd.RunOptionSet(p, args);

        if (CifarNNFile == null || CifarDataBatch == null)
        {
            Console.WriteLine("Invalid arguments, use --help");
            Environment.Exit(1);
        }

        /* Initialize parameters */
        Options.InitializeNNAnalysis();
        NeuralNet nn = CIFAR.GetNN(CifarNNFile);


        ImageDataset data;

        if (raw_directory)
        {
            // our raw data format (see lmdb2raw.py)
            data = CIFAR.ReadDirectoryData(CifarDataBatch);
        }
        else
        {
            // Plain old CIFAR binary format        
            data = CIFAR.ReadData(CifarDataBatch, CIFAR.ALL_IMAGES, 0);
        }

        if (just_accuracy)
        {
            NNAccuracy.GetAccuracy(nn,data.Dataset);
            return;
        }

        if (just_loss)
        {
            NNAccuracy.GetLoss(nn, data.Dataset);
            return;
        }


        if (only_filter)
        {
            string filtered = RobustnessOptions.Registry + "-filtered-" + filter_conf.ToString();

            Console.WriteLine("Orig {0} data", data.Dataset.Count());

            var ds = NNAccuracy.KeepAboveConfidenceThreshold(nn, data.Dataset, filter_conf);

            Console.WriteLine("Kept {0} data", ds.Count());

            ImageDataset ret = new ImageDataset(ds,
                CIFAR.InputCoordinates.ChannelCount,
                CIFAR.InputCoordinates.RowCount,
                CIFAR.InputCoordinates.ColumnCount, true);

            CIFAR.WriteData(filtered, ret);
            return;

        }



        RobustnessOptions.Dump();


        string synthImagesName = RobustnessOptions.Registry + "-synth";

        int labelcount = data.Dataset.LabelCount();

        ImageDataset acc = new ImageDataset(new Dataset(labelcount),
            CIFAR.InputCoordinates.ChannelCount,
            CIFAR.InputCoordinates.RowCount,
            CIFAR.InputCoordinates.ColumnCount, true);
        int state = 0;

        Action<LabelWithConfidence> snapshot = x =>
            {
                acc.Dataset.Data.Add(new MemAccessor<double[]>(x.datum));
                acc.Dataset.Labels.Add(new MemAccessor<int>(x.actualLabel));
                state++;
                if (state >= 4)
                {
                    CIFAR.WriteData(synthImagesName, acc);
                    state = 0;
                }
            };

        ImageDataset synth = Robustness.SynthesizeCounterexamplesAndStore(nn, data, snapshot);

        if (synth.Dataset.Count() == 0)
        {
            Console.WriteLine("Did not synthesize any counterexamples, nothing to dump ...");
            return;
        }

        if (raw_directory)
        {
            throw new NotImplementedException("Output to raw directory format not yet implemented!");
        }
        else
        {
            CIFAR.WriteData(RobustnessOptions.Registry + "-synth", synth);
        }

    }
}
