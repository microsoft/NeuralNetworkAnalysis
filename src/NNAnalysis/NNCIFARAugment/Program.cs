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

        int how_many = 1;
        RANDTYPE randomness = RANDTYPE.UNIFORM;

        var p = new OptionSet();
        
        p.Add<string>("dataset=",      "CIFAR dataset file name",          (x => CifarDataBatch = x));
        p.Add<int>("how-many=",        "Number of new images per image",   (x => how_many = x));
        p.Add<string>("randomness=",   "Gaussian|Uniform",                 (x => randomness = (x.Equals("Gaussian") ? RANDTYPE.GAUSSIAN : RANDTYPE.UNIFORM)));

        int xoffset = 0;
        int yoffset = 0;
        bool geometric = false;
        p.Add("geometric", "Use geometric transform", (x => geometric = (x != null)));
        p.Add<int>("xoffset=", "x-offset for geometric transform", (x => xoffset = x));
        p.Add<int>("yoffset=", "y-offset for geometric transform", (x => yoffset = x));

        bool random = false;
        double epsilon = 0.0;
        p.Add("random", "Use random perturbation", (x => random = (x != null)));
        p.Add<double>("epsilon=", "Distance (for uniform) or standard deviation (for gaussian) random perturbation", (x => epsilon = x));

        bool brightness = false;
        double brightness_offset = 0.0;
        p.Add("brightness", "Use brightness perturbation", (x => brightness = (x != null)));
        p.Add<double>("brightness-offset=", "Brightness offset (<= RobustnessOptions.MaxValue - RobustnessOptions.MinValue)", (x => brightness_offset = x));

        bool contrast = false;
        double contrast_min_factor = 1.0;
        double contrast_max_factor = 1.0;
        p.Add("contrast", "Use contrast perturbation", (x => contrast = (x != null)));
        p.Add<double>("contrast-min-factor=", "Contrast min factor (0.0-1.0)", (x => contrast_min_factor = x));
        p.Add<double>("contrast-max-factor=", "Contrast max factor (0.0-1.0)", (x => contrast_max_factor = x));


        bool lossy_jpeg = false;
        int photoquality = 90;
        p.Add("lossy-jpeg", "Use lossy jpeg perturbation (default photoquality = 90)", (x => lossy_jpeg = (x != null)));
        p.Add<int>("jpeg-photoquality=", "Lossy jpeg photoquality", (x => photoquality = x));


        bool rotate = false;
        float angle = 45.0F;
        p.Add("rotation", "Rotation transformation (default angle = 45.0)", (x => rotate = (x != null)));
        p.Add<double>("rotation-angle=", (x => angle = (float)x));


        bool perturbe_only = false;
        p.Add("perturbe-only", "Only perturbe (not augment)", (x => perturbe_only = (x != null)));

        Cmd.RunOptionSet(p, args);

        if (CifarDataBatch == null)
        {
            Console.WriteLine("Invalid arguments, use --help");
            Environment.Exit(1);
        }

        /* Initialize parameters */
        Options.InitializeNNAnalysis();

        // Plain old CIFAR binary format        
        ImageDataset data = CIFAR.ReadData(CifarDataBatch, CIFAR.ALL_IMAGES, 0);

        IAugmentor augmentor = null; // TODO

        if (geometric)
        {
            augmentor = new AugmentGeometric(CIFAR.InputCoordinates, randomness, how_many, xoffset, yoffset);
            goto KONT;
        }
        if (random)
        {
            augmentor = new AugmentRandom(CIFAR.InputCoordinates, randomness, how_many, epsilon);
            goto KONT;
        }
        if (brightness)
        {
            augmentor = new AugmentBrightness(CIFAR.InputCoordinates, randomness, how_many, brightness_offset);
            goto KONT;
        }
        if (contrast)
        {
            augmentor = new AugmentContrast(CIFAR.InputCoordinates, how_many, contrast_min_factor, contrast_max_factor);
            goto KONT;
        }
        if (lossy_jpeg)
        {
            augmentor = new AugmentLossyJpeg(CIFAR.InputCoordinates, how_many, photoquality);
            goto KONT;
        }
        if (rotate)
        {
            augmentor = new AugmentRotation(CIFAR.InputCoordinates, how_many, angle);
            goto KONT;
        }


KONT:

        int count = data.Dataset.Count();

        ImageDataset initial = null;

        if (perturbe_only)
        {
            initial = new ImageDataset(new Dataset(10), CIFAR.InputCoordinates.ChannelCount,
                CIFAR.InputCoordinates.RowCount,
                CIFAR.InputCoordinates.ColumnCount, true);
        }
        else
            initial = data;

        for(int i = 0; i < count; i++)
        {
            double[] datum = data.Dataset.GetDatum(i);
            int label = data.Dataset.GetLabel(i);
            var augmented = augmentor.Augment(datum);
            initial.Update(augmented, label);
        }

        if (perturbe_only)
        {
            CIFAR.WriteData(CifarDataBatch + ".perturbed", initial);
        }
        else
        {
            CIFAR.WriteData(CifarDataBatch + ".augmented", initial);
        }

    }

}
