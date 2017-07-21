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

namespace NNAnalysis
{
    namespace Utils
    {
        public class NNAccuracy
        {
            public static object lockObject = new Object();

            public static Dataset Filter(NeuralNet nn, Dataset ds, Func<NeuralNet,double[],int,bool> predicate)
            {

                Dataset ret = new Dataset(ds.LabelCount());

                for (int i = 0; i < ds.Count(); i++)
                {
                    double[] datum;
                    int ground_label;

                    datum = ds.GetDatum(i);
                    ground_label = ds.GetLabel(i);

                    if (predicate(nn,datum,ground_label))
                    {
                        ret.Data.Add(new MemAccessor<double[]>(datum));
                        ret.Labels.Add(new MemAccessor<int>(ground_label));
                    }

                }

                return ret;
            }

            public static Dataset KeepAboveConfidenceThreshold(NeuralNet net, Dataset ds, double conf)
            {

                return Filter(net,ds, (nn, datum, ground_label) => {
                    LabelWithConfidence lab = Utils.ULabel.LabelWithConfidence(nn, datum, true);
                    return (lab.softMaxValue >= conf); 
                    });
            }

            public static Dataset KeepMisclass(NeuralNet net, Dataset ds)
            {
                return Filter(net, ds, (nn, datum, ground_label) =>
                {
                    LabelWithConfidence lab = Utils.ULabel.LabelWithConfidence(nn, datum, true);
                    return (lab.actualLabel != ground_label);
                });
            }




            public static double GetAccuracy(NeuralNet nn, Dataset ds)
            {
                int cnt = 0;
                int prg = 0;
                // Parallel.For(0, ds.Count(), RobustnessOptions.ParallelOptions, i =>
                for (int i = 0; i < ds.Count(); i++)
                {


                    //if (i < 10000) { continue;  }

                    double[] datum;
                    int ground_label;

                    datum = ds.GetDatum(i);
                    ground_label = ds.GetLabel(i);


                    var labconf = Utils.ULabel.LabelWithConfidence(nn,ds.GetDatum(i),true);
                    int label = labconf.actualLabel;
                    // int label = Utils.ULabel.Label(nn, ds.GetDatum(i), true);

                    lock (lockObject)
                    {
                        prg++;
                       // Console.WriteLine("Confidence = {0}", labconf.softMaxValue);
                    }

                    if (label == ground_label)
                    {
                        lock (lockObject)
                        {
                            cnt++;
                        }
                    }
                    else
                    {
                        // Console.WriteLine("Missclassifciation: " + label + " vs " + testImages.Dataset.GetLabel(i));
                    }
                    lock (lockObject)
                    {
                        Console.Write("\r{0:0.000}%, Accuracy:{1:0.000}%", (double)prg * 100.0 / ds.Count(), (double)cnt * 100.0 / prg);

                        //int[] foobar = Utils.UArray.ToRGBArray(datum, 1.0, 0.0);
                        //Utils.UDraw.DisplayImageAndPause(foobar, 32, 32, true);

                    }

                };

                Console.WriteLine("\nCorrectly classified = {0}", cnt);
                Console.WriteLine("Total images         = {0}", ds.Count());
                double acc = (double)cnt / ds.Count();
                Console.Write("\nAccuracy: ");
                Console.WriteLine(acc);
                Console.WriteLine("ReLU Collisions = {0}", Instrumentation.Collisions);
                return acc;
            }

            public static double GetLoss(NeuralNet nn, Dataset ds)
            {
                double loss = 0.0;
                int prg = 0;
                Parallel.For(0, ds.Count(), RobustnessOptions.ParallelOptions, i =>
                //                for (int i =0; i < ds.Count(); i++)
                {
                    double[] softmax = Utils.ULabel.RunWithSoftmax(nn, ds.GetDatum(i), true);
                    lock (lockObject)
                    {
                        prg++;
                        int lab = ds.GetLabel(i);

                        // safety for infinity ... 
                        if (softmax[lab] == 0.0)
                        {
                            softmax[lab] += 1e-10;
                        }
                        loss += Math.Log(softmax[lab]);
                        Console.Write("\r{0:0.000}%", (double)prg * 100.0 / ds.Count());
                    }

                });

                Console.Write("\nTotal loss: ");
                Console.WriteLine(loss);

                return loss;
            }


        }
    }
}
