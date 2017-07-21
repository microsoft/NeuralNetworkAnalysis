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
using System.Collections;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

using System.Diagnostics;


namespace NNAnalysis
{
    namespace Utils
    {

        public static class RobustnessOptions
        {
            // Filename to register the counterexample and information about it
            public static string Registry = "generic-registy";
 
            // What percentage of the dataset should we iterate over?
            public static double DataSetPercentage = 1.0;
            // Do the CEGAR loop or not?
            public static bool CEGAR = true;
            // Generate examples with optimization or just by bounding?
            public static bool DoOptimization = true;
            // When bounding for counterexample generation, how far should we look?
            public static double Epsilon = 20.0; // 48.0; // 14.69534; // 50.0;

            // Minimum range of each entry?
            public static double MinValue = 0.0;
            public static double MaxValue = 255.0;

            // The final label of the image should be not just bigger than others but a lot bigger. Default is 0.0.
            public static double LabelConfidenceDiff = 0.0f;
            // Objective kind
            public static LPSObjectiveKind ObjectiveKind = LPSObjectiveKind.MinLinf;
            // Timeout to solver
            public static long LPTimeMilliSeconds = 8 * 60 * 1000;
            // Parallelism
            public static ParallelOptions ParallelOptions = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };
            // If confidence of counterexample is low, then just ignore it 
            public static bool IgnoreLowConfidence = false;
            public static double LowConfidenceThreshold = 0.55;

            // Don't go more than CEGARGiveUpIterations times around CEGAR
            public static double CEGARGiveUpIterations = 4;

            // Slack (as a percentage of the interecept) for linear constraint
            // strict inequalities. Oh well if the interecept is 0.0 then we will 
            // get 0.0 but it does not matter too much. In general I am not really
            // certain about how important this StrictInequalityLambda is.
            public static double StrictInequalityLambda      = 0.00001;

            // Sample a random percentage of the live constraints, defer the rest
            public static double LiveConstraintSamplingRatio = 0.1;

            /* The formula that we generate may or may not guarantee that
             * the synthesized image is different than the original by at least the 
             * quantization noise. Setting this flag to true ensures that we round
             * the image and check that it is indeed a counterexample. For images,
             * each pixel is a byte representing an integer 0-255 so we just use integer
             * rounding (by the generated bounding formula we guarantee that it's in [0,255]
             * *****************************************************************************/
            public static bool QuantizationSafety = true;
            
            // Pause and display synthesized images in the main loop?
            public static bool DisplaySynthesizedImagesAndPause = false;

            // Use rationals (default) or integers for variables in the LP
            public static bool Integrality = false; 


            // For pre-processed image data, if we are to dump them in a file we need to scale
            // them appropriately: new = old * scale + offset
            public static double ScalePreProcessed = 1.0;
            public static double OffsetPreProcessed = 0.0;

            // save PNG files? (default off) 
            public static bool SavePNGCounterexamples = false;


            public static string ReLULogFile = "";

            public static void Dump()
            {
                Console.WriteLine("Robustness options:");
                Console.WriteLine("Registry:                    {0}", Registry);
                Console.WriteLine("DatasetPercentage:           {0}", DataSetPercentage);
                Console.WriteLine("CEGAR:                       {0}", CEGAR);
                Console.WriteLine("DoOptimization:              {0}", DoOptimization);
                Console.WriteLine("Epsilon (bound):             {0}", Epsilon);
                Console.WriteLine("LabelConfidenceDiff:         {0}", LabelConfidenceDiff);
                Console.WriteLine("LPTimeMilliSeconds:          {0}", LPTimeMilliSeconds);
                Console.WriteLine("IgnoreLowConfidence:         {0}", IgnoreLowConfidence);
                Console.WriteLine("LowConfidenceThreshold:      {0}", LowConfidenceThreshold);
                Console.WriteLine("CEGARGiveUpIterations:       {0}", CEGARGiveUpIterations);
                Console.WriteLine("LiveConstraintSamplingRatio: {0}", LiveConstraintSamplingRatio);
                Console.WriteLine("QuantizationSafety:          {0}", QuantizationSafety);
                Console.WriteLine("Integrality:                 {0}", Integrality);
                Console.WriteLine("MinValue:                    {0}", MinValue);
                Console.WriteLine("MaxValue:                    {0}", MaxValue);
                Console.WriteLine("ScalePreProcessed:           {0}", ScalePreProcessed);
                Console.WriteLine("OffsetPreProcessed:          {0}", OffsetPreProcessed);
                Console.WriteLine("SavePNGCounterexamples:      {0}", SavePNGCounterexamples);
            }

        }


        public class DiffInfo
        {
            public Vector<double> diff;
            public int number;
        }



        public class Robustness
        {
            private static Object lockObj = new Object();


            // A dictionary such that for each class we have a difference compared to
            // a class that we have found a counterexample to! 
            private static Dictionary<Tuple<int,int>,DiffInfo> diffDict = new Dictionary<Tuple<int,int>,DiffInfo>();


            public static bool Satisfiable(LPSConstraint ct, Vector<double> image_plus_eps)
            {

                // Native inner product more efficient
                double lhs = image_plus_eps * ct.Term.GetCoefficients(); // ct.Term.GetCoefficients().SubVector(0, image.Count);
                double rhs = 0.0;
                bool sat = false;

                rhs = -ct.Term.Intercept;

                switch (ct.Inequality)
                {
                    case InequalityType.EQ:
                        sat = (lhs == rhs);
                        break;
                    case InequalityType.GE:
                        sat = (lhs >= rhs);
                        break;
                    case InequalityType.GT:
                        sat = (lhs > rhs);
                        break;
                    case InequalityType.LE:
                        sat = (lhs <= rhs);
                        break;
                    case InequalityType.LT:
                        sat = (lhs < rhs);
                        break;
                }

                return sat;
            }


            /// <summary>
            /// Generate symbolic inputs and (potentially) a term for the epsilon of the objective.
            /// </summary>
            public static Tuple<LPSTerm[],LPSTerm> GenSymbolicInputs(int inputDimension)
            {
                LPSTerm[] inputs = null;
                LPSTerm epsilon = null;

                LPSTerm.ResetVariableFactory(inputDimension + 1);
                var all = LPSTerm.FreshVariables(inputDimension+1);
                epsilon = all[inputDimension];
                inputs = new LPSTerm[inputDimension];
                Array.Copy(all, inputs, inputDimension);

                return new Tuple<LPSTerm[], LPSTerm>(inputs, epsilon);
            }


            public static ImageDataset SynthesizeCounterexamplesAndStore(NeuralNet nn, ImageDataset ds, Action<LabelWithConfidence> snapshot)
            {
                List<double[]> data = new List<double[]>();
                List<int> labs = new List<int>();

                List<LabelWithConfidence> results = SynthesizeCounterexamples(nn, ds, snapshot);

                for (int i=0; i < results.Count(); i++)
                {
                    data.Add(results[i].datum);
                    labs.Add(results[i].actualLabel);
                }

                Dataset newdata = new Dataset(data,labs,ds.Dataset.LabelCount());
                return new ImageDataset(newdata, ds.ChannelCount, ds.RowCount, ds.ColumnCount, ds.IsColor);

            }



            /// <summary>
            /// Generate and return a list of counterexamples by iterating over the training set
            /// </summary>
            /// <param name="datasetname"></param>
            /// <param name="options"></param>
            /// <param name="nn"></param>
            /// <param name="ds"></param>
            /// <returns></returns>
            public static List<LabelWithConfidence> SynthesizeCounterexamples(
                NeuralNet nn,
                ImageDataset ds, 
                Action<LabelWithConfidence> snapshot)
            {

                /* Initialization stuff */
                List<LabelWithConfidence> counterexamples = new List<LabelWithConfidence>();
                SynthRegistry reg = new SynthRegistry(RobustnessOptions.Registry + ".csv", RobustnessOptions.Registry);

                /* How many training points to do */
                int trainingPointsToDo = (int)Math.Round(ds.Dataset.Count() * RobustnessOptions.DataSetPercentage);
                int completed = 0;

                /* The symbolic variables: NB we use the dimension PostCrop to avoid generating lots of useless variables */
                Tuple<LPSTerm[],LPSTerm> inputs = GenSymbolicInputs(nn.InputDimensionPostCrop);

                // Alternatively (the code is thread-safe already):
                // Parallel.For(0, ds.Dataset.Count(), RobustnessOptions.ParallelOptions, i =>
                for (int i = 0; i < ds.Dataset.Count(); i++)
                {
                        if (completed < trainingPointsToDo)
                        {
                            Console.WriteLine("Image count = {0}", i);
                            NNInstrumentation instr = new NNInstrumentation();
                            LabelWithConfidence imageLab = ULabel.LabelWithConfidence(nn, instr, ds.Dataset.GetDatum(i), true);

                            Nullable<LabelWithConfidence> synthLab = null;

                            try
                            {
                                var stopwatch = new Stopwatch();
                                stopwatch.Start();

                                synthLab = SynthesizeCounterexample(
                                    nn,
                                    inputs.Item1,
                                    inputs.Item2,
                                    imageLab,
                                    instr,
                                    ds.Dataset.GetLabel(i),
                                    ds.RowCount,
                                    ds.ColumnCount,
                                    ds.IsColor);

                                stopwatch.Stop();
                                Console.WriteLine("Processed image in {0} milliseconds", stopwatch.ElapsedMilliseconds);
                                GC.Collect();

                            }
                            catch
                            {
                                lock(lockObj)
                                {
                                    completed++;
                                }
                                continue;
                            }


                            lock (lockObj)
                            {
                                completed++;
                                if (synthLab.HasValue)
                                {
                                    // VERY IMPORTANTLY: Change the label of the counterexample
                                    // to be the label of the original point! This was a horrible bug.
                                    var forRetraining = synthLab.Value;
                                    forRetraining.actualLabel = imageLab.actualLabel;
                                    counterexamples.Add(forRetraining);
                                    snapshot(forRetraining);

                                    Console.WriteLine("forRetraining.label  = {0}", forRetraining.actualLabel);
                                    Console.WriteLine("synthLab.Value.label = {0}", synthLab.Value.actualLabel);

                                    reg.RecordAtomically(
                                        RobustnessOptions.Registry,
                                        i,
                                        imageLab,
                                        synthLab.Value,
                                        RobustnessOptions.ScalePreProcessed,
                                        RobustnessOptions.OffsetPreProcessed,
                                        ds.RowCount,
                                        ds.ColumnCount,
                                        ds.IsColor);
                                }
                                Console.WriteLine("Counterexamples/Processed-so-far: " + counterexamples.Count() + "/" + completed);
                            }
                        }
                }
                return counterexamples;
            }



            /// <summary>
            /// Synthesize a counterexample from an existing labelled image.
            /// </summary>
            /// <param name="options"></param>
            /// <param name="nn">The model.</param>
            /// <param name="imageLab">The image and labeling information from the network.</param>
            /// <param name="instr"></param>
            /// <param name="realLabel">The label of the image from the training set.</param>
            /// <param name="rowSize"></param>
            /// <param name="colSize"></param>
            /// <param name="isColor"></param>
            /// <returns>NULL if we were not able to synthesize a counterexample, otherwise some information about it.</returns>
            public static Nullable<LabelWithConfidence> SynthesizeCounterexample
                ( NeuralNet nn
                , LPSTerm[] inputs             // Symbolic inputs (cropped)
                , LPSTerm epsilon              // Epsilon variable
                , LabelWithConfidence imageLab // Original image classification info (uncropped)
                , NNInstrumentation instr
                , int realLabel                // Ground truth for this image (from training set)
                , int rowSize                  // Original (uncropped) row size
                , int colSize                  // Original (uncropped) col size
                , bool isColor)
            {
                int origLabel = imageLab.actualLabel;
                int targetLabel = imageLab.secBestLabel;

                int input_dimension_pre_crop  = nn.InputDimensionPreCrop;
                int input_dimension_post_crop = nn.InputDimensionPostCrop;


                double[] orig_image = imageLab.datum;
                double[] orig_image_crop = nn.CropMaybe(DenseVector.OfArray(orig_image)).ToArray();


                if (realLabel != origLabel)
                {
                    Console.WriteLine("This image is misclassifed already! Skipping.");
                    return null;
                }
                if (RobustnessOptions.IgnoreLowConfidence && imageLab.softMaxValue < RobustnessOptions.LowConfidenceThreshold)
                {
                    Console.WriteLine("This image is misclassifed with low confidence! Skipping.");
                    return null;
                }

                // Fast path:
                // DiffInfo diff_info;
                /* *********************
                 * DV: Commenting out the fast path for now (but we are still keeping the Dictionary, for debugging)
                 * *********************
                if (diffDict.TryGetValue(new Tuple<int,int>(origLabel,targetLabel),out diff_info))
                {
                    Console.WriteLine("Got a hit in the difference cache!");
                    Vector<double> diff_counterexample = diff_info.diff;

                    Vector<double> cand = DenseVector.OfArray(orig_image) + diff_counterexample;


                    Console.WriteLine("oooooooooooooooo Checking with the fast path!");

                    double[] cand_arr_crop = nn.CropMaybe(cand).ToArray();

                    if (RobustnessOptions.QuantizationSafety)
                    {
                        Utils.UArray.InPlaceRoundDoubleArray(cand_arr_crop);
                    }

                    LabelWithConfidence candLab = Utils.ULabel.LabelWithConfidence(nn, cand_arr_crop,false); // Already  cropped, don't crop!

                    if (candLab.actualLabel != origLabel)
                    {

                        Console.WriteLine("=> Real counterexample (from fast path)!");
                        diff_info.number++;
                        return candLab;
                    }

                    Console.WriteLine("xxxx Fast path failed, continuing with symbolic interpreter ...");
                    // otherwise continue with the slow path ...
                }
                ***********************/

                var state = new LPSState(instr, orig_image_crop);
                int nomodelcount = 0;

                double[] newImageUnrounded;

            NOMODELLOOP:
                if (nomodelcount++ > 0) return null;

                state.ClearConstraints();

                LPSTerm[] output = nn.EvaluateNNSymbolicPostCrop(state, inputs);

                // Just some tracing ...
                // ReportSparsity(output);

                LPSConstraints currentCts = state.CurrentCts;
                LPSConstraints deferredCts = state.DeferredCts;

                // Conjoin the label formula
                currentCts.And(NNetFormulas.LabelFormula(output, targetLabel, RobustnessOptions.LabelConfidenceDiff));

                // If we are just looking for bounds, then the variables themselves will contain "origin" bounds
                if (RobustnessOptions.DoOptimization)
                {
                    NNETObjectives.AddEpsilonBounds(currentCts, inputs, epsilon, orig_image_crop);
                }

                // Ensure that at least *one* entry is different by at least 1.0
                if (RobustnessOptions.QuantizationSafety)
                {
                    NNETObjectives.AddQuantizationSafety(currentCts, inputs, orig_image_crop);
                }

                // Create objective
                Nullable<LPSObjective> objective = null;
                if (RobustnessOptions.DoOptimization)
                {
                    switch (RobustnessOptions.ObjectiveKind)
                    {
                        case LPSObjectiveKind.MinLinf:
                            objective = NNETObjectives.MinLInf(currentCts, inputs, epsilon, orig_image_crop);
                            break;
                        case LPSObjectiveKind.MaxConf:
                            objective = NNETObjectives.MaxConf(output, origLabel, targetLabel);
                            break;
                        default:
                            break;
                    }
                }
                if (!RobustnessOptions.CEGAR)
                {
                    currentCts.And(deferredCts);
                    deferredCts = new LPSConstraints();
                }

                // CEGAR loop header
                LabelWithConfidence newLab;

                Console.WriteLine(
                    "Current constraints: {0}, deferred: {1}", 
                    currentCts.Count, 
                    deferredCts.Count);

                LPSolver lps = new LPSolver(
                    input_dimension_post_crop, 
                    currentCts.Count + deferredCts.Count, 
                    orig_image_crop, 
                    RobustnessOptions.Epsilon);

                lps.AddConstraints(currentCts, objective);

                int cegar_iterations = 0;

                while (true)
                {
                    if (cegar_iterations++ > RobustnessOptions.CEGARGiveUpIterations)
                    {
                        Console.WriteLine("xxxxxxxxxxxxxxxx Giving up CEGAR, could not find model!");
                        goto NOMODELLOOP;
                    }

                    var newImage = lps.SolveLowLevelLP();

                    currentCts = new LPSConstraints();
                    if (newImage == null)
                    {
                        Console.WriteLine("xxxxxxxxxxxxxxxx No model!");
                        goto NOMODELLOOP;
                    }

                    Console.WriteLine("oooooooooooooooo Found model!");
                    newImageUnrounded = new double[newImage.Length];
                    Array.Copy(newImage, newImageUnrounded, newImage.Length);

                    if (RobustnessOptions.QuantizationSafety)
                    {
                        Utils.UArray.InPlaceRoundDoubleArray(newImage);
                    }
                    int samcount = Utils.UArray.ComputeRoundIdenticals(orig_image_crop, newImage);
                    Console.WriteLine("Synthesized image has {0} identical inputs (after rounding) to original (cropped)", samcount);
                    // Now, try to label the new example
                    newLab = Utils.ULabel.LabelWithConfidence(nn, newImage,false); // Already  cropped, don't crop!
                    if (newLab.actualLabel != targetLabel)
                    {
                        if (newLab.actualLabel == realLabel)
                        {
                            // Here the synthesized image is not really a counterexample. 
                            // This could be due to either (a) quantization errors or (b) CEGAR 
                            // underapproximation. But the only thing we can try and do here is
                            // add mor constraints and try to resolve. 

                            if (RobustnessOptions.CEGAR)
                                Console.WriteLine("Not really a counterexample, going round CEGAR loop.");

                            int added = 0;
                            
                            // new_image_plus_eps = newImage : 0.0 
                            // so that the length matches the coefficients of each constraint ... 
                            double[] newimage_plus_eps = new double[newImage.Length+1];
                            Array.Copy(newImageUnrounded,newimage_plus_eps,newImage.Length);
                            newimage_plus_eps[newImage.Length] = 0.0;
                            Vector<double> newImageVec_eps = DenseVector.OfArray(newimage_plus_eps);

                            var denumerator = deferredCts.GetEnumerator();

                            Parallel.For(0, deferredCts.Count, i =>
                            {
                                LPSConstraint curr_deferred;
                                if (added > 699) return;

                                lock (lockObj)
                                {

                                    denumerator.MoveNext();
                                    curr_deferred = (LPSConstraint)denumerator.Current;

                                    if (curr_deferred.Added == true) return;

                                }

                                bool sat = Satisfiable(curr_deferred, newImageVec_eps);

                                lock (lockObj)
                                {
                                    if (!sat)
                                    {
                                        lps.AddConstraint(curr_deferred);
                                        // currentCts.And(curr_deferred.Term, curr_deferred.Inequality);
                                        curr_deferred.Added = true;
                                        added++;
                                    }
                                }
                            });



                            Console.WriteLine();
                            Console.WriteLine("Added {0} constraints for CEGAR", added);
                            if (added == 0)
                            {
                                Console.WriteLine("=> CEGAR cannot improve things.");
                                goto NOMODELLOOP;
                                // return null;
                            }
                            // lps.AddConstraints(currentCts, null);
                            continue;
                        }
                        else
                        {
                            Console.WriteLine("=> Real counterexample! (Although with different label than expected)");
                            break;
                        }
                    }
                    else
                    {
                        Console.WriteLine("=> Real counterexample! (New image has second-best label");
                        break;
                    }
                }

                if (RobustnessOptions.DisplaySynthesizedImagesAndPause)
                {
                    Utils.UDraw.DisplayImageAndPause(Utils.UArray.ToIntArray(imageLab.datum), rowSize, colSize, isColor);
                    Utils.UDraw.DisplayImageAndPause(Utils.UArray.ToIntArray(newLab.datum), rowSize, colSize, isColor);
                }

                /* NB: Uncrop the image in newLab */
                newLab.datum = nn.UnCropMaybe(DenseVector.OfArray(orig_image), DenseVector.OfArray(newLab.datum)).ToArray();


                double[] tmp = nn.UnCropMaybe(DenseVector.OfArray(orig_image), DenseVector.OfArray(newImageUnrounded)).ToArray();
                Vector<double> diff_val = DenseVector.OfArray(tmp) - DenseVector.OfArray(orig_image);

                var key = new Tuple<int, int>(origLabel, newLab.actualLabel);
                DiffInfo dinfo;
                if (diffDict.TryGetValue(key, out dinfo))
                {
                    dinfo.number++;
                }
                else
                {
                    dinfo = new DiffInfo();
                    dinfo.diff = diff_val;
                    dinfo.number = 1;
                    diffDict.Add(new Tuple<int, int>(origLabel, newLab.actualLabel), dinfo);
                }

                return newLab;
            }

            /// <summary>
            /// Given the symbolic output we check which columns are completely zero,
            /// which effectively implies that the corresponding variables do not participate
            /// in the Jacobian.
            /// </summary>
            /// <param name="output"></param>
            public static void ReportSparsity(LPSTerm[] output)
            {
                var matrix = LPSTerm.UnderlyingMatrix(output);
                Vector<double> zeros = DenseVector.Create(output.Length,0.0);

                // int sparse_count = 0;

                List<Tuple<int, double>> stats = new List<Tuple<int, double>>();

                for (int i=0; i < matrix.ColumnCount; i++)
                {
                    var col = matrix.Column(i);

                    stats.Add(new Tuple<int, double>(i, col.Maximum()));

                }

                stats.Sort(delegate(Tuple<int,double> t1, Tuple<int,double> t2)
                {
                    return t1.Item2.CompareTo(t2.Item2);
                });

                foreach (var s in stats)
                {
                    Console.WriteLine(s.Item2);
                }

            }
        }
    }
}
