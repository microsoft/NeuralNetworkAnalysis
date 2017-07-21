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
using System.IO;
using System.Drawing;

using NNAnalysis.Utils;

namespace NNAnalysis
{
    public struct SynthEntry
    {
        public String datasetName;      // Data set name
        public int datasetIndex;        // Position in the dataset
        public String origPngName;      
        public int origLabel;          
        public String synthPngName;
        public int synthLabel;
        public double lInfDist;
        public double l1Dist;
        public double confOrig;
        public double confSynth;
        public double sndBestDiffOrig;
        public double sndBestDiffSynth;
    }


    public class SynthRegistry
    {
        Object inMemLock_ = new Object();
        Object fileLoc_ = new Object();

        String csvFileName_;
        StreamWriter registryCsvFileWriter_;
        String dataDirectory_;

        public void SerializeEntry(SynthEntry e)
        {
            lock (fileLoc_)
            {
                registryCsvFileWriter_.WriteLine(
                    e.datasetName + "," +
                    e.origPngName + "," +
                    e.origLabel + "," +
                    e.synthPngName + "," +
                    e.synthLabel + "," +
                    e.lInfDist + "," +
                    e.l1Dist + "," +
                    e.confOrig + "," +
                    e.sndBestDiffOrig + "," +
                    e.confSynth + "," +
                    e.sndBestDiffSynth);

                registryCsvFileWriter_.Flush();
            }
        }

        public SynthRegistry(String csvFileName, String dataDirectory)
        {
            csvFileName_ = csvFileName;
            registryCsvFileWriter_ = new StreamWriter(csvFileName);
            dataDirectory_ = dataDirectory;

            bool exists = System.IO.Directory.Exists(dataDirectory);
            if (!exists) System.IO.Directory.CreateDirectory(dataDirectory);

            if (registryCsvFileWriter_ == null)
            {
                throw new Exception("Can't open counterexample registry file!");
            }

        }


        public String CreatePnG(
            String origin, 
            String datasetname,
            int datasetIndex, 
            double[] input, 
            double scale, double offset,
            int numRows, int numCols, bool isColor, bool isRowOrder = true)
        {
            String path = Path.Combine(dataDirectory_, origin + "-" + datasetname + "-" + datasetIndex + ".png");

            if (RobustnessOptions.SavePNGCounterexamples)
            {
                // Create PnG file for orig
                int[] imagePixels = Utils.UArray.ToRGBArray(input, scale, offset);
                Bitmap image = isColor ?
                    Utils.UDraw.DrawRGBPixels(imagePixels, numRows, numCols, isRowOrder) :
                    Utils.UDraw.DrawGrayscalePixels(imagePixels, numRows, numCols, isRowOrder);
                image.Save(path);
            }
            return path;
        }

        public void RecordAtomically(
            String datasetName, 
            int datasetIndex,
            LabelWithConfidence origLab,
            LabelWithConfidence synthLab,
            double scale, double offset,
            int numRows, int numCols, bool isColor, bool isRowOrder = true)
        {
            SynthEntry e = new SynthEntry();
            e.datasetName = datasetName;

            e.origPngName = CreatePnG("orig", datasetName, datasetIndex, origLab.datum, scale, offset, numRows, numCols, isColor, isRowOrder);
            e.origLabel = origLab.actualLabel;

            e.synthPngName = CreatePnG("snth", datasetName, datasetIndex, synthLab.datum, scale, offset, numRows, numCols, isColor, isRowOrder);
            e.synthLabel = synthLab.actualLabel;

            double[] diff = new double[origLab.datum.Length];
            for (int i = 0; i < origLab.datum.Length; i++ )
            {
                diff[i] = 5 * (origLab.datum[i] - synthLab.datum[i]) + 100; // -20 ... 20 -> -100 .. 100 -> 0 .. 200
            }

            CreatePnG("diffx5o100", datasetName, datasetIndex,diff,scale,offset,numRows,numCols,isColor,isRowOrder);

            Console.WriteLine("Orig path  =" + e.origPngName);
            Console.WriteLine("Synth path =" + e.synthPngName);


            e.lInfDist = Utils.UMath.LInfinityDistance(origLab.datum, synthLab.datum);
            e.l1Dist = Utils.UMath.L1Distance(origLab.datum, synthLab.datum);

            e.confOrig = origLab.softMaxValue;
            e.confSynth = synthLab.softMaxValue;

            e.sndBestDiffOrig = origLab.diffFromSecondBest;
            e.sndBestDiffSynth = synthLab.diffFromSecondBest;

            SerializeEntry(e);

        }

    }

}
