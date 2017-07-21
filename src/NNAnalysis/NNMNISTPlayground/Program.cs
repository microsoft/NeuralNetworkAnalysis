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
using System.Drawing;
using System.Drawing.Imaging;
using System.Drawing.Printing;
using System.IO;


namespace NNMNISTPlayground
{
    class Program
    {
        static void Main(string[] args)
        {
            string net = args[0];
            string pn1 = args[1];
            string pn2 = args[2];

            int equal = 0;
            int unequal = 0;

            Options.InitializeNNAnalysis();

            NeuralNet nn = MNIST.GetNN(net);

            Console.WriteLine("Initialized network");


            Bitmap x1 = new Bitmap(pn1);
            Bitmap x2 = new Bitmap(pn2);

            int[] dat1 = UDraw.FromBitmap(x1, MNIST.InputCoordinates.RowCount, MNIST.InputCoordinates.ColumnCount, false);
            int[] dat2 = UDraw.FromBitmap(x2, MNIST.InputCoordinates.RowCount, MNIST.InputCoordinates.ColumnCount, false);

            int lab1 = NNAnalysis.Utils.ULabel.Label(nn, UArray.ToDoubleArray(dat1), true);

            int lab2 = NNAnalysis.Utils.ULabel.Label(nn, UArray.ToDoubleArray(dat2), true);

            if (lab1 == lab2)
            {
                equal++;
            }
            else
            {
                unequal++;
            }

            Console.Write("Label for {0} is: {1}-{2}", pn1, lab1, lab2);
            Console.WriteLine(", equals = {0}", (lab1 == lab2));
            //Console.WriteLine("Label for {0} is: {1}", pn2, lab2);
        }
    }
}
