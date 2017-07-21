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


    /// <summary>
    /// Class used to convert between an index in a 1D array to the channel/row/column of the image the array represents
    /// 
    /// And the math:
    /// (x, y, z) -> a = (x * Ny * Nz + y * Nz + z)
    /// x = a % (Nx * Ny * Nz) / (Ny * Nz) = a / (Ny * Nz)
    /// y = a % (Ny * Nz) / Nz
    /// z = a % (Nz) / 1 = a % Nz
    /// </summary>
    public class ImageCoordinates
    {
        int channelCount_;   // e.g. The input ImageCoordinates for MNIST (grayscale) = 1, CIFAR (RGB) = 3
        int rowCount_;
        int columnCount_;
        public ImageCoordinates(int channelCount, int rowCount, int columnCount)
        {
            channelCount_ = channelCount;
            rowCount_ = rowCount;
            columnCount_ = columnCount;
        }
        public int GetImageChannel(int index)
        {
            return index / (rowCount_ * columnCount_);
        }
        public int GetImageRow(int index)
        {
            return (index % (rowCount_ * columnCount_)) / columnCount_;
        }
        public int GetImageColumn(int index)
        {
            return index % columnCount_;
        }
        public int GetIndex(int channel, int row, int column)
        {
            return rowCount_ * columnCount_ * channel + columnCount_ * row + column;
        }
        public int ChannelCount
        {
            get { return channelCount_; }
        }
        public int RowCount
        {
            get { return rowCount_; }
        }
        public int ColumnCount
        {
            get { return columnCount_; }
        }
    }

    namespace Utils
    {
        public static class UImageCoordinate
        {
            // TODO: Cross-check logic
            public static int ComputeOutputCounts(int kernelDimension, int imageDimension, int stride, int padding, bool padEnding)
            {
                double outputCountFloat = ((double)(imageDimension + 2 * padding - kernelDimension)) / stride + 1;
                int outputCount = (int)(padEnding ? Math.Ceiling(outputCountFloat) : Math.Floor(outputCountFloat));
                // Remove last kernel application if it starts in the padding
                return outputCount + (((outputCount - 1) * stride >= imageDimension + padding) ? -1 : 0);
            }
        }
    }

}
