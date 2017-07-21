package protobuf;

import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import caffe.Caffe;
import caffe.Caffe.BlobProto;
import caffe.Caffe.ConvolutionParameter;
import caffe.Caffe.LayerParameter;
import caffe.Caffe.NetParameter;
import caffe.Caffe.PoolingParameter;

public class Main {	
	public static void writeNN(DataOutputStream out, NetParameter nn) throws IOException {
		out.writeInt(nn.getLayerCount());
		System.out.println("Writing layers: " + nn.getLayerCount());
		for(LayerParameter layer : nn.getLayerList()) {
			writeLayer(out, layer);
		}
	}
	
	public static void writeLayer(DataOutputStream out, LayerParameter layer) throws IOException {
		String type = layer.getType();
		System.out.println("Processing layer " + layer.getName() + " of type " + type);
		if(type.equals("Data")) { // 0
			out.writeInt(0);
			writeDataLayer(out, layer);
		} else if(type.equals("InnerProduct")) { // 1
			out.writeInt(1);
			writeInnerProductLayer(out, layer);
		} else if(type.equals("ReLU")) { // 2
			out.writeInt(2);
			writeReLULayer(out, layer);
		} else if(type.equals("SoftmaxWithLoss")) { // 3
			out.writeInt(3);
			writeSoftmaxLayer(out, layer);
		} else if(type.equals("Convolution")) { // 4
			out.writeInt(4);
			writeConvolutionLayer(out, layer);
		} else if(type.equals("Pooling")) { // 5
			out.writeInt(5);
			writePoolingLayer(out, layer);
		} else if(type.equals("Dropout")) { // 6
			out.writeInt(6);
			// Dropout seems to be the identity layer in Caffe!
		} else {
			throw new RuntimeException("Unknown layer type: " + type);
		}
	}
	
	public static void writeDataLayer(DataOutputStream out, LayerParameter layer) throws IOException {

	    boolean has_transform_param = layer.hasTransformParam();
	    out.writeInt(has_transform_param? 1:0);
	    if (!has_transform_param) return;

	    Caffe.TransformationParameter param = layer.getTransformParam();
	    // Scale
	    boolean has_scale = param.hasScale();
	    out.writeInt(has_scale? 1 : 0);
	    if (has_scale) { out.writeFloat(param.getScale()); }
	    // Mirror
	    boolean has_mirror = param.hasMirror();
	    out.writeInt(has_mirror? 1:0);
	    // Cropsize
	    boolean has_crop_size = param.hasCropSize();
	    out.writeInt(has_crop_size? 1 : 0);
	    if (has_crop_size) { out.writeInt(param.getCropSize()); }
	    // MeanValue
	    int mean_val_cnt = param.getMeanValueCount();
	    // If > 0 then it can either be a one-element list, 
	    // to be subtracted from every channel,
	    // or a #channel-element-sized list.
	    out.writeInt(mean_val_cnt);
	    for (int i=0; i < param.getMeanValueCount(); i++)
	    {
		out.writeFloat(param.getMeanValue(i));
	    }
	    // Mean File
	    boolean has_mean_file = param.hasMeanFile();
	    out.writeInt(has_mean_file? 1: 0);
	    if (has_mean_file) {
		String mean_file = param.getMeanFile();
		InputStream input = new FileInputStream(mean_file);
		Caffe.BlobProto blob = Caffe.BlobProto.parseFrom(input);
		int count = blob.getDataCount();
		out.writeInt(count);
		for (int i=0; i < count; i++) {
		    out.writeFloat(blob.getData(i));
		}		
	    }
	    // Implicit invariant: either mean-file is present or mean-value(s) but not both...
	}
	
	public static void writeInnerProductLayer(DataOutputStream out, LayerParameter layer) throws IOException {
		if(layer.getBlobsCount() != 2) {
			throw new RuntimeException("Unexpected number of blobs for inner product layer: " + layer.getBlobsCount());
		}
		// Caffe format:
		// data : X * Y is a matrix with X rows, Y columns, data[i][j] is row i column j, array format is data[X][Y]
		// x = input : 1 * K
		// A = blob 0 : N * K
		// B = blob 1 : 1 * N
		// y = output : N
		// Computes: y = x * A' + B
		// Computes: y[i] = \sum_j A[i][j] * x[j] + B[i]
		
		// Compute K and N
		int kn = layer.getBlobs(0).getDataCount();
		int n = layer.getBlobs(1).getDataCount();
		int k = kn/n;
		if(k*n != kn) {
			throw new RuntimeException("Invalid layer dimensions: " + kn + " != " + k + "*" + n);
		}
		System.out.println("Input dimension: " + k);
		System.out.println("Output dimension: " + n);
		// Write K
		out.writeInt(k);
		// write N
		out.writeInt(n);
		// Write A[i][j] and B[j]
		for(BlobProto blob : layer.getBlobsList()) {
			for(float value : blob.getDataList()) {
				out.writeFloat(value);
			}
		}
	}
	
	public static void writeReLULayer(DataOutputStream out, LayerParameter layer) {}
	
	public static void writeSoftmaxLayer(DataOutputStream out, LayerParameter layer) {}
	
	public static void writeConvolutionLayer(DataOutputStream out, LayerParameter layer) throws IOException {
		ConvolutionParameter param = layer.getConvolutionParam();
		int numOutput = param.getNumOutput();
		int kernalSize = param.getKernelSize();
		int padding = param.getPad();
		out.writeInt(numOutput);
		out.writeInt(kernalSize);
		out.writeInt(padding);

		int group = param.getGroup();
		// Safeguarding against funkier strides and groups that we don't support well.
		if (group != 1) {
		    throw new RuntimeException("Not-supported: Convolution layer has group != 1");
		}
		int stride = param.getStride();
		if (stride != 1) {
		    throw new RuntimeException("Not-supported: Convolution layer has stride != 1");
		}
		
		// blobs[0]: numOutput * channels * kernalSize (columns) * kernalSize (rows)
		// blobs[1]: numOutput
		for(BlobProto blob : layer.getBlobsList()) {
			out.writeInt(blob.getDataCount());
			for(float value : blob.getDataList()) {
				out.writeFloat(value);
			}
		}
	}
	
	public static void writePoolingLayer(DataOutputStream out, LayerParameter layer) throws IOException {
		PoolingParameter param = layer.getPoolingParam();
		int kernelSize = param.getKernelSize();
		int stride = param.getStride();
		int padding = param.getPad();
		out.writeInt(kernelSize);
		out.writeInt(stride);
		out.writeInt(padding);
		out.writeInt(param.getPool() == PoolingParameter.PoolMethod.MAX ? 0 : 1);

	}
	
	public static void writeCaffeModel(InputStream input, OutputStream output) throws IOException {
		writeNN(new DataOutputStream(output), Caffe.NetParameter.parseFrom(input));
	}
	
	public static void writeCaffeModel(String inputName, String outputName) throws IOException {
		InputStream input = new FileInputStream(inputName);
		OutputStream output = new FileOutputStream(outputName);
		writeCaffeModel(input, output);
		input.close();
		output.close();
	}
	
	public static void main(String[] args) throws IOException {
            if (args.length < 2) {
                System.out.println("./Main <input file> <output file>");
            }
            else {
                writeCaffeModel(args[0], args[1]);
            }
	}
}
