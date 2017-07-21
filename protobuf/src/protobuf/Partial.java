package protobuf;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import caffe.Caffe.NetParameter;

public class Partial {
	public static NetParameter getPartialNN(NetParameter nn, int layersToSkip) {
		NetParameter.Builder builder = NetParameter.newBuilder(nn);
		builder.clearLayer();
		for(int i=0; i<layersToSkip; i++) {
			System.out.println("Skipping layer type: " + nn.getLayer(i).getType());
		}
		for(int i=layersToSkip; i<nn.getLayerCount(); i++) {
			System.out.println("Adding layer type: " + nn.getLayer(i).getType());
			builder.addLayer(nn.getLayer(i));
		}
		return builder.build();
	}
	
	public static void writeCaffeModel(String inputName, String outputName, int layersToSkip) throws IOException {
		InputStream input = new FileInputStream(inputName);
		OutputStream output = new FileOutputStream(outputName);
		output.write(getPartialNN(NetParameter.parseFrom(input), layersToSkip).toByteArray());
		input.close();
		output.close();
	}
	
	public static void main(String[] args) throws IOException {
		if(args.length == 0) {
			String name = "lenet_iter_10000";
			String input = "examples/mnist";
			String output = "examples/partial";
			int layersToSkip = 5;
			writeCaffeModel(input + "/" + name + ".caffemodel", output + "/init_" + name + ".caffemodel", layersToSkip);			
		} else if(args.length == 1) {
			String name = "lenet_iter_10000";
			String outputSuffix = args.length == 0 ? "" : args[0];
			String input = "examples/partial";
			String output = "/Users/obastani/Documents/temp/clone/MLDiffTest/MLDiff/data/nns/mnist";
			Main.writeCaffeModel(input + "/" + name + ".caffemodel", output + "/" + name + "-partial" + outputSuffix);
		} else {
			throw new RuntimeException("Invalid arguments!");
		}
	}
}
