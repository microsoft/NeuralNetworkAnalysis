What's in here?
===============

The little utility here can be used to convert Caffe protobuf network
descriptions to the format recognized by our tool.

NB: make sure that when you train Caffe networks you dump protobuf
descriptions of these networks. Recent versions of Caffe changed the
default to a non-protobuf format, so you need to manually revert by
changing configuration files in your Caffe examples directory.

How to build and run
--------------------
First download the Java protobuf jar in `lib/protobuf-java-2.6.0.jar`. To build:
```
ant
``` 
To run: 
``` 
java -cp lib/protobuf-java-2.6.0.jar:bin protobuf.Main <input> <output>
```
To clean:
```
ant clean
```