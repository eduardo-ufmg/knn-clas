#!/bin/bash

protoc --cpp_out=../src/common classifier.proto
protoc --python_out=../scripts classifier.proto
