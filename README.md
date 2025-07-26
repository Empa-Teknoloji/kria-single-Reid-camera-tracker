<h1 align="center">KRIA SINGLE CAMERA REID TRACKER</h1>

## Introduction
This repository contains source code of Kria Single Camera ReID Tracker application. 
This application is a modified version of the original Xilinx aibox-dist project, adapted for single camera tracking applications. The application performs real-time pedestrian detection, tracking, and Re-Identification on a single camera stream using machine learning acceleration on Xilinx Kria SoM platforms.

## Original Project Attribution

This work is based on the Xilinx aibox-dist project:
- **Original Repository**: https://github.com/Xilinx/aibox-dist
- **License**: Apache-2.0
- **Copyright**: (C) 2010 - 2021 Xilinx, Inc.

## Modifications

This version has been modified to support single camera tracking, removing the multi-camera distributed architecture from the original project.

## Development Guide

If you want to cross compile the source in Linux PC machine, follow these steps, otherwise skip this section.

1. Refer to the `K260 SOM Starter Kit Tutorial` to build the cross-compilation SDK, and install it to the path you choose or default. Suppose it's SDKPATH.

2. Run "./build.sh ${SDKPATH}" in the source code folder of current application, to build the application. <a name="build-app"></a>

3. The build process in [2](#build-app) will produce a rpm package aibox-dist-1.0.1-1.aarch64.rpm under build/, upload to the board, and run "rpm -ivh --force ./aibox-dist-1.0.1-1.aarch64.rpm" to update install.

## Setting up the Board and Application Deployment
A step by step tutorial and details on how to setup the board and run this application is given in the [AIBox Distributed Reid Documentation](https://xilinx.github.io/kria-apps-docs/kv260/2022.1/build/html/docs/aibox/docs/app_deployment_aib.html#). Please visit the documentation page for more details.


### **How to Run The Application**

This application is designed for single camera tracking with pedestrian detection and ReID capabilities.

### Start The Application

**Note** The application needs to be run with ***sudo***.

#### Examples

  This example starts tracking by using the MIPI camera as input source.

  ```bash
  sudo aibox-dist-cam -m 
  ```

  You can also start the tracking by using video files as the input source, this is the recommended mode to start with.

  ```bash
  sudo aibox-dist-cam -f video_file.h264
  ```

**Note**: Only one instance of aibox-dist-cam application can run at a time because it requires exclusive access to a DPU engine and there is only one instance of DPU that exists in the platform.

##### Command Options

The example shows a simple way to invoke the server application.

User can get more and detailed application options as following by invoking

`aibox-dist-cam --help`

```
Usage:
  aibox-dist-cam [OPTION?] - Single camera tracking application with pedestrian detection and ReID on Xilinx Kria SoM.

Help Options:
  -h, --help                        Show help options
  --help-all                        Show all help options
  --help-gst                        Show GStreamer Options

Application Options:
  -m, --mipi=                       use MIPI camera as input source, auto detect, fail if no mipi connected
  -u, --usb=media ID                usb camera media device id, e.g. 0 for /dev/media0
  -f, --file=file path              location of h26x file as input
  -i, --infile-type=h264            input file type: [h264 | h265]
  -W, --width=1920                  resolution w of the input
  -H, --height=1080                 resolution h of the input
  -r, --framerate=30                framerate of the input
  -c, --camid=0                     cam ID
  -l, --fileloop=loop file source   loop file source 
  -o, --outmedia-type=h264          output file type: [h264 | h265]
  -p, --port=554                    Port to listen on (default: 554)
  -n, --nodet                       no AI inference
  -A, --audio                       RTSP with I2S audio
  -R, --report                      report fps
  -a, --noattach                    Not attach AI info, default: true
  --ROI-off                         turn off ROI
  --control-rate=low-latency        Encoder parameter control-rate
  --target-bitrate=3000             Encoder parameter target-bitrate
  --gop-length=60                   Encoder parameter gop-length
  --profile                         Encoder parameter profile.
  --level                           Encoder parameter level
  --tier                            Encoder parameter tier
  --encodeEnhancedParam             String for fully customizing the encoder in the form "param1=val1, param2=val2,...", where paramn is the name of the encoder parameter
```

### Display and Monitoring

The single camera tracker application provides real-time display of tracking results with pedestrian detection and ReID visualization directly on the connected display or through RTSP streaming.

## Files structure

The application is installed as:

* Binary File Directory: /opt/xilinx/kv260-aibox-dist/bin

  | filename | description |
  |----------|-------------|
  |aibox-dist-cam | Single camera tracking application with pedestrian detection and ReID|

* Configuration file directory: /opt/xilinx/kv260-aibox-dist/share/ivas/

  | filename | description |
  |-|-|
  |cam_setup.json       |           Config of camera set up and calibration.
  |ped_pp.json       |           Config of refinedet preprocess.
  | refinedet.json   |           Config of refinedet.
  | crop.json        |           Config of cropping for reid.
  | reid.json        |           Config of reid.

*  Model files: => /opt/xilinx/kv260-aibox-dist/share/vitis_ai_library/models

  The model files integrated in the application use the B3136 DPU configuration.

  |      foldername      | description |
  |----------------------|-------------|
  |person-orientation_pruned_558m_pt | Model files for personal orientation|
  |personreid-res18_pt   | Model files for ReID |
  |refinedet_pruned_0_92 | Model files for refinedet |
  |FairMot               | Model files for pedestrian detection|


<p align="center"><sup>Copyright&copy; 2021 Xilinx</sup></p>
