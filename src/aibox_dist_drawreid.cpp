/*
 * Copyright 2021 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <math.h>
#include <vvas/vvas_kernel.h>
#include <gst/vvas/gstinferencemeta.h>
#define __STDC_FORMAT_MACROS 1
#include <stdint.h>
#include "../cros_mt_reid/src/mtmc_reid.hpp"
#include "../cros_mt_reid/src/structure.hpp"

// Additional includes for TCP/UDP server
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <string>
#include <sstream>
#include <cstring>

using namespace vitis::ai;
using namespace cv;
using namespace std;

enum
{
  LOG_LEVEL_ERROR,
  LOG_LEVEL_WARNING,
  LOG_LEVEL_INFO,
  LOG_LEVEL_DEBUG
};

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define LOG_MESSAGE(level, ...) {\
  do {\
    char *str; \
    if (level == LOG_LEVEL_ERROR)\
      str = (char*)"ERROR";\
    else if (level == LOG_LEVEL_WARNING)\
      str = (char*)"WARNING";\
    else if (level == LOG_LEVEL_INFO)\
      str = (char*)"INFO";\
    else if (level == LOG_LEVEL_DEBUG)\
      str = (char*)"DEBUG";\
    if (level <= log_level) {\
      printf("[%s %s:%d] %s: ",__FILENAME__, __func__, __LINE__, str);\
      printf(__VA_ARGS__);\
      printf("\n");\
    }\
  } while (0); \
}


int log_level = LOG_LEVEL_WARNING;


// Forward declarations
class SingleTracker;
class NetworkServer;
struct vvas_xoverlaypriv;

// Helper function declarations
bool isPointInBbox(int x, int y, const cv::Rect& bbox);
std::pair<bool, std::string> findBboxAtCoordinates(int x, int y, 
                                                  const std::vector<std::pair<cv::Rect, int>>& tracked_objects,
                                                  const std::vector<cv::Rect>& bboxes);
int getIdAtCoordinates(int x, int y, const std::vector<std::pair<cv::Rect, int>>& tracked_objects);
void drawHighlightBox(VVASFrame *inframe, vvas_xoverlaypriv *kpriv,
                     int x, int y, int size, Mat& lumaImg, Mat& chromaImg);
void drawSelectedIdBox(VVASFrame *inframe, vvas_xoverlaypriv *kpriv,
                      int selected_id, const std::vector<std::pair<cv::Rect, int>>& tracked_objects,
                      Mat& lumaImg, Mat& chromaImg);

static SingleTracker* g_tracker = nullptr;
static NetworkServer* g_server = nullptr;

// Global variables for server communication
static std::atomic<bool> g_auto_mode(false);
static std::atomic<int> g_touch_x(0);
static std::atomic<int> g_touch_y(0);
static std::atomic<bool> g_new_touch(false);
static std::atomic<int> g_selected_id(-1);  // ID of the selected object to highlight
static std::string g_last_button = "";
static std::mutex g_button_mutex;

// Network server class for TCP/UDP communication
class NetworkServer {
private:
    int tcp_socket;
    int udp_socket;
    struct sockaddr_in tcp_addr, udp_addr;
    std::atomic<bool> running;
    std::thread tcp_thread;
    std::thread udp_thread;
    
    void handleTcpConnections() {
        listen(tcp_socket, 5);
        LOG_MESSAGE(LOG_LEVEL_INFO, "TCP server listening on port 8080");
        
        while (running) {
            struct sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);
            int client_socket = accept(tcp_socket, (struct sockaddr*)&client_addr, &client_len);
            
            if (client_socket < 0) {
                if (running) {
                    LOG_MESSAGE(LOG_LEVEL_ERROR, "Failed to accept TCP connection");
                }
                continue;
            }
            
            LOG_MESSAGE(LOG_LEVEL_INFO, "TCP client connected");
            
            // Handle client in a separate thread
            std::thread([this, client_socket]() {
                char buffer[1024];
                while (running) {
                    int bytes_received = recv(client_socket, buffer, sizeof(buffer) - 1, 0);
                    if (bytes_received <= 0) {
                        break;
                    }
                    buffer[bytes_received] = '\0';
                    processMessage(std::string(buffer));
                }
                close(client_socket);
                LOG_MESSAGE(LOG_LEVEL_INFO, "TCP client disconnected");
            }).detach();
        }
    }
    
    void handleUdpMessages() {
        LOG_MESSAGE(LOG_LEVEL_INFO, "UDP server listening on port 8081");
        
        while (running) {
            char buffer[1024];
            struct sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);
            
            int bytes_received = recvfrom(udp_socket, buffer, sizeof(buffer) - 1, 0, 
                                        (struct sockaddr*)&client_addr, &client_len);
            
            if (bytes_received <= 0) {
                if (running) {
                    LOG_MESSAGE(LOG_LEVEL_ERROR, "Failed to receive UDP message");
                }
                continue;
            }
            
            buffer[bytes_received] = '\0';
            processMessage(std::string(buffer));
        }
    }
    
    void processMessage(const std::string& message) {
        LOG_MESSAGE(LOG_LEVEL_ERROR, "Received message: %s", message.c_str());
        
        if (message.find("TOUCH:") == 0) {
            // Parse TOUCH:x:y message
            size_t first_colon = message.find(':', 6);
            if (first_colon != std::string::npos) {
                try {
                    int x = std::stoi(message.substr(6, first_colon - 6));
                    int y = std::stoi(message.substr(first_colon + 1));
                    
                    g_touch_x.store(x);
                    g_touch_y.store(y);
                    g_new_touch.store(true);
                    
                    LOG_MESSAGE(LOG_LEVEL_INFO, "Touch coordinates received: (%d, %d)", x, y);
                } catch (const std::exception& e) {
                    LOG_MESSAGE(LOG_LEVEL_ERROR, "Failed to parse touch coordinates: %s", e.what());
                }
            }
        }
        else if (message.find("MODE:") == 0) {
            // Parse MODE:AUTO or MODE:MANUAL
            std::string mode = message.substr(5);
            bool auto_mode = (mode == "AUTO");
            g_auto_mode.store(auto_mode);
            
            LOG_MESSAGE(LOG_LEVEL_ERROR, "Mode changed to: %s", auto_mode ? "AUTO" : "MANUAL");
        }
        else if (message.find("BUTTON:") == 0) {
            // Parse BUTTON:direction
            std::string button = message.substr(7);
            
            {
                std::lock_guard<std::mutex> lock(g_button_mutex);
                g_last_button = button;
            }
            
            LOG_MESSAGE(LOG_LEVEL_INFO, "Button pressed: %s", button.c_str());
        }
    }
    
public:
    NetworkServer() : running(false) {
        // Initialize TCP socket
        tcp_socket = socket(AF_INET, SOCK_STREAM, 0);
        if (tcp_socket < 0) {
            LOG_MESSAGE(LOG_LEVEL_ERROR, "Failed to create TCP socket");
            return;
        }
        
        // Initialize UDP socket
        udp_socket = socket(AF_INET, SOCK_DGRAM, 0);
        if (udp_socket < 0) {
            LOG_MESSAGE(LOG_LEVEL_ERROR, "Failed to create UDP socket");
            close(tcp_socket);
            return;
        }
        
        // Set socket options
        int opt = 1;
        setsockopt(tcp_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        setsockopt(udp_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        
        // Configure TCP address
        memset(&tcp_addr, 0, sizeof(tcp_addr));
        tcp_addr.sin_family = AF_INET;
        tcp_addr.sin_addr.s_addr = INADDR_ANY;
        tcp_addr.sin_port = htons(8080);
        
        // Configure UDP address
        memset(&udp_addr, 0, sizeof(udp_addr));
        udp_addr.sin_family = AF_INET;
        udp_addr.sin_addr.s_addr = INADDR_ANY;
        udp_addr.sin_port = htons(8081);
        
        // Bind sockets
        if (bind(tcp_socket, (struct sockaddr*)&tcp_addr, sizeof(tcp_addr)) < 0) {
            LOG_MESSAGE(LOG_LEVEL_ERROR, "Failed to bind TCP socket");
            close(tcp_socket);
            close(udp_socket);
            return;
        }
        
        if (bind(udp_socket, (struct sockaddr*)&udp_addr, sizeof(udp_addr)) < 0) {
            LOG_MESSAGE(LOG_LEVEL_ERROR, "Failed to bind UDP socket");
            close(tcp_socket);
            close(udp_socket);
            return;
        }
        
        LOG_MESSAGE(LOG_LEVEL_INFO, "Network server initialized successfully");
    }
    
    ~NetworkServer() {
        stop();
    }
    
    void start() {
        if (running) {
            return;
        }
        
        running = true;
        tcp_thread = std::thread(&NetworkServer::handleTcpConnections, this);
        udp_thread = std::thread(&NetworkServer::handleUdpMessages, this);
        
        LOG_MESSAGE(LOG_LEVEL_INFO, "Network server started");
    }
    
    void stop() {
        if (!running) {
            return;
        }
        
        running = false;
        
        // Close sockets to unblock threads
        if (tcp_socket >= 0) {
            close(tcp_socket);
            tcp_socket = -1;
        }
        
        if (udp_socket >= 0) {
            close(udp_socket);
            udp_socket = -1;
        }
        
        // Join threads
        if (tcp_thread.joinable()) {
            tcp_thread.join();
        }
        
        if (udp_thread.joinable()) {
            udp_thread.join();
        }
        
        LOG_MESSAGE(LOG_LEVEL_INFO, "Network server stopped");
    }
};

// Single camera tracker implementation
class SingleTracker {
private:
  std::shared_ptr<FTD_Structure> tracker;
  uint64_t frame_count;
  cv::Mat transform; // Identity transform for single camera
  
  // Reset counters for tracking issues
  int reset_counter = 0;
  uint64_t last_reset_frame = 0;
  
public:
  SingleTracker() : frame_count(0) {
    // Create identity transform matrix for single camera tracking
    transform = cv::Mat::eye(3, 3, CV_32F);
    // Initialize tracker with camera ID 0
    tracker = std::make_shared<FTD_Structure>(transform, 0);
    
    // Configure tracker parameters for single camera tracking
    if (tracker) {
      // Set tracking parameters
      // max_age: Maximum number of frames to keep a track alive without detections
      tracker->max_age = 10000;
      // min_hits: Minimum number of detections before a track is confirmed
      tracker->min_hits = 3;
    }
  }
  
  ~SingleTracker() {
    // Release resources
    tracker.reset();
  }
  
  // Reset tracker if needed (e.g., when tracking is lost or has errors)
  void reset() {
    // Create a new tracker instance
    tracker.reset();
    tracker = std::make_shared<FTD_Structure>(transform, 0);
    
    // Configure tracker parameters
    if (tracker) {
      tracker->max_age = 30;
      tracker->min_hits = 3;
    }
    
    // Update reset statistics
    reset_counter++;
    last_reset_frame = frame_count;
    
    LOG_MESSAGE(LOG_LEVEL_WARNING, "Tracker reset at frame %lu (reset count: %d)", 
               frame_count, reset_counter);
  }
  
  // Process new detections and update tracks
  std::vector<std::pair<cv::Rect, int>> update(const std::vector<cv::Rect>& bboxes, 
                                              const std::vector<cv::Mat>& features) {
    // Prepare input characteristics for the tracker
    std::vector<InputCharact> input_characts;
    
    // Build input from bounding boxes and features
    for (size_t i = 0; i < bboxes.size() && i < features.size(); ++i) {
      if (features[i].empty()) {
        continue; // Skip if feature vector is empty
      }
      cv::Rect_<float> rect(bboxes[i].x, bboxes[i].y, bboxes[i].width, bboxes[i].height);
      input_characts.push_back(std::make_tuple(
        features[i],       // feature vector
        0,                 // orientation (default 0)
        rect,              // bounding box
        1.0f,              // detection score
        0,                 // class label (0 for person)
        -1                 // local ID (assigned by tracker)
      ));
    }
    
    // Update tracker with new detections
    frame_count++;
    std::vector<std::pair<cv::Rect, int>> tracked_objects;
    
    // If there are no detections, return empty result
    if (input_characts.empty()) {
      return tracked_objects;
    }
    
    try {
      // Update the tracker and get results
      auto results = tracker->Update(frame_count, input_characts);
      
      // Convert results to vector of rect-id pairs
      for (const auto& result : results) {
        uint64_t id = std::get<0>(result);
        cv::Rect_<float> rect = std::get<1>(result);
        cv::Rect tracked_rect(rect.x, rect.y, rect.width, rect.height);
        tracked_objects.push_back(std::make_pair(tracked_rect, (int)id));
      }
    } catch (const std::exception& e) {
      LOG_MESSAGE(LOG_LEVEL_ERROR, "Tracking update failed: %s", e.what());
      
      // Reset the tracker if necessary
      if (frame_count - last_reset_frame > 100) { // Don't reset too often
        reset();
      }
      
      // Return empty tracking results in case of error
      tracked_objects.clear();
    }
    
    return tracked_objects;
  }
};


#define MAX_CLASS_LEN 1024
#define MAX_LABEL_LEN 1024
#define MAX_ALLOWED_CLASS 20
#define MAX_ALLOWED_LABELS 20

struct color
{
  unsigned int blue;
  unsigned int green;
  unsigned int red;
};

struct vvass_xclassification
{
  color class_color;
  char class_name[MAX_CLASS_LEN];
};


struct vvas_xoverlaypriv
{
  float font_size;
  unsigned int font;
  int line_thickness;
  int y_offset;
  color label_color;
  char label_filter[MAX_ALLOWED_LABELS][MAX_LABEL_LEN];
  unsigned char label_filter_cnt;
  unsigned short classes_count;
  vvass_xclassification class_list[MAX_ALLOWED_CLASS];
  uint64_t dbg_drawind;
};

/* Get y and uv color components corresponding to givne RGB color */
void
convert_rgb_to_yuv_clrs (color clr, unsigned char *y, unsigned short *uv)
{
  Mat YUVmat;
  Mat BGRmat (2, 2, CV_8UC3, Scalar (clr.red, clr.green, clr.blue));
  cvtColor (BGRmat, YUVmat, cv::COLOR_BGR2YUV_I420);
  *y = YUVmat.at < uchar > (0, 0);
  *uv = YUVmat.at < uchar > (2, 0) << 8 | YUVmat.at < uchar > (2, 1);
  return;
}

static void DrawReID( VVASFrame *inframe, vvas_xoverlaypriv *kpriv,
  int xmin, int xmax, int ymin, int ymax, int64_t lable,
  Mat& lumaImg, Mat& chromaImg)
{
  /* Check whether the frame is NV12 or BGR and act accordingly */
  char label_s[256];
  sprintf(label_s, "%lu", lable);
  std::string label_string(label_s);

  if (inframe->props.fmt == VVAS_VFMT_Y_UV8_420)
  {
    unsigned char yScalar;
    unsigned short uvScalar;
    color clr = {255, 0, 0};
    convert_rgb_to_yuv_clrs(clr, &yScalar, &uvScalar);
    /* Draw rectangle on y an uv plane */
    int new_xmin = floor(xmin / 2) * 2;
    int new_ymin = floor(ymin / 2) * 2;
    int new_xmax = floor(xmax / 2) * 2;
    int new_ymax = floor(ymax / 2) * 2;
    int h = new_ymax - new_ymin;
    int w = new_xmax - new_xmin;

    /* Lets not draw anything when the origin is (0,0) */
    if (new_xmin || new_ymin)
    {
      rectangle(lumaImg, Point(new_xmin, new_ymin), Point(new_xmax, new_ymax),
                Scalar(yScalar), kpriv->line_thickness, 1, 0);
      rectangle(chromaImg, Point(new_xmin / 2, new_ymin / 2),
                Point(new_xmax / 2, new_ymax / 2),
                Scalar(uvScalar), kpriv->line_thickness, 1, 0);
    }
    {
      int baseline, y_offset = 0;
      Size textsize = getTextSize(label_string, kpriv->font, kpriv->font_size, 1, &baseline);
      if ((h < 1) && (w < 1))
      {
        if (kpriv->y_offset)
        {
          y_offset = kpriv->y_offset;
        }
        else
        {
          y_offset = (inframe->props.height * 0.10);
        }
      }
      if ( lable >= 0 ) {
      /* Draw filled rectangle for labelling, both on y and uv plane */
      rectangle(lumaImg, Rect(Point(new_xmin, new_ymin - textsize.height), textsize),
                Scalar(yScalar), FILLED, 1, 0);
      textsize.height /= 2;
      textsize.width /= 2;
      rectangle(chromaImg, Rect(Point(new_xmin / 2, new_ymin / 2 - textsize.height), textsize),
                Scalar(uvScalar), FILLED, 1, 0);

      /* Draw label text on the filled rectanngle */
      convert_rgb_to_yuv_clrs(kpriv->label_color, &yScalar, &uvScalar);
      putText(lumaImg, label_string, cv::Point(new_xmin, new_ymin + y_offset), kpriv->font, kpriv->font_size,
              Scalar(yScalar), kpriv->line_thickness, 1);
      putText(chromaImg, label_string, cv::Point(new_xmin / 2, new_ymin / 2 + y_offset / 2), kpriv->font,
              kpriv->font_size / 2, Scalar(uvScalar), kpriv->line_thickness, 1);
      }
    }
  }
}

extern "C"
{
  int32_t xlnx_kernel_init (VVASKernel * handle)
  {
    vvas_xoverlaypriv *kpriv =
        (vvas_xoverlaypriv *) malloc (sizeof (vvas_xoverlaypriv));
    memset (kpriv, 0, sizeof (vvas_xoverlaypriv));

    // Initialize the single camera tracker
    try {
      g_tracker = new SingleTracker();
      LOG_MESSAGE(LOG_LEVEL_INFO, "Successfully initialized single camera tracker");
    } catch (const std::exception& e) {
      LOG_MESSAGE(LOG_LEVEL_ERROR, "Failed to initialize single camera tracker: %s", e.what());
      return -1;
    }

    // Initialize and start the network server
    try {
      g_server = new NetworkServer();
      g_server->start();
      LOG_MESSAGE(LOG_LEVEL_INFO, "Successfully initialized network server");
    } catch (const std::exception& e) {
      LOG_MESSAGE(LOG_LEVEL_ERROR, "Failed to initialize network server: %s", e.what());
      // Continue without server functionality
    }

    json_t *jconfig = handle->kernel_config;
    json_t *val, *karray = NULL, *classes = NULL;

    /* Initialize config params with default values */
    log_level = LOG_LEVEL_WARNING;
    kpriv->font_size = 0.5;
    kpriv->font = 0;
    kpriv->line_thickness = 1;
    kpriv->y_offset = 0;
    kpriv->label_color = {0, 0, 0};
    strcpy(kpriv->label_filter[0], "class");
    strcpy(kpriv->label_filter[1], "probability");
    kpriv->label_filter_cnt = 2;
    kpriv->classes_count = 0;
    kpriv->dbg_drawind = 0;

      val = json_object_get (jconfig, "debug_level");
    if (!val || !json_is_integer (val))
        log_level = LOG_LEVEL_WARNING;
    else
        log_level = json_integer_value (val);

      val = json_object_get (jconfig, "font_size");
    if (!val || !json_is_integer (val))
        kpriv->font_size = 0.5;
    else
        kpriv->font_size = json_integer_value (val);

      val = json_object_get (jconfig, "font");
    if (!val || !json_is_integer (val))
        kpriv->font = 0;
    else
        kpriv->font = json_integer_value (val);

      val = json_object_get (jconfig, "thickness");
    if (!val || !json_is_integer (val))
        kpriv->line_thickness = 1;
    else
        kpriv->line_thickness = json_integer_value (val);

      val = json_object_get (jconfig, "y_offset");
    if (!val || !json_is_integer (val))
        kpriv->y_offset = 0;
    else
        kpriv->y_offset = json_integer_value (val);

    /* get label color array */
      karray = json_object_get (jconfig, "label_color");
    if (!karray)
    {
      LOG_MESSAGE (LOG_LEVEL_ERROR, "failed to find label_color");
      return -1;
    } else
    {
      kpriv->label_color.blue =
          json_integer_value (json_object_get (karray, "blue"));
      kpriv->label_color.green =
          json_integer_value (json_object_get (karray, "green"));
      kpriv->label_color.red =
          json_integer_value (json_object_get (karray, "red"));
    }

    karray = json_object_get (jconfig, "label_filter");

    if (!json_is_array (karray)) {
      LOG_MESSAGE (LOG_LEVEL_ERROR, "failed to find key label_filter");
      return -1;
    }

    // Parse label_filter array (strings)
    kpriv->label_filter_cnt = json_array_size (karray);
    if (kpriv->label_filter_cnt > MAX_ALLOWED_LABELS) {
      kpriv->label_filter_cnt = MAX_ALLOWED_LABELS;
    }
    for (unsigned int index = 0; index < kpriv->label_filter_cnt; index++) {
      val = json_array_get (karray, index);
      if (!json_is_string (val)) {
        LOG_MESSAGE (LOG_LEVEL_ERROR, "label_filter[%d] is not a string", index);
        return -1;
      } else {
        strncpy (kpriv->label_filter[index],
            (char *) json_string_value (val), MAX_LABEL_LEN - 1);
        LOG_MESSAGE (LOG_LEVEL_DEBUG, "label_filter[%d] = %s", index,
            kpriv->label_filter[index]);
      }
    }

    // Parse classes array (objects with name and color properties)
    karray = json_object_get (jconfig, "classes");
    if (!json_is_array (karray)) {
      LOG_MESSAGE (LOG_LEVEL_ERROR, "failed to find key classes");
      return -1;
    }

    kpriv->classes_count = json_array_size (karray);
    for (unsigned int index = 0; index < kpriv->classes_count; index++) {
      classes = json_array_get (karray, index);
      if (!classes) {
        LOG_MESSAGE (LOG_LEVEL_ERROR, "failed to get class object");
        return -1;
      }

      val = json_object_get (classes, "name");
      if (!json_is_string (val)) {
        LOG_MESSAGE (LOG_LEVEL_ERROR, "name is not found for array %d", index);
        return -1;
      } else {
        strncpy (kpriv->class_list[index].class_name,
            (char *) json_string_value (val), MAX_CLASS_LEN - 1);
        LOG_MESSAGE (LOG_LEVEL_DEBUG, "name %s",
            kpriv->class_list[index].class_name);
      }

      val = json_object_get (classes, "green");
      if (!val || !json_is_integer (val))
        kpriv->class_list[index].class_color.green = 0;
      else
        kpriv->class_list[index].class_color.green = json_integer_value (val);

      val = json_object_get (classes, "blue");
      if (!val || !json_is_integer (val))
        kpriv->class_list[index].class_color.blue = 0;
      else
        kpriv->class_list[index].class_color.blue = json_integer_value (val);

      val = json_object_get (classes, "red");
      if (!val || !json_is_integer (val))
        kpriv->class_list[index].class_color.red = 0;
      else
        kpriv->class_list[index].class_color.red = json_integer_value (val);
    }

    handle->kernel_priv = (void *) kpriv;
    return 0;
  }

  uint32_t xlnx_kernel_deinit (VVASKernel * handle)
  {
    LOG_MESSAGE (LOG_LEVEL_DEBUG, "enter");
    vvas_xoverlaypriv *kpriv = (vvas_xoverlaypriv *) handle->kernel_priv;

    // Clean up single camera tracker
    if (g_tracker != nullptr) {
      try {
        delete g_tracker;
        g_tracker = nullptr;
        LOG_MESSAGE(LOG_LEVEL_INFO, "Successfully cleaned up single camera tracker");
      } catch (const std::exception& e) {
        LOG_MESSAGE(LOG_LEVEL_ERROR, "Error cleaning up tracker: %s", e.what());
      }
    }

    // Clean up network server
    if (g_server != nullptr) {
      try {
        delete g_server;
        g_server = nullptr;
        LOG_MESSAGE(LOG_LEVEL_INFO, "Successfully cleaned up network server");
      } catch (const std::exception& e) {
        LOG_MESSAGE(LOG_LEVEL_ERROR, "Error cleaning up server: %s", e.what());
      }
    }

    if (kpriv)
      free (kpriv);

    return 0;
  }


  uint32_t xlnx_kernel_start (VVASKernel * handle, int start,
      VVASFrame * input[MAX_NUM_OBJECT], VVASFrame * output[MAX_NUM_OBJECT])
  {
    static int num = 0;
    static std::vector<std::pair<cv::Rect, int>> *ids = nullptr; // Storage for tracking IDs
    VVASFrame *inframe = input[0];
    vvas_xoverlaypriv *kpriv = (vvas_xoverlaypriv *)handle->kernel_priv;

    vector<Rect> bboxes;
    vector<Mat> feats;
    if (inframe->props.fmt == VVAS_VFMT_Y_UV8_420)
    {
      LOG_MESSAGE(LOG_LEVEL_DEBUG, "Input frame is in NV12 format\n");
      Mat lumaImg(input[0]->props.height, input[0]->props.stride, CV_8UC1, (char *)inframe->vaddr[0]);
      Mat chromaImg(input[0]->props.height / 2, input[0]->props.stride / 2, CV_16UC1, (char *)inframe->vaddr[1]);

      GstInferenceMeta *infer_meta = ((GstInferenceMeta *)gst_buffer_get_meta((GstBuffer *)
                                                                inframe->app_priv,
                                                            gst_inference_meta_api_get_type()));

      if (infer_meta == NULL || infer_meta->prediction == NULL)
      {
          LOG_MESSAGE(LOG_LEVEL_INFO, "vvas meta data is not available for draw tracking box");
          return 0; // Return 0 instead of false (incorrect return type)
      }

      {
        struct cros_reid_info* data = (struct cros_reid_info*)infer_meta->prediction->reserved_3;
        if (data != nullptr) {
          for(auto &p : data->person_infos) {
            Rect rect(p.bbox[0], p.bbox[1], p.bbox[2], p.bbox[3]);
            bboxes.push_back(rect);
            feats.push_back(p.reid_feat); // Use the reid_feat field from person_info
          }
        }

        // Apply tracking to get IDs for the detected objects
        std::vector<std::pair<cv::Rect, int>> tracked_objects;
        if (!bboxes.empty() && g_tracker != nullptr) {
          LOG_MESSAGE(LOG_LEVEL_DEBUG, "Running tracking update with %zu detections", bboxes.size());
          tracked_objects = g_tracker->update(bboxes, feats);
          LOG_MESSAGE(LOG_LEVEL_DEBUG, "Tracking update completed with %zu tracked objects", tracked_objects.size());
          
          // Store tracking result for later use
          if (ids) {
            delete ids;
            ids = nullptr;
          }
          ids = new std::vector<std::pair<cv::Rect, int>>(tracked_objects);
          infer_meta->prediction->reserved_5 = (void*)ids;
        } else if (g_tracker == nullptr) {
          LOG_MESSAGE(LOG_LEVEL_ERROR, "Tracker is not initialized");
        }

        // Handle server messages
        if (g_auto_mode.load()) {
          // AUTO mode - handle touch coordinates
          if (g_new_touch.load()) {
            int touch_x = g_touch_x.load();
            int touch_y = g_touch_y.load();
            g_new_touch.store(false);
            
            LOG_MESSAGE(LOG_LEVEL_INFO, "Processing touch at (%d, %d) in AUTO mode", touch_x, touch_y);
            
            // Check if touch coordinates are inside any of the drawn bounding boxes
            int selected_id = -1;
            std::string bbox_info = "";
            bool found = false;
            
            if (data != nullptr && ids != nullptr) {
              int ratiow = 1, ratioh = 1;
              
              // Check each drawn bounding box
              for (int i = 0; i < data->person_infos.size() && i < ids->size(); i++) {
                auto rect = (*ids)[i].first;
                int wmin = rect.x * ratiow;
                int wmax = (rect.x + rect.width) * ratiow;
                int hmin = rect.y * ratioh;
                int hmax = (rect.y + rect.height) * ratioh;
                int id = (*ids)[i].second;
                
                // Check if touch is inside this bounding box
                if (touch_x >= wmin && touch_x <= wmax && touch_y >= hmin && touch_y <= hmax) {
                  selected_id = id;
                  found = true;
                  
                  // Create bbox info string
                  std::ostringstream oss;
                  oss << "Tracked Object - ID: " << id 
                      << ", Bbox: (" << wmin << "," << hmin 
                      << "," << (wmax-wmin) << "," << (hmax-hmin) << ")";
                  bbox_info = oss.str();
                  break;
                }
              }
            }
            
            if (found) {
              g_selected_id.store(selected_id);
              
              // Print to terminal (console output)
              printf("=== OBJECT SELECTED ===\n");
              printf("Touch coordinates: (%d, %d)\n", touch_x, touch_y);
              printf("Selected object info: %s\n", bbox_info.c_str());
              printf("Selected ID: %d\n", selected_id);
              printf("======================\n");
              fflush(stdout);
              
              LOG_MESSAGE(LOG_LEVEL_ERROR, "Touch hit: %s", bbox_info.c_str());
              LOG_MESSAGE(LOG_LEVEL_ERROR, "Selected ID: %d", selected_id);
              
              // Draw red highlight box at touch point
              drawHighlightBox(inframe, kpriv, touch_x, touch_y, 50, lumaImg, chromaImg);
            } else {
              g_selected_id.store(-1);  // Clear selection
              printf("=== NO OBJECT SELECTED ===\n");
              printf("Touch coordinates: (%d, %d)\n", touch_x, touch_y);
              printf("No object found at touch location\n");
              printf("=========================\n");
              fflush(stdout);
              
              LOG_MESSAGE(LOG_LEVEL_INFO, "Touch at (%d, %d) - No bbox found", touch_x, touch_y);
            }
          }
        } else {
          // MANUAL mode - handle button presses
          std::string current_button;
          {
            std::lock_guard<std::mutex> lock(g_button_mutex);
            current_button = g_last_button;
            g_last_button = ""; // Clear after reading
          }
          
          if (!current_button.empty()) {
            LOG_MESSAGE(LOG_LEVEL_INFO, "MANUAL mode - Button pressed: %s", current_button.c_str());
          }
        }

        std::ostringstream oss;
        oss << "drawreid: see attached data ptr: " << infer_meta->prediction->reserved_3 << "\n";
        if (data != NULL) {
          //int ratiow = 1920 / 480, ratioh = 1080 / 360;
          int ratiow = 1, ratioh = 1;
          int selected_id = g_selected_id.load();
          
          // Draw tracked objects with their IDs
          for (int i = 0; data && i < data->person_infos.size(); i++) {
            int wmin, wmax, hmin, hmax, id;
            if (ids == nullptr || i >= ids->size()) {
              struct person_info & tmp = data->person_infos[i];
              wmin = tmp.bbox[0] * ratiow;
              wmax = (tmp.bbox[0] + tmp.bbox[2]) * ratiow;
              hmin = tmp.bbox[1] * ratioh;
              hmax = (tmp.bbox[1] + tmp.bbox[3]) * ratioh;
              id = -1;
            } else {
              auto rect = (*ids)[i].first;
              wmin = rect.x * ratiow;
              wmax = (rect.x + rect.width)  * ratiow;
              hmin = rect.y * ratioh;
              hmax = (rect.y + rect.height)  * ratioh;
              id = (*ids)[i].second;
            }
            // Make sure the coordinates are valid before drawing
            if (wmin >= 0 && hmin >= 0 && wmax > wmin && hmax > hmin) {
              // Draw the normal tracking box
              DrawReID(inframe, kpriv, wmin, wmax, hmin, hmax, id,
                      lumaImg, chromaImg);
              
              // If this is the selected ID, draw an additional red highlight box
              if (selected_id >= 0 && id == selected_id) {
                drawSelectedIdBox(inframe, kpriv, id, tracked_objects, lumaImg, chromaImg);
                LOG_MESSAGE(LOG_LEVEL_ERROR, "Tracking: Object %d at (%d,%d,%d,%d)", 
                         id, wmin, hmin, wmax, hmax);
              }
              
              oss << "Frame Ind: " << data->frame_id
                  << ", drawreid channel " << 0 // Using 0 for single camera
                  << ": bbox[0]=" << wmin << ", bbox[1]=" << hmin
                  << ", bbox[2]=" << wmax << ", bbox[3]=" << hmax
                  << ", id=" << id << "\n";
              
              // Log tracking details at debug level

            }
          }
        }

    unsigned char yScalar;
    unsigned short uvScalar;
    color clr = {255, 0, 0};
    convert_rgb_to_yuv_clrs(clr, &yScalar, &uvScalar);

    if (log_level == LOG_LEVEL_DEBUG)
    {
      std::ostringstream label;
      label << (data ? data->frame_id : -1) << ", " << kpriv->dbg_drawind++;
      // Add tracking info to the debug display
      if (g_tracker != nullptr) {
        label << " (tracked: " << (ids ? ids->size() : 0) << ")";
      }
      // Add server mode info
      label << " [" << (g_auto_mode.load() ? "AUTO" : "MANUAL") << "]";
      
      putText(lumaImg, label.str(), cv::Point(200, 200), kpriv->font, kpriv->font_size,
            Scalar(yScalar), kpriv->line_thickness, 1);
      putText(chromaImg, label.str(), cv::Point(100, 100), kpriv->font, kpriv->font_size / 2,
            Scalar(uvScalar), kpriv->line_thickness, 1);
    }

        // Clean up tracking results
        if (ids) {
          delete ids;
          ids = nullptr;
          if (infer_meta && infer_meta->prediction) {
            infer_meta->prediction->reserved_5 = NULL;
          }
        }
        
        // Clean up detection data
        if (data) {
          LOG_MESSAGE(LOG_LEVEL_DEBUG, "CH %lu, drawreid remove result %lu, %p.", 
                     (infer_meta && infer_meta->prediction) ? (uint64_t)infer_meta->prediction->reserved_2 : 0, 
                     data->frame_id, data);
          delete data;
          if (infer_meta && infer_meta->prediction) {
            infer_meta->prediction->reserved_3 = NULL;
          }
        }

      }

      GstInferencePrediction *root = infer_meta->prediction;
      num++;

      return 0;
    }
    else
    {
      LOG_MESSAGE(LOG_LEVEL_WARNING, "Unsupported color format\n");
      return 0;
    }
return 0;
  }

  int32_t xlnx_kernel_done (VVASKernel * handle)
  {
    LOG_MESSAGE (LOG_LEVEL_DEBUG, "done");
    return 0;
  }
}

// Helper function to check if coordinates are within a bounding box
bool isPointInBbox(int x, int y, const cv::Rect& bbox) {
    return x >= bbox.x && x <= (bbox.x + bbox.width) && 
           y >= bbox.y && y <= (bbox.y + bbox.height);
}

// Helper function to find bbox information at given coordinates
std::pair<bool, std::string> findBboxAtCoordinates(int x, int y, 
                                                  const std::vector<std::pair<cv::Rect, int>>& tracked_objects,
                                                  const std::vector<cv::Rect>& bboxes) {
    std::ostringstream info;
    bool found = false;
    
    // Check tracked objects first
    for (const auto& obj : tracked_objects) {
        if (isPointInBbox(x, y, obj.first)) {
            info << "Tracked Object - ID: " << obj.second 
                 << ", Bbox: (" << obj.first.x << "," << obj.first.y 
                 << "," << obj.first.width << "," << obj.first.height << ")";
            found = true;
            break;
        }
    }
    
    // If not found in tracked objects, check original detections
    if (!found) {
        for (size_t i = 0; i < bboxes.size(); i++) {
            if (isPointInBbox(x, y, bboxes[i])) {
                info << "Detection - Index: " << i 
                     << ", Bbox: (" << bboxes[i].x << "," << bboxes[i].y 
                     << "," << bboxes[i].width << "," << bboxes[i].height << ")";
                found = true;
                break;
            }
        }
    }
    
    return std::make_pair(found, info.str());
}

// Helper function to get ID at given coordinates
int getIdAtCoordinates(int x, int y, const std::vector<std::pair<cv::Rect, int>>& tracked_objects) {
    // Check tracked objects for the ID
    for (const auto& obj : tracked_objects) {
        if (isPointInBbox(x, y, obj.first)) {
            return obj.second;
        }
    }
    return -1;  // No ID found
}

// Helper function to draw a red highlight box around the selected ID
void drawSelectedIdBox(VVASFrame *inframe, vvas_xoverlaypriv *kpriv,
                      int selected_id, const std::vector<std::pair<cv::Rect, int>>& tracked_objects,
                      Mat& lumaImg, Mat& chromaImg) {
    
    // Find the bounding box for the selected ID
    cv::Rect selected_bbox;
    bool found = false;
    
    for (const auto& obj : tracked_objects) {
        if (obj.second == selected_id) {
            selected_bbox = obj.first;
            found = true;
            break;
        }
    }
    
    if (!found) {
        return;  // Selected ID not found
    }
    
    if (inframe->props.fmt == VVAS_VFMT_Y_UV8_420) {
        unsigned char yScalar;
        unsigned short uvScalar;
        color clr = {0, 0, 255}; // Red color
        convert_rgb_to_yuv_clrs(clr, &yScalar, &uvScalar);
        
        // Use the same coordinate system as the drawn boxes
        int ratiow = 1, ratioh = 1;
        int wmin = selected_bbox.x * ratiow;
        int wmax = (selected_bbox.x + selected_bbox.width) * ratiow;
        int hmin = selected_bbox.y * ratioh;
        int hmax = (selected_bbox.y + selected_bbox.height) * ratioh;
        
        // Add some padding to make the red box larger than the original
        int padding = 10;
        int xmin = std::max(0, wmin - padding);
        int ymin = std::max(0, hmin - padding);
        int xmax = std::min((int)inframe->props.width, wmax + padding);
        int ymax = std::min((int)inframe->props.height, hmax + padding);
        
        // Align coordinates for NV12 format
        int new_xmin = floor(xmin / 2) * 2;
        int new_ymin = floor(ymin / 2) * 2;
        int new_xmax = floor(xmax / 2) * 2;
        int new_ymax = floor(ymax / 2) * 2;
        
        // Draw thick red rectangle on y and uv planes
        int thick_line = kpriv->line_thickness * 4;  // Make it thicker
        rectangle(lumaImg, Point(new_xmin, new_ymin), Point(new_xmax, new_ymax),
                  Scalar(yScalar), thick_line, 1, 0);
        rectangle(chromaImg, Point(new_xmin / 2, new_ymin / 2),
                  Point(new_xmax / 2, new_ymax / 2),
                  Scalar(uvScalar), thick_line, 1, 0);
        
    }
}

// Helper function to draw a red highlight box
void drawHighlightBox(VVASFrame *inframe, vvas_xoverlaypriv *kpriv,
                     int x, int y, int size, Mat& lumaImg, Mat& chromaImg) {
    // Draw a red box around the touch point
    int half_size = size / 2;
    int xmin = std::max(0, x - half_size);
    int ymin = std::max(0, y - half_size);
    int xmax = std::min((int)inframe->props.width, x + half_size);
    int ymax = std::min((int)inframe->props.height, y + half_size);
    
    if (inframe->props.fmt == VVAS_VFMT_Y_UV8_420) {
        unsigned char yScalar;
        unsigned short uvScalar;
        color clr = {0, 0, 255}; // Red color
        convert_rgb_to_yuv_clrs(clr, &yScalar, &uvScalar);
        
        // Align coordinates for NV12 format
        int new_xmin = floor(xmin / 2) * 2;
        int new_ymin = floor(ymin / 2) * 2;
        int new_xmax = floor(xmax / 2) * 2;
        int new_ymax = floor(ymax / 2) * 2;
        
        // Draw rectangle on y and uv planes
        rectangle(lumaImg, Point(new_xmin, new_ymin), Point(new_xmax, new_ymax),
                  Scalar(yScalar), kpriv->line_thickness * 2, 1, 0);
        rectangle(chromaImg, Point(new_xmin / 2, new_ymin / 2),
                  Point(new_xmax / 2, new_ymax / 2),
                  Scalar(uvScalar), kpriv->line_thickness * 2, 1, 0);
    }
}