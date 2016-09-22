#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <fstream>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/layers/memory_data_layer.hpp"
#include "caffe/layers/memory_roi_layer.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace caffe;
using boost::shared_ptr;
using std::string;

typedef shared_ptr<Net<float> > NetP;
typedef shared_ptr<Blob<float> > BlobP;

int feature_extraction_pipeline(int argc, char** argv);

int main(int argc, char** argv) {
  return feature_extraction_pipeline(argc, argv);
//  return feature_extraction_pipeline<double>(argc, argv);
}

// this one is for multibox output which is float form 0 to 1
std::vector<std::vector<int> > scale_rois(const std::vector<std::vector<float> >& rois_arr, int w, int h)
{
  std::vector<std::vector<int> > res;
  for(int i = 0; i < rois_arr.size(); i++)
  {
    for(int j = 0; j < rois_arr[i].size(); j+=4)
    {
      std::vector<int> roi;
      int x_tl, y_tl, x_br, y_br;
      float xr, yr, wr, hr;
      float x1, y1, x2, y2;
      xr = rois_arr[i][j + 0];
      yr = rois_arr[i][j + 1];
      wr = rois_arr[i][j + 2];
      hr = rois_arr[i][j + 3];

      x1 = xr - wr / float(2);
      y1 = yr - hr / float(2);
      x2 = xr + wr / float(2);
      y2 = yr + hr / float(2);

      if (x1 < 0) x1 = 0.0f;
      if (x1 > 1) x1 = 1.0f;
      if (y1 < 0) y1 = 0.0f;
      if (y1 > 1) y1 = 1.0f;
      if (x2 < 0) x2 = 0.0f;
      if (x2 > 1) x2 = 1.0f;
      if (y2 < 0) y2 = 0.0f;
      if (y2 > 1) y2 = 1.0f;
      if (x2 < x1) std::swap(x1, x2);
      if (y2 < y1) std::swap(y1, y2);

      x_tl = x1 * (w - 1);
      y_tl = y1 * (h - 1);
      x_br = x2 * (w - 1);
      y_br = y2 * (h - 1);

      roi.push_back(x_tl);
      roi.push_back(y_tl);
      roi.push_back(x_br - x_tl);
      roi.push_back(y_br - y_tl);

      res.push_back(roi);
    }
  }

  return res;
}

//this one is used in frcnn for multibox scaled output which is scaled respective to original image size
std::vector<std::vector<int> > scale_rois(const std::vector<std::vector<int> >& rois, float coeff)
{
  std::vector<std::vector<int> > res;
  for(int i = 0; i < rois.size(); i++)
  {
    std::vector<int> roi;
    roi.push_back(static_cast<int>(rois[i][0]*coeff));
    roi.push_back(static_cast<int>(rois[i][1]*coeff));
    roi.push_back(static_cast<int>(rois[i][2]*coeff));
    roi.push_back(static_cast<int>(rois[i][3]*coeff));
    res.push_back(roi);
  }
  return res;
}

//this one used by multibox to convert location prediction into float rois
std::vector<std::vector<float> > lp2rois(const std::vector<std::vector<float> >& rois_arr, const std::vector<std::vector<float> >& priors)
{
  std::vector<std::vector<float> > res(rois_arr);
  for(int i = 0; i < rois_arr.size(); i++)
  {
    for(int j = 0; j < rois_arr[i].size(); j+=4) {
      float xp = priors[j/4][0];
      float yp = priors[j/4][1];
      res[i][j+0] += xp;
      res[i][j+1] += yp;
    }
  }
  return res;
}

void set_up_multibox(NetP net, const cv::Mat& im)
{
  shared_ptr<MemoryDataLayer<float> > mdl = boost::dynamic_pointer_cast<MemoryDataLayer<float> >(net->layer_by_name("data"));
  CHECK(mdl) << "There is no memory data layer in the net proto";

  cv::Mat img;
  cv::resize(im, img, cv::Size(224, 224), cv::INTER_AREA);
  std::vector<cv::Mat> images(1, img);
  std::vector<int> labels(1, 0);
  mdl->AddMatVector(images, labels);
}

void set_up_frcnn(NetP net, const cv::Mat& im, const std::vector<std::vector<int> >& rois)
{
  const int num_rois = 100;
  const int min_side = 600;
  int w = im.cols;
  int h = im.rows;

  std::vector<std::vector<int> > rois2net(rois.begin(), rois.begin()+num_rois);
  double coeff = double(min_side)/double(std::min(w,h));
  cv::Mat img;
  cv::resize(im, img, cv::Size(w*coeff, h*coeff), cv::INTER_AREA);
  std::vector<std::vector<int> > s_rois = scale_rois(rois2net, coeff);

  shared_ptr<MemoryDataLayer<float> > mdl = boost::dynamic_pointer_cast<MemoryDataLayer<float> >(net->layer_by_name("data"));
  CHECK(mdl) << "There is no memory data layer in the net proto";

  mdl->AddCvMat(img, 0);

  shared_ptr<MemoryROILayer<float> > mrl = boost::dynamic_pointer_cast<MemoryROILayer<float> >(net->layer_by_name("rois"));
  CHECK(mdl) << "There is no memory roi layer in the net proto";
  mrl->AddROIsSingleLevel(s_rois);
}

template <class Dtype>
void dump_ftrs(const std::vector<std::vector<Dtype> >& ftrs, const std::string& of)
{
  std::ofstream os(of.c_str());
  for(int i = 0; i < ftrs.size(); i++)
  {
    for(int j = 0; j < ftrs[i].size(); j++)
      os << ftrs[i][j] << " ";
    os << std::endl;
  }
  os.close();
}

std::vector<std::vector<int> > load_rois(const std::string& dump)
{
  std::vector<std::vector<int> > rois;
  int x, y, w, h;
  std::ifstream is(dump.c_str());
  CHECK(is.good()) << "Wrong roi dump path: " << dump ;
  while(is >> x >> y >> w >> h)
  {
    std::vector<int> roi;
    roi.push_back(x);
    roi.push_back(y);
    roi.push_back(w);
    roi.push_back(h);
    rois.push_back(roi);
  }
  is.close();

  return rois;
}

std::vector<std::vector<float> > load_priors(const std::string& dump)
{
  std::vector<std::vector<float> > priors;
  float x, y, fake;
  std::ifstream is(dump.c_str());
  while(is >> x >> y >> fake >> fake)
  {
    std::vector<float> center;
    center.push_back(x);
    center.push_back(y);
    priors.push_back(center);
  }
  is.close();
  return priors;
}

std::vector<std::vector<float> > get_ftrs(NetP net, const std::string& layer)
{
  const BlobP blob = net->blob_by_name(layer);
  int batch_size = blob->num();
  int dim_features = blob->count() / batch_size;
  std::vector<std::vector<float> > ftrs_vec;

  const float* raw_ftrs;
  for(int i = 0; i < batch_size; i++)
  {
    raw_ftrs = blob->cpu_data() + blob->offset(i);
    std::vector<float> ftrs(raw_ftrs, raw_ftrs+dim_features);
    LOG(ERROR) << layer << " feature demension is " << ftrs.size();
    ftrs_vec.push_back(ftrs);
  }
  return ftrs_vec;
}


int feature_extraction_pipeline(int argc, char** argv) {
  //::google::InitGoogleLogging(argv[0]);

  bool multibox = true;
  const std::string picture="/home/zoellick/Изображения/getImage.jpg"; //hc

  CHECK_EQ(argc, 6) << "Arguments: prototxt, model, score_layer, window_layer, [rois|priors]";

  const std::string prototxt(argv[1]);
  const std::string model(argv[2]);
  const std::string score_l(argv[3]);
  const std::string window_l(argv[4]);
  const std::string aux(argv[5]);

  const int device_id = 0;

  cv::Mat im = cv::imread(picture);
  CHECK(im.data) << "Image loading failed";

  shared_ptr<Net<float> > net(new Net<float>(prototxt, caffe::TEST));
  net->CopyTrainedLayersFrom(model);
  Caffe::set_mode(Caffe::GPU);
  Caffe::SetDevice(device_id);

  CHECK(net->has_blob(score_l))
      << "Unknown feature blob name " << score_l
      << " in the network " << prototxt;

  CHECK(net->has_blob(window_l))
      << "Unknown feature blob name " << window_l
      << " in the network " << prototxt;

  if(multibox)
      set_up_multibox(net, im);
  else 
  {
      std::vector<std::vector<int> > rois = load_rois(aux);
      set_up_frcnn(net, im, rois);
  }
  std::vector<Blob<float>* > results = net->Forward();
        
  std::vector<std::vector<float> > scores = get_ftrs(net, score_l);
  std::vector<std::vector<float> > windows = get_ftrs(net, window_l);

  if(multibox)
  {
    std::vector<std::vector<float> > priors = load_priors(aux);
    windows = lp2rois(windows, priors);
    std::vector<std::vector<int> > o_rois = scale_rois(windows, im.cols, im.rows);
    dump_ftrs(o_rois, "rois.txt");
  }
  else
    dump_ftrs(windows, "bboxes.txt");
  
  dump_ftrs(scores, "scores.txt");

  LOG(ERROR)<< "Successfully extracted the features!";
  return 0;
}

