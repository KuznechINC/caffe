#include <opencv2/core/core.hpp>

#include <boost/algorithm/string.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/multibox_image_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
MultiBoxImageDataLayer<Dtype>::MultiBoxImageDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      transform_param_(param.transform_param()) {
}

template <typename Dtype>
MultiBoxImageDataLayer<Dtype>::~MultiBoxImageDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void MultiBoxImageDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  data_transformer_.reset(new DataTransformer<Dtype>(transform_param_, this->phase_));
  data_transformer_->InitRand();
  const int new_height = this->layer_param_.multibox_image_data_param().new_height();
  const int new_width  = this->layer_param_.multibox_image_data_param().new_width();
  const bool is_color  = this->layer_param_.multibox_image_data_param().is_color();
  string root_folder = this->layer_param_.multibox_image_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and groundtruths
  const string& source = this->layer_param_.multibox_image_data_param().source();
  //LOG(INFO) << "Opening file with training data " << source;
  std::ifstream infile(source.c_str());
  string filename;
  string groundtruth;
  string tmpline;
  //while (infile >> filename >> groundtruth) {
  //  lines_.push_back(std::make_pair(filename, groundtruth));
  while(!infile.eof()) {
    getline(infile, tmpline, '\n');
    int j = tmpline.find(' ');
    filename = tmpline.substr(0, j);
    groundtruth = tmpline.substr(j+1,tmpline.length());

    std::vector<std::string> windowstrings;
    vector<vector<Dtype> > windowsvec;
    boost::split(windowstrings, groundtruth, boost::is_any_of(";"), boost::token_compress_on);
    //windowsvec.resize(windowstrings.size());
    for (int wnd = 0; wnd<windowstrings.size()-1; wnd++)
    {
      std::vector<std::string> coordstrings;
      vector<Dtype> coordsvec;
      //coordsvec.resize(4);
      boost::split(coordstrings, windowstrings[wnd], boost::is_any_of(" "), boost::token_compress_on);
      for (int coord=0; coord<coordstrings.size();coord++)
        coordsvec.push_back(Dtype(atof(coordstrings[coord].c_str())));
      windowsvec.push_back(coordsvec);
    }

    if (filename!="")
      lines_.push_back(std::make_pair(filename, windowsvec));
  }

// TEST IMAGES:

//  for (int liness_id_ = 333470; liness_id_ < lines_.size(); ++liness_id_) {
//    CHECK_GT(lines_.size(), liness_id_);
 
    // get img_data
//    LOG(INFO) << "Loading " << lines_[liness_id_].first << " " << liness_id_ << " " << lines_.size();
//    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[liness_id_].first, new_height, new_width, is_color);
//    CHECK(cv_img.data) << "Could not load " << lines_[liness_id_].first << " " << liness_id_;
//  }
// TEST IMAGES

  this->prior_boxes_.clear();
  const string& priors_file_name = this->layer_param_.multibox_image_data_param().priors_file();
  //LOG(INFO) << "Opening file with prior boxes " << priors_file_name;
  std::ifstream priors_infile(priors_file_name.c_str());
  string xc;
  string yc;
  string ceps_w;
  string ceps_h;
  vector<Dtype> prior_coords;
//  prior_coords.resize(4);
  prior_coords.resize(5);
  while (priors_infile >> xc >> yc >> ceps_w >> ceps_h) {
    prior_coords[0] = Dtype(atof(xc.c_str()));
    prior_coords[1] = Dtype(atof(yc.c_str()));
    prior_coords[2] = Dtype(atof(ceps_w.c_str()));
    prior_coords[3] = Dtype(atof(ceps_h.c_str()));
    prior_coords[4] = Dtype(0);//delete it
    this->prior_boxes_.push_back(prior_coords);
  }

  if (this->layer_param_.multibox_image_data_param().shuffle()) {
    // randomly shuffle data
    //LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  //LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.multibox_image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.multibox_image_data_param().rand_skip();
    //LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first, new_height, new_width, is_color);
  const int channels = cv_img.channels();
  const int height = cv_img.rows;
  const int width = cv_img.cols;
  // image
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const int batch_size = this->layer_param_.multibox_image_data_param().batch_size();
  if (crop_size > 0) {
    top[0]->Reshape(batch_size, channels, crop_size, crop_size);
    this->prefetch_img_data_.Reshape(batch_size, channels, crop_size, crop_size);
    this->transformed_img_data_.Reshape(1, channels, crop_size, crop_size);
  } else {
    top[0]->Reshape(batch_size, channels, height, width);
    this->prefetch_img_data_.Reshape(batch_size, channels, height, width);
    this->transformed_img_data_.Reshape(1, channels, height, width);
  }
  //LOG(INFO) << "output image data size: " << top[0]->num() << "," << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();

  // groundtruth
  const int shape_dim = this->layer_param_.multibox_image_data_param().shape_dim(); // shape dimension. For current MultiBox it's c - number of channels, h - height, w - width, b - bias, so shape_dim = 4

  //Datum datum;
  //datum.ParseFromString(lines_[lines_id_].second);

  // reshaping groundtruth data blob. We don't know yet its future size, so for now its 1,1,1,1. We'll reshape it again in the future
  top[1]->Reshape(1,1,1,1);
  (this->prefetch_gt_data_).Reshape(1,1,1,1);
  // reshaping groundtruth shape blob
  top[2]->Reshape(batch_size, shape_dim, 1, 1);
  (this->prefetch_gt_shape_).Reshape(batch_size, shape_dim, 1, 1); // (n,c,h,w), c = channels for each of n data points = channel number, height, width

  this->CreatePrefetchThread();
}

template <typename Dtype>
void MultiBoxImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void MultiBoxImageDataLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(this->prefetch_img_data_.count());
  CHECK(this->prefetch_gt_data_.count());
  CHECK(this->prefetch_gt_shape_.count());
  CHECK(this->transformed_img_data_.count());
  MultiBoxImageDataParameter multibox_image_data_param = this->layer_param_.multibox_image_data_param();
  const int batch_size = multibox_image_data_param.batch_size();
  const int new_height = multibox_image_data_param.new_height();
  const int new_width = multibox_image_data_param.new_width();
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const bool is_color = multibox_image_data_param.is_color();
  string root_folder = multibox_image_data_param.root_folder();
  const int shape_dim = multibox_image_data_param.shape_dim();
  const float threshold = multibox_image_data_param.threshold();
  vector<Dtype> temp_data;
  vector<vector<Dtype> > prior_boxes = this->prior_boxes_;
  Dtype ceps_c = Dtype(0.2);
  Dtype ceps_eps = Dtype(0.01);

  // Reshape on single input batches for inputs of varying dimension.
  if (batch_size == 1 && crop_size == 0 && new_height == 0 && new_width == 0) {
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first, 0, 0, is_color);
    this->prefetch_img_data_.Reshape(1, cv_img.channels(), cv_img.rows, cv_img.cols);
    this->transformed_img_data_.Reshape(1, cv_img.channels(), cv_img.rows, cv_img.cols);
  }

  Dtype* prefetch_img_data = this->prefetch_img_data_.mutable_cpu_data();
  Dtype* prefetch_gt_data = this->prefetch_gt_data_.mutable_cpu_data();
  Dtype* prefetch_gt_shape = this->prefetch_gt_shape_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  int b = 0;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
 
    // get img_data
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first, new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first << " " << lines_id_;
    read_time += timer.MicroSeconds();

    // get gt_data
    //Datum datum;
    //datum.ParseFromString(lines_[lines_id_].second);
    vector<vector<Dtype> > windowdata = lines_[lines_id_].second;



    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = this->prefetch_img_data_.offset(item_id);
    this->transformed_img_data_.set_cpu_data(prefetch_img_data + offset);
    vector<int> transformation;
    //transformation.resize(5);

    transformation = this->data_transformer_->TransformRandomScale(cv_img, &(this->transformed_img_data_));
    //transformation.push_back(1);
    //transformation.push_back(2);
    //transformation.push_back(3);
    //transformation.push_back(4);
    //transformation.push_back(5);
    trans_time += timer.MicroSeconds();

    Dtype scaled_width = Dtype(transformation[0]);
    Dtype scaled_height = Dtype(transformation[1]);
    Dtype scale_w = Dtype(crop_size)/scaled_width; // scale of crop wrt to random scaled image
    Dtype scale_h = Dtype(crop_size)/scaled_height;
//    Dtype scale_w = scaled_width/Dtype(new_width);
//    Dtype scale_h = scaled_height/Dtype(new_height);
    Dtype w_off = Dtype(transformation[2]);
    Dtype h_off = Dtype(transformation[3]);
    bool do_mirror = bool(transformation[4]);

    // Now we'll insert loaded datum into top_gt_data, correct it's shape and save it in top_gt_shape:
    int c = 4;
    int h = 1;
    int initw = windowdata.size();
    int w = initw;
    //int c = datum.channels();   // number of "channels" in window
    //int h = datum.height();     // "height" (=1)
    //int w = datum.width();      // width aka number of windows in image

    //LOG(INFO) << "Im:" << item_id << " " << lines_[lines_id_].first << " width:" << new_width << " height:" << new_height << " scaled_width:" << scaled_width << " scaled_height:" << scaled_height << " abs crop coords wrt scaled image:" << w_off + crop_size/2 << " " << h_off + crop_size/2 << " " << crop_size << " " << crop_size << " rel crop coords wrt scaled image:" << (w_off + crop_size/2)/scaled_width << " " << (h_off + crop_size/2)/scaled_height << " " << scale_w << " " << scale_h << " initial number of windows:" << initw;
// somewhere from here --
    for (int wnd = 0; wnd < initw; wnd++) {
      // This is current groundtruth window wrt to full image:
      Dtype wnd_xc = windowdata[wnd][0];
      Dtype wnd_yc = windowdata[wnd][1];
      Dtype wnd_w = windowdata[wnd][2];
      Dtype wnd_h = windowdata[wnd][3];

//      Dtype wnd_xc = Dtype(datum.float_data(wnd*c + 0));
//      Dtype wnd_yc = Dtype(datum.float_data(wnd*c + 1));
//      Dtype wnd_w = Dtype(datum.float_data(wnd*c + 2));
//      Dtype wnd_h = Dtype(datum.float_data(wnd*c + 3));
      // This is current groundtruth window wrt to crop:
      Dtype wnd_xc_c = wnd_xc-(w_off/scaled_width);
      Dtype wnd_yc_c = wnd_yc-(h_off/scaled_height);
      Dtype wnd_w_c = (1/scale_w)*wnd_w;
      Dtype wnd_h_c = (1/scale_h)*wnd_h;
      // This is left and right, top and bottom values:
      Dtype wnd_xl_c = wnd_xc_c-(wnd_w_c/2);
      Dtype wnd_xr_c = wnd_xc_c+(wnd_w_c/2);
      Dtype wnd_yt_c = wnd_yc_c-(wnd_h_c/2);
      Dtype wnd_yb_c = wnd_yc_c+(wnd_h_c/2);
      // Cut the values with crop borders:
      Dtype wnd_xl_cut = wnd_xl_c;
      Dtype wnd_xr_cut = wnd_xr_c;
      Dtype wnd_yt_cut = wnd_yt_c;
      Dtype wnd_yb_cut = wnd_yb_c;
      if (wnd_xl_cut<0)
        wnd_xl_cut=0;
      if (wnd_xl_cut>1)
        wnd_xl_cut=1;
      if (wnd_xr_cut<0)
        wnd_xr_cut=0;
      if (wnd_xr_cut>1)
        wnd_xr_cut=1;
      if (wnd_yt_cut<0)
        wnd_yt_cut=0;
      if (wnd_yt_cut>1)
        wnd_yt_cut=1;
      if (wnd_yb_cut<0)
        wnd_yb_cut=0;
      if (wnd_yb_cut>1)
        wnd_yb_cut=1;
      Dtype wnd_xc_cut = (wnd_xr_cut - wnd_xl_cut)/2 + wnd_xl_cut;
      Dtype wnd_yc_cut = (wnd_yb_cut - wnd_yt_cut)/2 + wnd_yt_cut;
      Dtype wnd_w_cut = wnd_xr_cut-wnd_xl_cut;
      Dtype wnd_h_cut = wnd_yb_cut-wnd_yt_cut;
      // Use threshold to decide do we keep this window or it's too mispresent in current crop?
      Dtype cropped_area = wnd_w_cut*wnd_h_cut;
      Dtype full_area = wnd_w_c*wnd_h_c;
      if (full_area<0.0001)
        full_area = 0.0001;
      int wnd_priorID = 0;
      if ((cropped_area/full_area)>threshold) {
        // mirror:
        if (do_mirror)
        {
          wnd_xc_cut = 1-wnd_xc_cut;        
        }
        Dtype min_distance = Dtype(FLT_MAX);
        // Match to priors:
        for (int pr = 0; pr< prior_boxes.size(); pr++)
        {
          //Dtype distance = Dtype(sqrt((wnd_xc_cut-prior_boxes[pr][0])*(wnd_xc_cut-prior_boxes[pr][0])+(wnd_yc_cut-prior_boxes[pr][1])*(wnd_yc_cut-prior_boxes[pr][1])));
          Dtype distance = Dtype(sqrt((wnd_xc_cut-prior_boxes[pr][0])*(wnd_xc_cut-prior_boxes[pr][0])+(wnd_yc_cut-prior_boxes[pr][1])*(wnd_yc_cut-prior_boxes[pr][1])+( ceps_c/(ceps_eps+wnd_w_cut) - prior_boxes[pr][2] )*(ceps_c/(ceps_eps+wnd_w_cut) - prior_boxes[pr][2])+(ceps_c/(ceps_eps+wnd_h_cut) - prior_boxes[pr][3])*(ceps_c/(ceps_eps+wnd_h_cut) - prior_boxes[pr][3])));
          if (distance<min_distance)
          {
            wnd_priorID = pr;
            min_distance = distance;
          }
        }
        temp_data.push_back(wnd_xc_cut - prior_boxes[wnd_priorID][0]);// wrt to prior
        temp_data.push_back(wnd_yc_cut - prior_boxes[wnd_priorID][1]);// wrt to prior
        temp_data.push_back(wnd_w_cut);
        temp_data.push_back(wnd_h_cut);
        temp_data.push_back(Dtype(wnd_priorID));
        this->prior_boxes_[wnd_priorID][4]=this->prior_boxes_[wnd_priorID][4]+1;
        //DLOG(INFO) << "temp_data:" << wnd_xc_cut - prior_boxes[wnd_priorID][0] << " " << wnd_yc_cut - prior_boxes[wnd_priorID][1] << " " << wnd_w_cut << " " << wnd_h_cut << " " << Dtype(wnd_priorID);
      }
      else {
        // don't use this window
        w=w-1;
      }
      //LOG(INFO) << "Wnd:" << wnd << " abs coords:" << wnd_xc << " " << wnd_yc << " " << wnd_w << " " << wnd_h << " wrt crop coords:" << wnd_xc_c << " " << wnd_yc_c << " " << wnd_w_c << " " << wnd_h_c << " do mirror:" << do_mirror << " cut crop coords (mirrored or not):" << wnd_xc_cut << " " << wnd_yc_cut << " " << wnd_w_cut << " " << wnd_h_cut << " cropped_area:" << cropped_area << " full_area:" << full_area << " cropped/full:" << cropped_area/full_area;
      //if ((cropped_area/full_area)>threshold)
        //LOG(INFO) << "Good window. Matched priorID:" << wnd_priorID << " xc wrt to prior:" << (wnd_xc_cut - prior_boxes[wnd_priorID][0]) << " yc wrt to prior:" << (wnd_yc_cut - prior_boxes[wnd_priorID][1]);
      //else
        //LOG(INFO) << "Not good window. Too small part of it got into crop. Skip it";      
    }
    int datum_size = (c+1) * h * w; // bias in top blob for this image
    prefetch_gt_shape[item_id * shape_dim + 0] = c+1;   // number of "channels" in window plus priorID
    prefetch_gt_shape[item_id * shape_dim + 1] = h;   // "height" (=1)
    prefetch_gt_shape[item_id * shape_dim + 2] = w;   // width aka number of windows in image
    prefetch_gt_shape[item_id * shape_dim + 3] = b;   // bias in top blob for this image
    b += datum_size;
    //LOG(INFO) << "Resulted number of windows:" << w << " datum_size:" << datum_size << " b:" << b;
    //for (int ik=0; ik<200; ik++)
      //LOG(INFO) << ik << " " << this->prior_boxes_[ik][4] << "|"<< ik+200 << " " << this->prior_boxes_[ik+200][4] << "|"<< ik+400 << " " << this->prior_boxes_[ik+400][4] << "|"<< ik+600 << " " << this->prior_boxes_[ik+600][4];
// -- to here we need to adjust bboxes wrt to what happened in Transformer and also match priors

    //if (datum_size>0)
    //  (this->prefetch_gt_data_).Reshape(1, b + datum_size, 1, 1);


    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      //LOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.multibox_image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
 // DLOG(INFO) << "We are here, temp_data.size()=" << temp_data.size();
  if (temp_data.size()>0)
    {
    //DLOG(INFO) << "NOT SINGLE! temp_data.size():" << temp_data.size();
      (this->prefetch_gt_data_).Reshape(1, temp_data.size(), 1, 1);
      prefetch_gt_data = this->prefetch_gt_data_.mutable_cpu_data();
    //DLOG(INFO) << "NOT SINGLE! (this->prefetch_gt_data_).count():" << (this->prefetch_gt_data_).count();      
      for (int i=0; i<temp_data.size(); i++)
      {
        //DLOG(INFO) << "temp_data[i]:" << temp_data[i];
        prefetch_gt_data[i]=temp_data[i];
        //DLOG(INFO) << "prefetch_gt_data[i]:" << prefetch_gt_data[i];
      }
    //DLOG(INFO) << "NOT SINGLE!";
    }
  else {
    //DLOG(INFO) << "SINGLE!";
    (this->prefetch_gt_data_).Reshape(1, 1, 1, 1);
    prefetch_gt_data = this->prefetch_gt_data_.mutable_cpu_data();
    //DLOG(INFO) << "SINGLE!";
  }
  batch_timer.Stop();
  //DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  //DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  //DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
  //DLOG(INFO) << "prefetch_gt_data:" << (this->prefetch_gt_data_).count();
  //for (int i=0; i<(this->prefetch_gt_data_).count(); i++)
    //DLOG(INFO) << (this->prefetch_gt_data_).cpu_data()[i];
}

// <NEW>
template <typename Dtype>
void MultiBoxImageDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  this->JoinPrefetchThread();
  // Reshape top groundtruth blob:
  top[1]->Reshape(1, this->prefetch_gt_data_.channels(), 1, 1);
  // Copy the data
  caffe_copy(this->prefetch_img_data_.count(), this->prefetch_img_data_.cpu_data(), top[0]->mutable_cpu_data());
  caffe_copy(this->prefetch_gt_data_.count(), this->prefetch_gt_data_.cpu_data(), top[1]->mutable_cpu_data());
  caffe_copy(this->prefetch_gt_shape_.count(), this->prefetch_gt_shape_.cpu_data(), top[2]->mutable_cpu_data());

//DLOG(INFO) << "top[1]->cpu_data()[i]";
//for (int i=0; i<this->prefetch_gt_data_.count(); i++)
  //DLOG(INFO) << top[1]->cpu_data()[i];

//DLOG(INFO) << "top[2]->cpu_data()[i]";
//for (int i=0; i<this->prefetch_gt_shape_.count(); i++)
  //DLOG(INFO) << top[2]->cpu_data()[i];


  // Start a new prefetch thread
  this->CreatePrefetchThread();
}

template <typename Dtype>
void MultiBoxImageDataLayer<Dtype>::CreatePrefetchThread() {
//DLOG(INFO) << "here!";
  this->data_transformer_->InitRand();
//DLOG(INFO) << "here 2!";
  StartInternalThread();
}

template <typename Dtype>
void MultiBoxImageDataLayer<Dtype>::JoinPrefetchThread() {
  StopInternalThread();
}

INSTANTIATE_CLASS(MultiBoxImageDataLayer);
REGISTER_LAYER_CLASS(MultiBoxImageData);

}  // namespace caffe
