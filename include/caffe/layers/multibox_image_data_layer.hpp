#ifndef MULTIBOX_IMAGE_DATA_LAYER_HPP
#define MULTIBOX_IMAGE_DATA_LAYER_HPP
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe
{
/**
 * @brief Provides data to the Net from image files for MultiBox training.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class MultiBoxImageDataLayer : public InternalThread , public Layer<Dtype> {
 public:
  explicit MultiBoxImageDataLayer(const LayerParameter& param);// : Layer<Dtype>(param) {}
  virtual ~MultiBoxImageDataLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MultiBoxImageData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 3; }

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {}

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

  virtual void CreatePrefetchThread();
  virtual void JoinPrefetchThread();
  virtual void InternalThreadEntry();

  Blob<Dtype> prefetch_img_data_;
  Blob<Dtype> prefetch_gt_data_;
  Blob<Dtype> prefetch_gt_shape_;
  Blob<Dtype> transformed_img_data_;

  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  vector<vector<Dtype> > prior_boxes_;
  vector<std::pair<std::string, vector<vector<Dtype> > > > lines_;
  int lines_id_;

  TransformationParameter transform_param_;
  shared_ptr<DataTransformer<Dtype> > data_transformer_;
};
}
#endif