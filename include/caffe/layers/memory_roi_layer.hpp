#ifndef MEMORY_ROI_LAYER_HPP_
#define MEMORY_ROI_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_data_layer.hpp"

namespace caffe
{
/**
 * @brief Provides rois to the ROIPoolingLayer from memory.
 */
template <class Dtype>
class MemoryROILayer: public BaseDataLayer<Dtype> {
 public:
  explicit MemoryROILayer(const LayerParameter& param)
      : BaseDataLayer<Dtype>(param) {}
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MemoryROI"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  virtual void AddROIsWithLevels( const std::vector<int>& levels,
          const vector<vector<int> >& rois);
  virtual void AddROIsSingleLevel(const vector<vector<int> >& rois);

  size_t num_rois() { return num_rois_; }
  size_t batch_size() {return batch_size_;}

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  size_t num_rois_;
  size_t batch_size_;
  Dtype* rois_;
  Blob<Dtype> added_rois_;
};
}
#endif //CAFFE_MEMORY_ROI_LAYER_H
