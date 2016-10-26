#include <vector>

#include "caffe/layers/memory_roi_layer.hpp"

namespace caffe {

template <typename Dtype>
void MemoryROILayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {
  batch_size_ = this->layer_param_.memory_roi_param().batch_size();
  added_rois_.Reshape(batch_size, 5);
  rois_ = nullptr;
  added_rois_.cpu_data();
}

template <typename Dtype>
void MemoryROILayer<Dtype>::AddROIsWithLevels(const std::vector<int>& levels,
      const vector<vector<int> >& rois) {
  num_rois_ = rois.size();
  const size_t num_lvls = levels.size();

  CHECK_GT(num_rois_,0) << "There are no rois passed into the Net";
  CHECK_EQ(num_rois_, num_lvls) << "Number of rois and levels must be the same";

  added_rois_.Reshape(num_rois_, 5);

  rois_ = added_rois_.mutable_cpu_data();
  //copy levels and rois into an array raw by raw: [level, x1, y1, x2, y2]
  for(int r_id = 0; r_id < rois.size(); ++r_id) {
    rois_[r_id*5] = static_cast<Dtype>(levels[r_id]);
    rois_[r_id*5+1] = static_cast<Dtype>(rois[r_id][0]);
    rois_[r_id*5+2] = static_cast<Dtype>(rois[r_id][1]);
    rois_[r_id*5+3] = static_cast<Dtype>(rois[r_id][2]+rois[r_id][0]);
    rois_[r_id*5+4] = static_cast<Dtype>(rois[r_id][3]+rois[r_id][1]);
  }
}

template <typename Dtype>
void MemoryROILayer<Dtype>::AddROIsSingleLevel(const vector<vector<int> >& rois) {
  vector<int> levels(rois.size(), 0);
  this->AddROIsWithLevels(levels, rois);
}

template <typename Dtype>
void MemoryROILayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(rois_) << "Rois have to be set first by calling AddROIs...";
  top[0]->Reshape(num_rois_, 5);
  top[0]->set_cpu_data(rois_);
}

INSTANTIATE_CLASS(MemoryROILayer);
REGISTER_LAYER_CLASS(MemoryROI);

} // namespace caffe
