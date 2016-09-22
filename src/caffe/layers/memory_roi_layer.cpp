#include <vector>

#include "caffe/layers/memory_roi_layer.hpp"

namespace caffe {

template <typename Dtype>
void MemoryROILayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {
  batch_size_ = this->layer_param_.memory_roi_param().batch_size();
  rois_ = NULL;
}

template <typename Dtype>
void MemoryROILayer<Dtype>::AddROIsWithLevels(const vector<int>& levels, 
      const vector<vector<vector<int> > >& rois_by_lvl) {
  CHECK(!levels.empty()) << "List of levels is empty";
  num_rois_ = 0;
  for(int lvl = 0; lvl < rois_by_lvl.size(); ++lvl) {
    num_rois_ += rois_by_lvl[lvl].size();
  }
  CHECK_GT(num_rois_,0) << "There are no rois passed into the Net";
  CHECK_EQ(levels.size(), rois_by_lvl.size()) << "Rois have to be specified"
      " only for specified levels";
  const int num_data = num_rois_*5;
  rois_ = new Dtype[num_data];
  //copy levels and rois into an array raw by raw: [level, x1, y1, x2, y2]
  int processed = 0;
  for(int lvl_id = 0; lvl_id < levels.size(); ++lvl_id) {
    for(int r_id = 0; r_id < rois_by_lvl[lvl_id].size(); ++r_id) {
      rois_[lvl_id*processed*5+r_id*5] = static_cast<Dtype>(levels[lvl_id]);

      rois_[lvl_id*processed*5+r_id*5+1] = static_cast<Dtype>(
        rois_by_lvl[lvl_id][r_id][0]);

      rois_[lvl_id*processed*5+r_id*5+2] = static_cast<Dtype>(
        rois_by_lvl[lvl_id][r_id][1]);

      rois_[lvl_id*processed*5+r_id*5+3] = static_cast<Dtype>(
        rois_by_lvl[lvl_id][r_id][2]+rois_by_lvl[lvl_id][r_id][0]);

      rois_[lvl_id*processed*5+r_id*5+4] = static_cast<Dtype>(
        rois_by_lvl[lvl_id][r_id][3]+rois_by_lvl[lvl_id][r_id][1]);
    }
    processed += rois_by_lvl[lvl_id].size();
  }
}

template <typename Dtype>
void MemoryROILayer<Dtype>::AddROIsSingleLevel(const vector<vector<int> >& rois) {
  vector<int> levels(1, 0);
  vector<vector<vector<int> > > rois_by_lvl;
  rois_by_lvl.push_back(rois);
  this->AddROIsWithLevels(levels, rois_by_lvl);
}

template <typename Dtype>
void MemoryROILayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(rois_) << "Rois have to be set first by calling AddROIs...";
  vector<int> shape(2);
  shape[0] = num_rois_;
  shape[1] = 5;
  top[0]->Reshape(shape);
  top[0]->set_cpu_data(rois_);
}

INSTANTIATE_CLASS(MemoryROILayer);
REGISTER_LAYER_CLASS(MemoryROI);

} // namespace caffe
