#ifndef MULTIBOX_LOCATION_LOSS_LAYER_HPP
#define MULTIBOX_LOCATION_LOSS_LAYER_HPP

#include "caffe/layers/loss_layer.hpp"

namespace caffe
{
/**
 * @brief Computes location loss for MultiBox detection task.
 *
 * At test time, this layer can be omitted. You can use outputs of previous inner product layer as coordinates of proposal
 *
 */
template <typename Dtype>
class MultiBoxLocationLossLayer : public LossLayer<Dtype> {
 public:
  explicit MultiBoxLocationLossLayer(const LayerParameter& param) : LossLayer<Dtype>(param) {}
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "MultiBoxLocationLoss"; }

 protected:
  /// @copydoc MultiBoxLocationLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  /**
   * @brief Computes the MultiBox location loss error gradient w.r.t. the location predictions.
   *
   ***********
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_;
};
}
#endif