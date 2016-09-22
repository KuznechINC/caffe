#ifndef MULTIBOX_CONFIDENCE_LOSS_LAYER_HPP
#define MULTIBOX_CONFIDENCE_LOSS_LAYER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe
{

/**
 * @brief Computes confidence loss for MultiBox detection task.
 *
 * At test time, this layer can be omitted. You can use outputs of previous inner product layer with added sigmoid nonlinearities as confidences of proposals
 *
 */

template <typename Dtype> class SigmoidLayer;

    template <typename Dtype>
class MultiBoxConfidenceLossLayer : public LossLayer<Dtype> {
 public:
  explicit MultiBoxConfidenceLossLayer(const LayerParameter& param) : LossLayer<Dtype>(param), sigmoid_layer_(new SigmoidLayer<Dtype>(param)), sigmoid_output_(new Blob<Dtype>()) {}
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "MultiBoxConfidenceLoss"; }

 protected:
  /// @copydoc MultiBoxConfidenceLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the MultiBox confidence loss error gradient w.r.t. the confidence predictions.
   *
   ***********
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  //Blob<Dtype> targets_;
  Blob<Dtype> diff_;
  /// The internal SigmoidLayer used to map predictions to probabilities.
  shared_ptr<SigmoidLayer<Dtype> > sigmoid_layer_;
  /// sigmoid_output stores the output of the SigmoidLayer.
  shared_ptr<Blob<Dtype> > sigmoid_output_;
  /// bottom vector holder to call the underlying SigmoidLayer::Forward
  vector<Blob<Dtype>*> sigmoid_bottom_vec_;
  /// top vector holder to call the underlying SigmoidLayer::Forward
  vector<Blob<Dtype>*> sigmoid_top_vec_;
};
}
#endif