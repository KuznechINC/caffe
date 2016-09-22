#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/multibox_location_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MultiBoxLocationLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //LossLayer<Dtype>::Reshape(bottom, top);
  top[0]->Reshape(1, 1, 1, 1);
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void MultiBoxLocationLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // bottom[0] = batch_size * (Nw*dim) values - batch_size (=number of images in batch) outputs of proposed Nw (=number of predicted windows in image) window centers location coordinates and width/height. batch_size = 128, Nw= 800, pred_dim = 4(i.e. xc,yc,w,h), so total number is 3200*128
  // bottom[1] = Shaped label data for windows: 1d in order: batch:{images{windows{window "channels" - xc,yc,w,h,priorid,classid}}}
  // bottom[2] = Shape of shaped label data: 2d: row = image in batch, cols:{c,h,w,b}, c = number of window "channels", h = 1 (MultiBox zhe), w = number of windows, b = bias = from which index starts this image in 1d-vector

  // gt = ground truth, pr = predicted
  
  int batch_size = bottom[0]->num(); // batch size
  int pred_size = bottom[0]->channels(); // size of predicted data for one image (example: 3200 , num_pred_windows(=800) * pred_dim(=4))
  int pred_dim = 4;          // prediction data dim (xc,yc,w,h = 4)
  int shape_dim = 4;         // shape parameter data dim (c,h,w,b = 4)
  int window_dim = 5;        // window data dim (xc,yc,w,h,priorID,classID = 6)
  Dtype loss = Dtype(0); // location loss
  //Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  //caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  caffe_set(bottom[0]->count(), Dtype(0), diff_.mutable_cpu_data());

  // i = images in batch
  for (int i=0; i<batch_size; i++)
  {
    int Nw = bottom[2]->cpu_data()[i*shape_dim+2]; // Nw = number of gt windows in image i
    int b = bottom[2]->cpu_data()[i*shape_dim+3];  // b = bias = index of i'th image in bottom[1]-vector
    for (int wnd=0; wnd<Nw; wnd++)
    {
      Dtype xc_gt = bottom[1]->cpu_data()[b + wnd*window_dim];            // xc of the gt window wnd of image i
      Dtype yc_gt = bottom[1]->cpu_data()[b + wnd*window_dim + 1];        // yc of the gt window wnd of image i
      Dtype w_gt = bottom[1]->cpu_data()[b + wnd*window_dim + 2];         // width of the gt window wnd of image i
      Dtype h_gt = bottom[1]->cpu_data()[b + wnd*window_dim + 3];         // height of the gt window wnd of image i
      int priorID = (int)(bottom[1]->cpu_data()[b + wnd*window_dim + 4]); // id of the prior, matched with gt window wnd of image i

      // Ok, let's find xc_pr, yc_pr, w_pr, h_pr for priorID (aka predicted window number priorID):

      Dtype xc_pr = bottom[0]->cpu_data()[i*pred_size + priorID*pred_dim];        // xc of the matched predicted window for gt window wnd of image i
      Dtype yc_pr = bottom[0]->cpu_data()[i*pred_size + priorID*pred_dim + 1];    // yc of the matched predicted window for gt window wnd of image i
      Dtype w_pr = bottom[0]->cpu_data()[i*pred_size + priorID*pred_dim + 2];     // width of the matched predicted window for gt window wnd of image i
      Dtype h_pr = bottom[0]->cpu_data()[i*pred_size + priorID*pred_dim + 3];     // height of the matched predicted window for gt window wnd of image i

      // and now we have add it to loss:
      Dtype deltaloss = Dtype(0);

      Dtype dFdl_xc = Dtype(0); // derivative - each prediction has it's own
      Dtype dFdl_yc = Dtype(0); // derivative - each prediction has it's own
      Dtype dFdl_w = Dtype(0); // derivative - each prediction has it's own
      Dtype dFdl_h = Dtype(0); // derivative - each prediction has it's own

      Dtype val = xc_pr-xc_gt;
      if (abs(val) < 1) {
        deltaloss+=0.5*val*val;
        dFdl_xc = val;
      } else {
        deltaloss+=abs(val) - 0.5;
        dFdl_xc = (Dtype(0) < val) - (val < Dtype(0));
      }
      val = yc_pr-yc_gt;
      if (abs(val) < 1) {
        deltaloss+=0.5*val*val;
        dFdl_yc = val;
      } else {
        deltaloss+=(abs(val) - 0.5);
        dFdl_yc = (Dtype(0) < val) - (val < Dtype(0));
      }
      val = w_pr-w_gt;
      if (abs(val) < 1) {
        deltaloss+=0.5*val*val;
        dFdl_w = val;
      } else {
        deltaloss+=(abs(val) - 0.5);
        dFdl_w = (Dtype(0) < val) - (val < Dtype(0));
      }
      val = h_pr-h_gt;
      if (abs(val) < 1) {
        deltaloss+=0.5*val*val;
        dFdl_h = val;
      } else {
        deltaloss+=(abs(val) - 0.5);
        dFdl_h = (Dtype(0) < val) - (val < Dtype(0));
      }
      loss+=deltaloss;
      // and also find derivatives and put them indo diff
      //Dtype dFdl = Dtype((xc_pr-xc_gt)+(yc_pr-yc_gt)+(w_pr-w_gt)+(h_pr-h_gt)); // derivative - each prediction has it's own
      //dFdl /= Dtype(batch_size); //??
      //bottom_diff[i*pred_size + priorID*pred_dim] += dFdl;
      //bottom_diff[i*pred_size + priorID*pred_dim + 1] += dFdl;
      //bottom_diff[i*pred_size + priorID*pred_dim + 2] += dFdl;
      //bottom_diff[i*pred_size + priorID*pred_dim + 3] += dFdl;
      diff_.mutable_cpu_data()[i*pred_size + priorID*pred_dim] = dFdl_xc;
      diff_.mutable_cpu_data()[i*pred_size + priorID*pred_dim + 1] = dFdl_yc;
      diff_.mutable_cpu_data()[i*pred_size + priorID*pred_dim + 2] = dFdl_w;
      diff_.mutable_cpu_data()[i*pred_size + priorID*pred_dim + 3] = dFdl_h;

      //LOG(INFO) << "Location loss: gt window number:" << wnd;
      //LOG(INFO) << "xc, yc, w, h (gt):" << xc_gt << " " << yc_gt << " " << w_gt << " " << h_gt;
      //LOG(INFO) << "xc, yc, w, h (pr):" << xc_pr << " " << yc_pr << " " << w_pr << " " << h_pr;
      //LOG(INFO) << "priorID:" << priorID;
      //LOG(INFO) << "delta loss:" << deltaloss << " dFdl_xc:" << dFdl_xc << " dFdl_yc:" << dFdl_yc << " dFdl_w:" << dFdl_w << " dFdl_h:" << dFdl_h;
    }
  }

  loss /= (Dtype(2)*Dtype(batch_size));
  top[0]->mutable_cpu_data()[0] = loss;
  //LOG(INFO) << "loss:" << loss;
}

template <typename Dtype>
void MultiBoxLocationLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype alpha = top[0]->cpu_diff()[0] / bottom[0]->num();
    caffe_cpu_axpby(
            bottom[0]->count(),              // count
            alpha,                              // alpha
            diff_.cpu_data(),                   // a
            Dtype(0),                           // beta
            bottom[0]->mutable_cpu_diff());  // b
  }
}

#ifdef CPU_ONLY
STUB_GPU(MultiBoxLocationLossLayer);
#endif

INSTANTIATE_CLASS(MultiBoxLocationLossLayer);
REGISTER_LAYER_CLASS(MultiBoxLocationLoss);

}  // namespace caffe
