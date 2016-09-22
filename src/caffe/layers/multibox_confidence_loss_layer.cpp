#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/multibox_confidence_loss_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
void MultiBoxConfidenceLossLayer<Dtype>::LayerSetUp(
     const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void MultiBoxConfidenceLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //LossLayer<Dtype>::Reshape(bottom, top);
  top[0]->Reshape(1, 1, 1, 1);
  //targets_.ReshapeLike(*bottom[0]);
  diff_.ReshapeLike(*bottom[0]);
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void MultiBoxConfidenceLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // bottom[0] = batch_size * Nw values - batch_size (=number of images in batch) outputs of proposed Nw (=number of windows in image) window confidences. batch_size = 128, Nw= 800, so total number is 800*128
  // bottom[1] = Shaped label data for windows: 1d in order: batch:{images{windows{window "channels" - xc,yc,w,h,priorid,classid}}}
  // bottom[2] = Shape of shaped label data: 2d: row = image in batch, cols:{c,h,w,b}, c = number of window "channels", h = 1 (MultiBox zhe), w = number of windows, b = bias = from which index starts this image in 1d-vector

  // gt = ground truth, pr = predicted
  
  int batch_size = bottom[0]->num(); // batch size
  int pred_size = bottom[0]->channels(); // size of predicted data for one image (example: 3200 , num_pred_windows(=800) * pred_dim(=4))

  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);

  //LOG(INFO) << "Confidence Loss bottom[0] (first 20 values) and it's sigmoided version:";
  //for (int i=0; i<batch_size*pred_size;i++)
  //for (int i=0; i<20;i++)
  //{
  //  LOG(INFO) <<  ((float)bottom[0]->cpu_data()[i]) << " " << ((float)sigmoid_top_vec_[0]->cpu_data()[i]);
  //}

  //LOG(INFO) << "Confidence Loss bottom[1]:";

  //for (int i=0; i<bottom[1]->channels();i++)
  //{
  //  LOG(INFO) <<  ((float)bottom[1]->cpu_data()[i]);
  //}

  //LOG(INFO) << "Confidence Loss bottom[2]:";

  //for (int i=0; i<bottom[2]->num()*4;i++)
  //{
  //  LOG(INFO) <<  ((float)bottom[2]->cpu_data()[i]);
  //}

  //int pred_dim = 4;          // prediction data dim (xc,yc,w,h = 4)
  int shape_dim = 4;         // shape parameter data dim (c,h,w,b = 4)
  int window_dim = 5;        // window data dim (xc,yc,w,h,priorID,classID = 6)
  Dtype loss = Dtype(0); // location loss
  //Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  //caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  //caffe_set(bottom[0]->count(), Dtype(0), targets_.mutable_cpu_data());
  Dtype beta = Dtype(this->layer_param_.multibox_confidence_loss_param().beta());

  // i = images in batch
  for (int i=0; i<batch_size; i++)
  {
    int Nw = bottom[2]->cpu_data()[i*shape_dim+2]; // Nw = number of gt windows in image i
    int b = bottom[2]->cpu_data()[i*shape_dim+3];  // b = bias = index of i'th image in bottom[1]-vector
    int *img_targets = new int[Nw];
    //LOG(INFO) << "image:" << i;
    for (int wnd_gt=0; wnd_gt<Nw; wnd_gt++)
    {
      //LOG(INFO) << "window ground truth:" << wnd_gt;
      int priorID = (int)(bottom[1]->cpu_data()[b + wnd_gt*window_dim + 4]); // id of the prior, matched with gt window wnd of image i. It's actually tk in formula for bootstrapping loss
      //LOG(INFO) << "priorID:" << priorID;
      img_targets[wnd_gt]=priorID;
    }
    for (int wnd_pr=0; wnd_pr<pred_size; wnd_pr++)
    {
      //Dtype pre_c_pr = bottom[0]->cpu_data()[i*pred_size + wnd_pr];
      Dtype c_pr = sigmoid_top_vec_[0]->cpu_data()[i*pred_size + wnd_pr];
      Dtype target = Dtype(0);
      for (int t=0;t<Nw;t++)
      {
        if (img_targets[t]==wnd_pr)
        {
          target = Dtype(1);
          break;
        }
      }
      Dtype threshold = (c_pr>Dtype(0.5)) ? Dtype(1) : Dtype(0);
      target = (beta*target+(1-beta)*threshold);
      //targets_.mutable_cpu_data()[i*pred_size + wnd_pr]=Dtype(target);
      // bootstrapping:
      //if (target==Dtype(1))
      //LOG(INFO) << "wnd_pr=" << wnd_pr << " c_pr=" << c_pr << " add to loss=" << Dtype((beta*target+(1-beta)*threshold)*log(c_pr)-(beta*(1-target)+(1-beta)*(1-threshold))*log(1-c_pr)) << " dFdc=" << Dtype((beta*target+(1-beta)*threshold)/(c_pr)-(-1)*((beta*(1-target)+(1-beta)*(1-threshold)))/(1-c_pr));
      if (c_pr<0.0000001)
        c_pr=0.0000001;
      if (c_pr>0.9999999)
        c_pr=0.9999999;
      Dtype deltaloss = -Dtype(target*log(c_pr)+(1-target)*(log(1-c_pr))); //no bootstrap MultiBox loss
      //Dtype deltaloss = Dtype(-(beta*target+(1-beta)*threshold)*log(c_pr)-(beta*(1-target)+(1-beta)*(1-threshold))*log(1-c_pr)); //bootstrap MultiBox loss
      loss+=deltaloss; //MultiBox
      Dtype dFdc = ((1-target)/(1-c_pr) - target/c_pr) * (c_pr)*(1-c_pr);
      //Dtype dFdc = -(target-c_pr) * (c_pr)*(1-c_pr);
      //Dtype dFdc = -(target/c_pr - (1-target)/(1-c_pr)) * (c_pr)*(1-c_pr); //no bootstrap Multibox dFdc
      //Dtype dFdc = (target*c_pr);///(c_pr*(1-c_pr));// * (c_pr)*(1-c_pr); //old Multibox dFdc
      //Dtype dFdc = Dtype(-(beta*target+(1-beta)*threshold)/(c_pr) - (-1) * ((beta*(1-target)+(1-beta)*(1-threshold)))/(1-c_pr));// derivative - each prediction has it's own //bootstrap 

      // rmse
      //loss+=Dtype((target-c_pr)*(target-c_pr)*0.5);
      //Dtype dFdc;
      //if (target==Dtype(1))
      //  dFdc = -Dtype(target-c_pr);
      //else
        //dFdc = Dtype(-0.000001);
      //if (target==Dtype(1))

      // cross entropy loss
      //Dtype deltaloss = Dtype(pre_c_pr * (target - (pre_c_pr >= 0)) - log(1 + exp(pre_c_pr - 2 * pre_c_pr * (pre_c_pr >= 0))));
      //loss -= deltaloss;
      
      //if ((wnd_pr<50)||(target>=0.2))
      //  LOG(INFO) << "wnd_pr=" << wnd_pr << " target=" << target << " c_pr=" << c_pr <<" add to loss=" << deltaloss << "  dFdc=" << dFdc;            
      //if (target>0)        
      diff_.mutable_cpu_data()[i*pred_size + wnd_pr] = dFdc;// MultiBox
      //else
      //  diff_.mutable_cpu_data()[i*pred_size + wnd_pr] = 0;// MultiBox
    }
  }
  loss /=batch_size;
  //LOG(INFO) << "loss:" << loss;
  //if ((loss/batch_size)>100000)
  //  LOG(FATAL) << "Suddenly Too much loss";
  //top[0]->mutable_cpu_data()[0] = loss/(batch_size);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void MultiBoxConfidenceLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    // First, compute the diff
//    const int count = bottom[0]->count();
//    const int num = bottom[0]->num();
//    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
//    const Dtype* targets = targets_.cpu_data();
//    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
//    caffe_sub(count, sigmoid_output_data, targets, bottom_diff);
    // Scale down gradient
//    const Dtype loss_weight = top[0]->cpu_diff()[0];
//    caffe_scal(count, loss_weight / num, bottom_diff);

//    LOG(INFO) << "Backprop: Confidence Loss:";
    //for (int i=0; i<batch_size*pred_size;i++)
//    for (int i=0; i<count;i++)
//    {
//      if ((i<50)||(targets[i]>0))
//        LOG(INFO) << "i: " << i << " c_pr:" << ((float)sigmoid_output_data[i]) << " target:" << ((float)targets[i]) << " diff:" << ((float)bottom_diff[i]);
//    }

    // MultiBox:
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
STUB_GPU(MultiBoxConfidenceLossLayer);
#endif

INSTANTIATE_CLASS(MultiBoxConfidenceLossLayer);
REGISTER_LAYER_CLASS(MultiBoxConfidenceLoss);

}  // namespace caffe
