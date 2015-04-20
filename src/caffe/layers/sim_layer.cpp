#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SimLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
}

template <typename Dtype>
void SimLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Produces only one value
  top[0]->Reshape(bottom[0]->num(), 1, 1, 1);
  
}

template <typename Dtype>
void SimLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* label_1 = bottom[0]->cpu_data();
  const Dtype* label_2 = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int num_1 = bottom[0]->num();
  int dim_1 = bottom[0]->count() / bottom[0]->num();
  int num_2 = bottom[0]->num();
  int dim_2 = bottom[0]->count() / bottom[0]->num();
  CHECK_EQ(num_1, num_2)
          << "Batch size must be the same (" << num_1 <<" "<< num_2<< ")";
          
  CHECK_EQ(dim_1, dim_2)
          << "Dimmension of vector must be the same (" << dim_1 <<" "<< dim_2<< ")";
  std::vector<int> compare_value(num_1,0);
  for (int i = 0; i < num_1; ++i)
  {
    if (label_1[i*dim_1] == label_2[i*dim_1])
      compare_value[i] = 1;
    
   // std::cout << "params after rebalance:" << std::endl;
   // std::cout << label_1[i*dim_1] << " "<<label_2[i*dim_1] <<" " <<compare_value[i] << std::endl; ;
 }
  
  for (int i = 0; i < num_1; ++i) {
      top_data[top[0]->offset(i, 0, 0)] = compare_value[i];
  }
  
  //degug
//   int s/*um_false = 0, sum_true = 0;
//   for (int i = 0; i < num_1; ++i) {
//     if (compare_value[i] == 1)
//       sum_true++;
//     else
//       sum_false++;
//   }
//   std::cout <*/< "Batch Statistic: "<< " True: "<< sum_true << " False: "<< sum_false<< std::endl;
   

  
}

INSTANTIATE_CLASS(SimLayer);
REGISTER_LAYER_CLASS(Sim);

}  // namespace caffe
