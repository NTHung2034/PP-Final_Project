#ifndef SVM_H
#define SVM_H

#include <vector>
#include <string>
#include <cstdint>

class SVMClassifier {
public:
    SVMClassifier();
    ~SVMClassifier();
    
    void train(const std::vector<float>& features,
               const std::vector<uint8_t>& labels,
               int num_samples,
               int feature_dim);
    
    void predict(const std::vector<float>& features,
                 int num_samples,
                 int feature_dim,
                 std::vector<uint8_t>& predictions);
    
    float evaluate(const std::vector<uint8_t>& predictions,
                   const std::vector<uint8_t>& ground_truth);
    
    void compute_confusion_matrix(const std::vector<uint8_t>& predictions,
                                  const std::vector<uint8_t>& ground_truth,
                                  int num_classes,
                                  std::vector<std::vector<int>>& confusion_matrix);
    
    bool save_model(const std::string& filepath);
    bool load_model(const std::string& filepath);
    
private:
    void* svm_model_;
    void* svm_params_;
};

#endif // SVM_H