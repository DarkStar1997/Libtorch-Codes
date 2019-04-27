#include <iostream>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <sstream>

int main()
{
    auto trainset = torch::data::datasets::MNIST("/home/rohan/PyTorch_tests/data")
            .map(torch::data::transforms::Stack<>());
    auto trainloader = torch::data::make_data_loader(std::move(trainset), torch::data::DataLoaderOptions().batch_size(64)
                                                     .workers(8).enforce_ordering(false));
    int count = 0;
    for(torch::data::Example<> &batch : *trainloader)
    {
        for(int i = 0; i < batch.data.size(0); i++)
        {
            std::cout << "Picture count: " << ++count << '\n';
            std::stringstream label;
            auto image = batch.data[i];
            std::cout << "Size of the image:\n" << image.sizes() << '\n';
            image.resize_({image.size(1), image.size(2)});
            label << "Label: " << batch.target[i].item<int>();
            cv::Mat img(cv::Size(image.size(0), image.size(1)), CV_BGR2RGB, image.data_ptr());
            cv::imshow(label.str(), img);
            cv::waitKey(1000);
        }
    }
}
