# Skin Cancer Classification Using Neural Networks

Trains a NN on a dataset of 2637 56*56 pixel images of skin cancer for 40 epochs, then tests on a dataset of 663 items. 

Expected output is a log of each epoch and its average cost, and a final output of a correct/total statistic.

Additional output are 
`image.png/image2.png` and `testData.png/testData2.png` : sample images before/after ZCA whitening

Additional output are misclassfied images with the format: `n_label_predicted.png`

`Concert_MNIST` contains Python scripts to construct ubyte images and labels from a dataset of .pngs (not needed to execute the program)

Dependencies: please run these commands before go build:
```
go get github.com/pkg/errors
go get gonum.org/v1/gonum/stat/distuv
go get gorgonia.org/tensor
go get gorgonia.org/tensor/native
go get gorgonia.org/vecf64
```

To run: 
```
go build
./code
```

