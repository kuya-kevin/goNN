package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"time"

	"gorgonia.org/tensor"
	"gorgonia.org/tensor/native"
)

func avg(a []float64) (retVal float64) {
	s := sum(a)
	return s / float64(len(a))
}

func sum(a []float64) (retVal float64) {
	for i := range a {
		retVal += a[i]
	}
	return retVal
}

func shuffleX(a [][]float64) {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	tmp := make([]float64, len(a[0]))
	for i := range a {
		j := r.Intn(i + 1)
		copy(tmp, a[i])
		copy(a[i], a[j])
		copy(a[j], tmp)
	}
}

func argmax(a []float64) (retVal int) {
	var max = math.Inf(-1)
	for i := range a {
		if a[i] > max {
			retVal = i
			max = a[i]
		}
	}
	return
}

//main steps:
//load image files, generate []label
//convert images + labels into *tensor.Dense
//visualize 100 of the images
//perform ZCA whitening
//visualize the whitened images
//create the neural network, 100 unit hidden layer
//create a slice of the costs (keep track of average cost over time)
//within each epoch, slice input into single image slices
//within each epoch, slice output labels into single slices
//within each epoch, call nn.Train() w/ a learn rate of 0.1, use the sliced single image + single labels as a training example
//train for 40 epochs
func nn_driver() {

	imgs, err := readImageFile(os.Open("train-images-idx3-24-ubyte"))
	if err != nil {
		log.Fatal(err)
	}
	labels, err := readLabelFile(os.Open("train-labels-idx1-24-ubyte"))
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("len imgs %d", len(imgs))
	//fmt.Println("trying to prepare images and labels")
	data := prepareX(imgs)

	lbl := prepareY(labels)

	visualize(data, 10, 10, "image.png")

	data2, err := zca(data)
	if err != nil {
		fmt.Println("zca errors")
		log.Fatal(err)
	}
	visualize(data2, 10, 10, "image2.png")

	_ = lbl //clear

	nat, err := native.MatrixF64(data2.(*tensor.Dense))
	if err != nil {
		fmt.Println("problem at native.MatrixF64")
		log.Fatal(err)
	}

	log.Printf("Start Training")
	nn := New(Width*Height, Width*Height, 10)
	costs := make([]float64, 0, data2.Shape()[0])
	for e := 0; e < 40; e++ { //epochs
		data2Shape := data2.Shape()
		var oneimg, onelabel tensor.Tensor
		for i := 0; i < data2Shape[0]; i++ {
			if oneimg, err = data.Slice(makeRS(i, i+1)); err != nil {
				log.Fatalf("Unable to slice one image %d", i)
			}
			if onelabel, err = lbl.Slice(makeRS(i, i+1)); err != nil {
				log.Fatalf("Unable to slice one label %d", i)
			}
			var cost float64
			if cost, err = nn.Train(oneimg, onelabel, 0.1); err != nil {
				log.Fatalf("Training error: %+v", err)
			}
			costs = append(costs, cost)
		}
		log.Printf("%d\t%v", e, avg(costs))
		shuffleX(nat)
		costs = costs[:0]
	}
	//log.Printf("End training")

	log.Printf("Start testing")
	testImgs, err := readImageFile(os.Open("test-images-idx3-24-ubyte"))
	if err != nil {
		log.Fatal(err)
	}

	testlabels, err := readLabelFile(os.Open("test-labels-idx1-24-ubyte"))
	if err != nil {
		log.Fatal(err)
	}

	testData := prepareX(testImgs)
	testLbl := prepareY(testlabels)
	shape := testData.Shape()
	testData2, err := zca(testData)
	if err != nil {
		log.Fatal(err)
	}

	visualize(testData, 10, 10, "testData.png")
	visualize(testData2, 10, 10, "testData2.png")

	var correct, total float64
	var oneimg, onelabel tensor.Tensor
	var predicted, errcount int
	for i := 0; i < shape[0]; i++ {
		if oneimg, err = testData.Slice(makeRS(i, i+1)); err != nil {
			log.Fatalf("Unable to slice one image %d", i)
		}
		if onelabel, err = testLbl.Slice(makeRS(i, i+1)); err != nil {
			log.Fatalf("Unable to slice one label %d", i)
		}

		label := argmax(onelabel.Data().([]float64))
		if predicted, err = nn.Predict(oneimg); err != nil {
			log.Fatalf("Failed to predict %d", i)
		}

		if predicted == label {
			correct++
		} else if errcount < 5 { //output pictures that the model classified incorrectly
			visualize(oneimg, 1, 1, fmt.Sprintf("%d_%d_%d.png", i, label, predicted))
			errcount++
		}
		total++
	}

	fmt.Printf("Correct/Totals: %v/%v = %1.3f\n", correct, total, correct/total)
}
