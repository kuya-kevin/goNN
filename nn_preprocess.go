package main

import (
	"math"

	"gorgonia.org/tensor"
	"gorgonia.org/tensor/native"
	"gorgonia.org/vecf64"
)

//conducts PCA = look at all images, figure out how each column correlates with one another
//finds inputs (columns and pixels) that are least correlated w one another
//take principal components found, multiply them by the inputs so that inputs are less correlated w one another
func zca(data tensor.Tensor) (retVal tensor.Tensor, err error) {
	var dataT, data2, sigma tensor.Tensor

	data2 = data.Clone().(tensor.Tensor)

	if err := minusMean(data2); err != nil { //subtract row mean
		return nil, err
	}
	if dataT, err = tensor.T(data2); err != nil { //transpose mean and make a copy of it
		return nil, err
	}

	if sigma, err = tensor.MatMul(dataT, data2); err != nil { //calcs sigma
		return nil, err
	}

	cols := sigma.Shape()[1]

	if _, err = tensor.Div(sigma, float64(cols), tensor.UseUnsafe()); err != nil { //divide sigma by number of columns-1: unbiased estimator
		return nil, err
	}

	s, u, _, err := sigma.(*tensor.Dense).SVD(true, true) //perform an SVD on sigma: break down matrix into its elements
	if err != nil {
		return nil, err
	}

	var diag, uᵀ, tmp tensor.Tensor
	if diag, err = s.Apply(invSqrt(0.08), tensor.UseUnsafe()); err != nil {
		return nil, err
	}
	diag = tensor.New(tensor.AsDenseDiag(diag))

	if uᵀ, err = tensor.T(u); err != nil {
		return nil, err
	}

	if tmp, err = tensor.MatMul(u, diag); err != nil {
		return nil, err
	}

	if tmp, err = tensor.MatMul(tmp, uᵀ); err != nil {
		return nil, err
	}

	if err = tmp.T(); err != nil {
		return nil, err
	}
	return tensor.MatMul(data, tmp)
}

func invSqrt(epsilon float64) func(float64) float64 {
	return func(a float64) float64 {
		return 1 / math.Sqrt(a+epsilon)
	}

}

func minusMean(a tensor.Tensor) error {
	nat, err := native.MatrixF64(a.(*tensor.Dense)) //returns a [][]float64, shares the same allocation as tensor a
	//use [][]float64 as an easy way to iterate through values in the tensor
	if err != nil {
		return err
	}
	for _, row := range nat {
		mean := avg(row)
		vecf64.Trans(row, -mean)
	}

	rows, cols := a.Shape()[0], a.Shape()[1]

	mean := make([]float64, cols)
	for j := 0; j < cols; j++ {
		var colMean float64
		for i := 0; i < rows; i++ {
			colMean += nat[i][j]
		}
		colMean /= float64(rows)
		mean[j] = colMean
	}

	for _, row := range nat {
		vecf64.Sub(row, mean)
	}
	return nil
}
