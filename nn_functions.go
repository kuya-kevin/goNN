package main

import (
	"encoding/binary"
	"image"
	"image/color"
	"image/png"
	"io"
	"math"
	"os"

	"gorgonia.org/tensor"
)

//RawImage should hold brightness values 0-255
type RawImage []byte
type Label uint8

const numLabels = 2 //0-1 classification
const pixelRange = 255

const (
	imageMagic = 0x00000803
	labelMagic = 0x00000801
	Width      = 28 //pixel size
	Height     = 28
)

//read ubyte3 label file
func readLabelFile(r io.Reader, e error) (labels []Label, err error) {
	if e != nil {
		return nil, e
	}

	var magic, n int32
	if err = binary.Read(r, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != labelMagic {
		return nil, os.ErrInvalid
	}
	if err = binary.Read(r, binary.BigEndian, &n); err != nil {
		return nil, err
	}
	labels = make([]Label, n)
	for i := 0; i < int(n); i++ {
		var l Label
		if err := binary.Read(r, binary.BigEndian, &l); err != nil {
			return nil, err
		}
		labels[i] = l
	}
	return labels, nil
}

//we want a []RawImage struct: []byte: slice of 56*56 bytes
//value of each byte represents how bright the pixel is, from 0-255

func readImageFile(r io.Reader, e error) (imgs []RawImage, err error) {

	if e != nil {
		return nil, e
	}

	var magic, n, nrow, ncol int32
	if err = binary.Read(r, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != imageMagic {
		return nil, err /*os.ErrInvalid*/
	}
	if err = binary.Read(r, binary.BigEndian, &n); err != nil {
		return nil, err
	}
	if err = binary.Read(r, binary.BigEndian, &nrow); err != nil {
		return nil, err
	}
	if err = binary.Read(r, binary.BigEndian, &ncol); err != nil {
		return nil, err
	}
	imgs = make([]RawImage, n)
	m := int(nrow * ncol)
	for i := 0; i < int(n); i++ {
		imgs[i] = make(RawImage, m) //0-255 value
		m_, err := io.ReadFull(r, imgs[i])
		if err != nil {
			return nil, err
		}
		if m_ != int(m) {
			return nil, os.ErrInvalid
		}
	}
	return imgs, nil
}

//helper: represent image as slice of floating points, scales from 0-255 to 0-1
func pixelWeight(px byte) float64 {
	retVal := (float64(px) / 255 * 0.999) + 0.001
	if retVal == 1.0 {
		return 0.999
	}
	return retVal
}

//convert []RawImage to tensor
func prepareX(M []RawImage) (retVal tensor.Tensor) {
	rows := len(M)
	cols := len(M[0])

	b := make([]float64, 0, rows*cols) //creates backing array w/ capacity of rows*cols
	for i := 0; i < rows; i++ {
		for j := 0; j < len(M[i]); j++ {
			b = append(b, pixelWeight(M[i][j]))
		}
	}
	res := tensor.New(tensor.WithShape(rows, cols), tensor.WithBacking(b))
	return res
}

//convert labels to tensors
func prepareY(N []Label) (retVal tensor.Tensor) {
	rows := len(N)
	cols := 10

	b := make([]float64, 0, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < 10; j++ {
			if j == int(N[i]) {
				b = append(b, 0.999)
			} else {
				b = append(b, 0.001)
			}
		}
	}
	return tensor.New(tensor.WithShape(rows, cols), tensor.WithBacking(b))
}

//reverses float64 back to byte 0-255
func reversePixelWeight(px float64) byte {
	return byte(((px - 0.001) / 0.999) * 255)
}

// visualize the first n images given a tensor, use 10x10 grid
func visualize(data tensor.Tensor, rows, cols int, filename string) (err error) {
	N := rows * cols

	sliced := data
	if N > 1 {
		sliced, err = data.Slice(makeRS(0, N), nil) // data[0:N, :] in python , slices tensor along first axis
		if err != nil {
			return err
		}
	}

	if err = sliced.Reshape(rows, cols, Width, Height); err != nil { //reshape tnesor into a 4D array, consider changing this to 255,255
		return err
	}
	imCols := Width * cols
	imRows := Height * rows
	rect := image.Rect(0, 0, imCols, imRows)
	canvas := image.NewGray(rect)

	for i := 0; i < cols; i++ {
		for j := 0; j < rows; j++ {
			var patch tensor.Tensor
			if patch, err = sliced.Slice(makeRS(i, i+1), makeRS(j, j+1)); err != nil {

				return err
			}

			patchData := patch.Data().([]float64)
			for k, px := range patchData {
				x := j*Width + k%Width
				y := i*Height + k/Height
				c := color.Gray{reversePixelWeight(px)}
				canvas.Set(x, y, c)
			}
		}
	}

	var f io.WriteCloser
	if f, err = os.Create(filename); err != nil {
		return err
	}

	if err = png.Encode(f, canvas); err != nil {
		f.Close()
		return err
	}

	if err = f.Close(); err != nil {
		return err
	}
	//fmt.Println("did we make it?")
	return nil
}

//helper: normalizes tensor values 0-1
func normalize(data tensor.Tensor) {
	raw := data.Data().([]float64)

	min, max := math.Inf(1), math.Inf(-1)
	for _, v := range raw {
		if v > max {
			max = v
		}
		if v < min {
			min = v
		}
	}
	for i, v := range raw {
		raw[i] = v - min/(max-min)
	}
}
