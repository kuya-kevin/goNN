package main

type rs struct {
	start, end, step int
}

func (s rs) Start() int { return s.start }
func (s rs) End() int   { return s.end }
func (s rs) Step() int  { return s.step }

// helper: makeRS creates a ranged slice.
func makeRS(start, end int) rs {
	step := 1

	return rs{
		start: start,
		end:   end,
		step:  step,
	}
}
