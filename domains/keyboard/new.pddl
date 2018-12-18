(define (problem t1)
	(:domain keyboardworld)
	(:objects
		cursor-0 cursor-1 cursor-2 cursor-3 cursor-4 cursor-5 cursor-6 - cursor-position
	)
	(:init
		(caps-off)
		(current-position cursor-0)
		(maps-to a lo-a)
		(maps-to a up-a)
		(maps-to b lo-b)
		(maps-to b up-b)
		(maps-to c lo-c)
		(maps-to c up-c)
		(maps-to d lo-d)
		(maps-to d up-d)
		(maps-to e lo-e)
		(maps-to e up-e)
		(maps-to f lo-f)
		(maps-to f up-f)
		(maps-to g lo-g)
		(maps-to g up-g)
		(maps-to h lo-h)
		(maps-to h up-h)
		(maps-to i lo-i)
		(maps-to i up-i)
		(maps-to j lo-j)
		(maps-to j up-j)
		(maps-to k lo-k)
		(maps-to k up-k)
		(maps-to l lo-l)
		(maps-to l up-l)
		(maps-to m lo-m)
		(maps-to m up-m)
		(maps-to n lo-n)
		(maps-to n up-n)
		(maps-to o lo-o)
		(maps-to o up-o)
		(maps-to p lo-p)
		(maps-to p up-p)
		(maps-to q lo-q)
		(maps-to q up-q)
		(maps-to r lo-r)
		(maps-to r up-r)
		(maps-to s lo-s)
		(maps-to s up-s)
		(maps-to t lo-t)
		(maps-to t up-t)
		(maps-to u lo-u)
		(maps-to u up-u)
		(maps-to v lo-v)
		(maps-to v up-v)
		(maps-to w lo-w)
		(maps-to w up-w)
		(maps-to x lo-x)
		(maps-to x up-x)
		(maps-to y lo-y)
		(maps-to y up-y)
		(maps-to z lo-z)
		(maps-to z up-z)
		(position-predecessor cursor-0 cursor-1)
		(position-predecessor cursor-1 cursor-2)
		(position-predecessor cursor-2 cursor-3)
		(position-predecessor cursor-3 cursor-4)
		(position-predecessor cursor-4 cursor-5)
		(position-predecessor cursor-5 cursor-6)
	)
	(:goal (and
		(char-at up-b cursor-0)
		(char-at lo-a cursor-1)
		(char-at up-n cursor-2)
		(char-at lo-a cursor-3)
		(char-at lo-n cursor-4)
		(char-at lo-a cursor-5)
		)
	)
)