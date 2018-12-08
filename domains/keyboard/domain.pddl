(define (domain keyboardworld)

  (:requirements
    :typing :strips :adl
  )

  (:types
    upper-char lower-char - character
    cursor-position character key - object)

  (:constants
    up-a up-b up-c up-d up-e up-f up-g up-h up-i up-j up-k up-l up-m up-n up-o up-p up-q up-r up-s up-t up-u up-v up-w up-x up-y up-z - upper-char
    lo-a lo-b lo-c lo-d lo-e lo-f lo-g lo-h lo-i lo-j lo-k lo-l lo-m lo-n lo-o lo-p lo-q lo-r lo-s lo-t lo-u lo-v lo-w lo-x lo-y lo-z - lower-char
    a b c d e f g h i j k l m n o p q r s t u v w x y z - key
    )

  (:predicates
    (char-at ?char - character ?pos - cursor-position)
    (current-position ?pos - cursor-position)
    (position-predecessor ?p1 ?p2 - cursor-position)
    (maps-to ?k - key ?c - character)
    (caps-on)
    (caps-off)
  )

  ; type key to type character at current cursor position
  ; caps
  (:action type-lower
    :parameters (?k - key ?c - lower-char ?curr-p ?next-p - cursor-position)
    :precondition (and
                    (caps-off)
                    (maps-to ?k ?c)
                    (current-position ?curr-p)
                    (position-predecessor ?curr-p ?next-p))
    :effect (and
              (char-at ?c ?curr-p)
              (not (current-position ?curr-p))
              (current-position ?next-p)
              )
    )

  (:action type-upper
    :parameters (?k - key ?c - upper-char ?curr-p ?next-p - cursor-position)
    :precondition (and
                    (caps-on)
                    (maps-to ?k ?c)
                    (current-position ?curr-p)
                    (position-predecessor ?curr-p ?next-p))
    :effect (and
              (not (current-position ?curr-p))
              (char-at ?c ?curr-p)
              (current-position ?next-p)
              )
    )

  (:action lock-caps
    :parameters ()
    :precondition (and (caps-off))
    :effect (and
              (not (caps-off))
              (caps-on))
    )

  (:action unlock-caps
    :parameters ()
    :precondition (and (caps-on))
    :effect (and
              (not (caps-on))
              (caps-off))
    )
)
