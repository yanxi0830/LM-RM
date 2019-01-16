(define (domain farmworld)

  (:requirements
    :strips :adl
  )

  (:predicates
    (have-pig)
    (have-cow)
    (have-chicken)
    (have-pork)
    (have-beef)
    (have-wings)
    (have-egg)
    (have-milk)
    (have-dessert)
  )

  (:action get-pig
    :parameters ()
    :precondition ()
    :effect (and
              (have-pig)
              )
    )

  (:action get-cow
    :parameters ()
    :precondition ()
    :effect (and
              (have-cow)
              )
    )

  (:action get-chicken
    :parameters ()
    :precondition ()
    :effect (and
              (have-chicken)
              )
    )

  ; BUTCHERSHOP
  ; pig -> pork
  ; cow -> beef
  ; chicken -> wings
  (:action make-pork
    :parameters ()
    :precondition (and (have-pig))
    :effect (and (not (have-pig)) (have-pork))
    )

  (:action make-beef
    :parameters ()
    :precondition (and (have-cow))
    :effect (and (not (have-cow)) (have-beef))
    )

  (:action make-wings
    :parameters ()
    :precondition (and (have-chicken))
    :effect (and
              (not (have-chicken))
              (have-wings))
    )

  ; FARMHOUSE
  ; chicken->egg
  ; cow->milk
  (:action get-egg
    :parameters ()
    :precondition (and (have-chicken))
    :effect (and (have-egg))
  )

  (:action get-milk
    :parameters ()
    :precondition (and (have-cow))
    :effect (and (have-milk))
    )

  ; KITCHEN
  (:action make-dessert
    :parameters ()
    :precondition (and (have-egg) (have-milk))
    :effect (and (not (have-egg)) (not (have-milk)) (have-dessert))
  )
)
