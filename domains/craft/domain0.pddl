(define (domain craftworld)

  (:requirements
    :strips :adl
  )

  (:predicates
    (have-wood)
    (have-grass)
    (have-iron)
    (have-plank)
    (have-stick)
    (have-cloth)
    (have-rope)
    (have-bridge)
    (have-bed)
    (have-axe)
    (have-shears)
    (have-gold)
    (have-gem)
  )

  (:action get-wood
    :parameters ()
    :precondition ()
    :effect (and
              (have-wood)
              )
    )

  (:action get-grass
    :parameters ()
    :precondition ()
    :effect (and
              (have-grass)
              )
    )

  (:action get-iron
    :parameters ()
    :precondition ()
    :effect (and
              (have-iron)
              )
    )

  ; wood + toolshed = plank
  ; grass + toolshed = rope
  ; stick + iron + toolshed = axe
  ; TOOLSHED
  (:action make-plank
    :parameters ()
    :precondition (and (have-wood))
    :effect (and (not (have-wood)) (have-plank))
    )

  (:action make-rope
    :parameters ()
    :precondition (and (have-grass))
    :effect (and (not (have-grass)) (have-rope))
    )

  (:action make-axe
    :parameters ()
    :precondition (and (have-stick) (have-iron))
    :effect (and
              (not (have-stick))
              (not (have-iron))
              (have-axe))
    )

  ; wood + workbench = stick
  ; plank + grass + workbench = bed
  ; stick + iron + workbench = shears
  ; WORKBENCH
  (:action make-stick
    :parameters ()
    :precondition (and (have-wood))
    :effect (and (not (have-wood)) (have-stick))
    )

  (:action make-bed
    :parameters ()
    :precondition (and (have-plank) (have-grass))
    :effect (and
              (not (have-plank))
              (not (have-grass))
              (have-bed))
    )

  (:action make-shears
    :parameters ()
    :precondition (and (have-stick) (have-iron))
    :effect (and
              (not (have-stick))
              (not (have-iron))
              (have-shears))
    )

  ; grass + factory = cloth
  ; iron + wood + factory = bridge
  ; FACTORY
  (:action make-cloth
    :parameters ()
    :precondition (and (have-grass))
    :effect (and
              (not (have-grass))
              (have-cloth))
    )

  (:action make-bridge
    :parameters ()
    :precondition (and (have-iron) (have-wood))
    :effect (and              (not (have-iron))
              (not (have-wood))
              (have-bridge))
    )

  ; have the bridge can get the gold
  (:action get-gold
    :parameters ()
    :precondition (and (have-bridge))
    :effect (and (have-gold))
    )

  ; have axe can get the gem
  (:action get-gem
    :parameters ()
    :precondition (and (have-axe))
    :effect (and (have-gem))
    )
)
