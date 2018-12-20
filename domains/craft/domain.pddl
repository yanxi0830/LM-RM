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
    (have-goldware)
    (have-ring)
    (have-saw)
    (have-bow)
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
  ; rope + stick + toolshed = bow
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

  (:action make-bow
    :parameters ()
    :precondition (and (have-rope) (have-stick))
    :effect (and (not (have-rope)) (not (have-stick)) (have-bow))
  )

  ; wood + workbench = stick
  ; plank + grass + workbench = bed
  ; stick + iron + workbench = shears
  ; iron + workbench = saw
  ; WORKBENCH
  (:action make-stick
    :parameters ()
    :precondition (and (have-wood))
    :effect (and (not (have-wood)) (have-stick))
    )

  (:action make-saw
    :parameters ()
    :precondition (and (have-iron))
    :effect (and (not (have-iron)) (have-saw))
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
  ; gold + factory = goldware
  ; gem + factory = gem
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

  (:action make-goldware
    :parameters ()
    :precondition (and (have-gold))
    :effect (and (not (have-gold)) (have-goldware))
  )

  (:action make-ring
    :parameters ()
    :precondition (and (have-gem))
    :effect (and (not (have-gem)) (have-ring))
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
