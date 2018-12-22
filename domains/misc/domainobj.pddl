(define (domain craftworld)

  (:requirements
    :strips :adl
  )

  (:types
    wood grass iron - raw-material
    raw-material plank stick cloth rope bridge bed axe shears gold gem - object
  )

  (:predicates
    (have ?o - object)
  )

  (:action get
    :parameters (?m - raw-material)
    :precondition ()
    :effect (and
              (have ?m)
              )
    )

  ; wood + toolshed = plank
  ; grass + toolshed = rope
  ; stick + iron + toolshed = axe
  ; TOOLSHED
  (:action make-plank
    :parameters (?w - wood ?p - plank)
    :precondition (and (have ?w))
    :effect (and (not (have ?w)) (have ?p))
    )

  (:action make-rope
    :parameters (?g - grass ?r - rope)
    :precondition (and (have ?g))
    :effect (and (not (have ?g)) (have ?r))
    )

  (:action make-axe
    :parameters (?s - stick ?i - iron ?a - axe)
    :precondition (and (have ?s) (have ?i))
    :effect (and
              (not (have ?s))
              (not (have ?i))
              (have ?a))
    )

  ; wood + workbench = stick
  ; plank + grass + workbench = bed
  ; stick + iron + workbench = shears
  ; WORKBENCH
  (:action make-stick
    :parameters (?w - wood ?s - stick)
    :precondition (and (have ?w))
    :effect (and (not (have ?w)) (have ?s))
    )

  (:action make-bed
    :parameters (?p - plank ?g - grass ?b - bed)
    :precondition (and (have ?p) (have ?g))
    :effect (and
              (not (have ?p))
              (not (have ?g))
              (have ?b))
    )

  (:action make-shears
    :parameters (?s - stick ?i - iron)
    :precondition (and (have ?s) (have ?i))
    :effect (and
              (not (have ?s))
              (not (have ?i))
              (have ?s))
    )

  ; grass + factory = cloth
  ; iron + wood + factory = bridge
  ; FACTORY
  (:action make-cloth
    :parameters (?g - grass ?c - cloth)
    :precondition (and (have ?g))
    :effect (and
              (not (have ?g))
              (have ?c))
    )

  (:action make-bridge
    :parameters (?i - iron ?w - wood ?b - bridge)
    :precondition (and (have ?i) (have ?w))
    :effect (and
              (not (have ?i))
              (not (have ?w))
              (have ?b))
    )

  ; have the bridge can get the gold
  (:action get-gold
    :parameters (?b - bridge ?g - gold)
    :precondition (and (have ?b))
    :effect (and (have ?g))
    )

  ; have axe can get the gem
  (:action get-gem
    :parameters (?g - gem ?a - axe)
    :precondition (and (have ?a))
    :effect (and (have ?g))
    )
)
