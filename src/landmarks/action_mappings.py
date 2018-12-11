OFFICE_EVENT2PROP = {
    '(goto-a)': 'a',
    '(goto-b)': 'b',
    '(goto-c)': 'c',
    '(goto-d)': 'd',
    '(goto-mail)': 'e',
    '(goto-coffee)': 'f',
    '(goto-office-mail)': 'g',
    '(goto-office-coffee)': 'g',
    '(goto-office-mail-coffee)': 'g'
}

KEYBOARD_EVENT2PROP = {
    '(lock-caps)': 'C',
    '(unlock-caps)': 'C'
}

CRAFT_EVENT2PROP = {
    '(get-wood)': 'a',
    '(get-grass)': 'd',
    '(get-iron)': 'f',
    '(make-plank)': 'b',
    '(make-rope)': 'b',
    '(make-axe)': 'b',
    '(make-bed)': 'c',
    '(make-stick)': 'c',
    '(make-shears)': 'c',
    '(make-cloth)': 'e',
    '(make-bridge)': 'e',
    '(get-gold)': 'g',
    '(get-gem)': 'h'
}

KEYBOARD_EVENTS = [e for e in 'abcdefghijklmnopqrstuvwxyzC']
CRAFT_EVENTS = [e for e in 'abcdefgh']
OFFICE_EVENTS = [e for e in 'abcdefg']


def mouse_world_action_to_prop(action):
    if action in KEYBOARD_EVENT2PROP:
        return KEYBOARD_EVENT2PROP[action]

    if 'type' in action:
        event = action.split(" ")[1]
        return event
    if 'caps' in action:
        return 'C'


def keyboard_world_action_to_prop(action):
    if action in KEYBOARD_EVENT2PROP:
        return KEYBOARD_EVENT2PROP[action]

    if 'type' in action:
        event = action.split(" ")[1]
        return event
    if 'caps' in action:
        return 'C'


def craft_world_action_to_prop(action):
    if action in CRAFT_EVENT2PROP:
        return CRAFT_EVENT2PROP[action]

    if 'wood' in action:
        return 'a'
    if 'plank' in action:
        return 'b'


def office_world_action_to_prop(action):
    if action in OFFICE_EVENT2PROP:
        return OFFICE_EVENT2PROP[action]

    return 'IMPOSSIBLE'


def action_to_prop(action, world):
    if world == "mouseworld":
        return mouse_world_action_to_prop(action)
    if world == "keyboardworld":
        return keyboard_world_action_to_prop(action)
    if world == "craftworld":
        return craft_world_action_to_prop(action)
    if world == "officeworld":
        return office_world_action_to_prop(action)


def get_all_events(world):
    if world == "mouseworld":
        return KEYBOARD_EVENTS
    if world == "keyboardworld":
        return KEYBOARD_EVENTS
    if world == "craftworld":
        return CRAFT_EVENTS
    if world == "officeworld":
        return OFFICE_EVENTS
