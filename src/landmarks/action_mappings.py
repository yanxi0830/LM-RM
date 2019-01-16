from translate.pddl.custom_utils import GroundAction

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
    'wood': 'a',
    'toolshed': 'b',
    'workbench': 'c',
    'grass': 'd',
    'factory': 'e',
    'iron': 'f',
    'gold': 'g',
    'gem': 'h'
}

KEYBOARD_EVENTS = [e for e in 'abcdefghijklmnopqrstuvwxyzC']
CRAFT_EVENTS = [e for e in 'abcdefgh']
OFFICE_EVENTS = [e for e in 'abcdefg']
FARM_EVENTS = [e for e in 'abcdef']


def mouse_world_action_to_prop(action):
    if action in KEYBOARD_EVENT2PROP:
        return KEYBOARD_EVENT2PROP[action]

    if 'type' in action:
        event = action.split(" ")[1]
        return event
    if 'caps' in action:
        return 'C'

    raise NotImplementedError(action + " is not supported")


def keyboard_world_action_to_prop(action):
    if action in KEYBOARD_EVENT2PROP:
        return KEYBOARD_EVENT2PROP[action]

    if 'type' in action:
        event = action.split(" ")[1]
        return event
    if 'caps' in action:
        return 'C'

    raise NotImplementedError(action + " is not supported")


def craft_world_action_to_prop(action):
    ground_action = GroundAction(action[1:-1])
    loc = ""
    if 'wood' in ground_action.operator:
        loc = 'wood'
    elif 'iron' in ground_action.operator:
        loc = 'iron'
    elif 'grass' in ground_action.operator:
        loc = 'grass'
    elif 'plank' in ground_action.operator or 'rope' in ground_action.operator or 'axe' in ground_action.operator or 'bow' in ground_action.operator:
        loc = 'toolshed'
    elif 'stick' in ground_action.operator or 'bed' in ground_action.operator or 'shears' in ground_action.operator or 'saw' in ground_action.operator:
        loc = 'workbench'
    elif 'cloth' in ground_action.operator or 'bridge' in ground_action.operator or 'goldware' in ground_action.operator or 'ring' in ground_action.operator:
        loc = 'factory'
    elif 'gold' in ground_action.operator:
        loc = 'gold'
    elif 'gem' in ground_action.operator:
        loc = 'gem'
    else:
        raise NotImplementedError(action + " is not supported")

    return CRAFT_EVENT2PROP[loc]


def office_world_action_to_prop(action):
    if action in OFFICE_EVENT2PROP:
        return OFFICE_EVENT2PROP[action]

    raise NotImplementedError(action + " is not supported")


def farm_world_action_to_prop(action):
    if 'get-pig' in action:
        return 'a'
    if 'get-chicken' in action:
        return 'b'
    if 'get-cow' in action:
        return 'c'
    if 'make-pork' in action or 'make-beef' in action or 'make-wings' in action:
        return 'd'  # butchershop
    if 'get-egg' in action or 'get-milk' in action:
        return 'e'  # farmhouse
    if 'make-dessert' in action:
        return 'f'


def action_to_prop(action, world):
    if world == "mouseworld":
        return mouse_world_action_to_prop(action)
    if world == "keyboardworld":
        return keyboard_world_action_to_prop(action)
    if world == "craftworld":
        return craft_world_action_to_prop(action)
    if world == "officeworld":
        return office_world_action_to_prop(action)
    if world == "farmworld":
        return farm_world_action_to_prop(action)

    raise NotImplementedError(world + " is not supported")


def get_all_events(world):
    if world == "mouseworld":
        return KEYBOARD_EVENTS
    if world == "keyboardworld":
        return KEYBOARD_EVENTS
    if world == "craftworld":
        return CRAFT_EVENTS
    if world == "officeworld":
        return OFFICE_EVENTS
    if world == "farmworld":
        return FARM_EVENTS

    raise NotImplementedError(world + " is not supported")
