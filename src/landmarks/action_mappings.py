ACTION2PROP = {
    '(goto-a)': 'a',
    '(goto-b)': 'b',
    '(goto-c)': 'c',
    '(goto-d)': 'd',
    '(goto-mail)': 'e',
    '(goto-coffee)': 'f',
    '(goto-office-mail)': 'g',
    '(goto-office-coffee)': 'g',
    '(goto-office-mail-coffee)': 'g',

    '(get-wood)': 'a',
    '(get-wood-1)': 'a',
    '(get-wood-2)': 'a',
    '(get-grass)': 'd',
    '(get-iron)': 'f',
    '(make-plank)': 'b',
    '(make-plank-3)': 'b',
    '(make-rope)': 'b',
    '(make-axe)': 'b',
    '(make-bed)': 'c',
    '(make-stick)': 'c',
    '(make-shears)': 'c',
    '(make-cloth)': 'e',
    '(make-bridge)': 'e',
    '(get-gold)': 'g',
    '(get-gem)': 'h',

    '(lock-caps)': 'C',
    '(unlock-caps)': 'C',
}


def action_to_prop(action):
    if action in ACTION2PROP:
        return ACTION2PROP[action]

    # type actions map to key-value
    if 'type' in action:
        event = action.split(" ")[1]
        return event
    if 'caps' in action:
        return 'C'
