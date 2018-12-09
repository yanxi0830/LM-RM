from worlds.game_objects import *
import numpy as np
import pygame, time


class Colors(Enum):
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)


class MouseWorldParams:
    def __init__(self, state_file, max_x=900, max_y=300, m_velocity=20, m_radius=10, k_radius=25):
        self.state_file = state_file
        self.max_x = max_x
        self.max_y = max_y
        self.m_velocity_delta = m_velocity
        self.m_velocity_max = 3 * m_velocity
        self.m_radius = m_radius
        self.k_radius = k_radius


class MouseWorld:

    def __init__(self, params):
        self.params = params
        self.keyboard_keys = []
        self._load_map()
        self.last_action = -1
        self.true_props = ""
        self.current_text_field = ""
        self.caps_on = False

    def _get_current_key_hover(self):
        ret = set()
        for k in self.keyboard_keys:
            if self.agent.is_on_key(k):
                ret.add(k)
        return ret

    def _type(self, keycode):
        if keycode == 'C':
            self.caps_on = not self.caps_on
        elif self.caps_on:
            self.current_text_field += keycode.upper()
        else:
            self.current_text_field += keycode

    def _update_events(self):
        self.true_props = ""
        current_key_hover = self._get_current_key_hover()
        if self.last_action == Actions.jump:
            for k in current_key_hover:
                self.true_props += k.keycode
                self._type(k.keycode)

    def draw_current_text_on_display(self, gameDisplay):
        x, y = 30, self.params.max_y - 30
        gameDisplay.blit(pygame.font.SysFont('Arial', 25).render(self.current_text_field, True, Colors.BLUE.value), (x, y))

    def get_true_propositions(self):
        return self.true_props

    def execute_action(self, a, elapsedTime=0.1):
        action = Actions(a)
        # updating the agents velocity
        self.agent.execute_action(action)
        self.last_action = action
        self.agent.update_position(elapsedTime)
        self._update_events()

        # bouncing off walls
        max_x, max_y = self.params.max_x, self.params.max_y
        if self.agent.pos[0] - self.agent.radius < 0 or self.agent.pos[0] + self.agent.radius > max_x:
            if self.agent.pos[0] - self.agent.radius < 0:
                self.agent.pos[0] = self.agent.radius
            else:
                self.agent.pos[0] = max_x - self.agent.radius
            self.agent.vel = self.agent.vel * np.array([-1.0, -1.0])

        if self.agent.pos[1] - self.agent.radius < 0 or self.agent.pos[1] + self.agent.radius > max_y:
            if self.agent.pos[1] - self.agent.radius < 0:
                self.agent.pos[1] = self.agent.radius
            else:
                self.agent.pos[1] = max_y - self.agent.radius
            self.agent.vel = self.agent.vel * np.array([-1.0, -1.0])

    def _load_keyboard(self):
        # QWERTY keys
        keys_r1 = "qwertyuiop"
        keys_r2 = "Casdfghjkl"
        keys_r3 = "zxcvbnm"
        k_radius = self.params.k_radius
        # same row = same y, diff x
        curr_x = k_radius * 4
        curr_y = k_radius * 2
        for ch in keys_r1:
            new_key_pos = [curr_x, curr_y]
            new_key = KeyboardKey(k_radius, ch, new_key_pos)
            self.keyboard_keys.append(new_key)
            curr_x += k_radius * 3

        curr_x = k_radius * 2
        curr_y += k_radius * 3
        for ch in keys_r2:
            new_key_pos = [curr_x, curr_y]
            new_key = KeyboardKey(k_radius, ch, new_key_pos)
            self.keyboard_keys.append(new_key)
            curr_x += k_radius * 3

        curr_x = k_radius * 8
        curr_y += k_radius * 3
        for ch in keys_r3:
            new_key_pos = [curr_x, curr_y]
            new_key = KeyboardKey(k_radius, ch, new_key_pos)
            self.keyboard_keys.append(new_key)
            curr_x += k_radius * 3

    def _load_map(self):
        actions = [Actions.up.value, Actions.left.value, Actions.right.value, Actions.down.value, Actions.jump.value]

        max_x = self.params.max_x
        max_y = self.params.max_y
        vel_delta = self.params.m_velocity_delta
        vel_max = self.params.m_velocity_max
        radius = self.params.m_radius

        # Adding the agent
        pos_a = [2 * radius + random.random() * (max_x - 2 * radius),
                 2 * radius + random.random() * (max_y - 2 * radius)]
        self.agent = MouseAgent(radius, pos_a, [0.0, 0.0], actions, vel_delta, vel_max)

        # Adding static keyboard keys
        self._load_keyboard()


class KeyboardKey:
    def __init__(self, radius, keycode, pos):
        self.radius = radius
        self.keycode = keycode
        self.pos = np.array(pos, dtype=np.float)

    def draw_on_display(self, gameDisplay):
        x, y = self.pos
        r = self.radius
        pygame.draw.rect(gameDisplay, Colors.BLACK.value, [x-r, y-r, r*2, r*2], 2)
        label = self.keycode
        if label == "C":
            label = "caps"
            x -= r - 5
        gameDisplay.blit(pygame.font.SysFont('Arial', 25).render(label, True, Colors.BLACK.value), (x, y))


class MouseAgent:
    def __init__(self, radius, pos, vel, actions, vel_delta, vel_max):
        self.radius = radius
        self.pos = np.array(pos, dtype=np.float)
        self.vel = np.array(vel, dtype=np.float)
        self.actions = actions
        self.reward = 0
        self.vel_delta = float(vel_delta)
        self.vel_max = float(vel_max)

    def update_position(self, elapsedTime):
        self.pos = self.pos + elapsedTime * self.vel

    def execute_action(self, action):
        delta = np.array([0.0, 0.0])
        if action == Actions.up:
            delta = np.array([0.0, -1.0])
        if action == Actions.down:
            delta = np.array([0.0, +1.0])
        if action == Actions.left:
            delta = np.array([-1.0, 0.0])
        if action == Actions.right:
            delta = np.array([+1.0, 0.0])
        self.vel += self.vel_delta * delta

        if action == Actions.jump:
            self.vel = np.array([0.0, 0.0])
        # check limits
        self.vel = np.clip(self.vel, -self.vel_max, self.vel_max)

    def get_actions(self):
        return self.actions

    def is_on_key(self, key):
        d = np.linalg.norm(self.pos - key.pos, ord=2)
        return d <= self.radius + key.radius

    def draw_on_display(self, gameDisplay):
        x, y = int(self.pos[0]), int(self.pos[1])
        pygame.draw.circle(gameDisplay, Colors.RED.value, (x, y), self.radius, 3)

def play():
    params = MouseWorldParams(None)
    max_x = params.max_x
    max_y = params.max_y
    game = MouseWorld(params)

    pygame.init()
    gameDisplay = pygame.display.set_mode((max_x, max_y))
    pygame.display.set_caption('Fake Keyboard')
    clock = pygame.time.Clock()
    crashed = False

    t_previous = time.time()
    actions = set()

    while not crashed:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True

            if event.type == pygame.KEYUP:
                if Actions.left in actions and event.key == pygame.K_LEFT:
                    actions.remove(Actions.left)
                if Actions.right in actions and event.key == pygame.K_RIGHT:
                    actions.remove(Actions.right)
                if Actions.up in actions and event.key == pygame.K_UP:
                    actions.remove(Actions.up)
                if Actions.down in actions and event.key == pygame.K_DOWN:
                    actions.remove(Actions.down)
                if Actions.jump in actions and event.key == pygame.K_SPACE:
                    actions.remove(Actions.jump)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    actions.add(Actions.left)
                if event.key == pygame.K_RIGHT:
                    actions.add(Actions.right)
                if event.key == pygame.K_UP:
                    actions.add(Actions.up)
                if event.key == pygame.K_DOWN:
                    actions.add(Actions.down)
                if event.key == pygame.K_SPACE:
                    actions.add(Actions.jump)

        t_current = time.time()
        t_delta = (t_current - t_previous)

        if len(actions) == 0:
            a = Actions.none
        else:
            a = random.choice(list(actions))

        # Executing the action
        game.execute_action(a.value, t_delta)

        # Printing Image
        gameDisplay.fill(Colors.WHITE.value)
        for k in game.keyboard_keys:
            k.draw_on_display(gameDisplay)
        game.agent.draw_on_display(gameDisplay)
        game.draw_current_text_on_display(gameDisplay)

        pygame.display.update()
        clock.tick(20)

        t_previous = t_current

    pygame.quit()


if __name__ == '__main__':
    play()
