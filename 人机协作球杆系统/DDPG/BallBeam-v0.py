import logging
import numpy as np
import random
import gym

logger = logging.getLogger(__name__)

class BallBeamEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 20
    }

    def __init__(self):
        self.max_x = 1.0  # 单位为米
        self.dt = 0.033  # 控制周期
        self.targetX = 0.5  # 目标点
        self.viewer = None

    def _reset(self):
        x = np.random.uniform(0, self.max_x, 1)
        self.state = np.array([x - self.targetX, x, 0, 0])
        return self.state



    def _step(self, action):  # action 为杆旋转速度
        dert_x, x, xdot, phi = self.state

        g = 9.8
        a = 5.0 / 7 * (x * action * action - g * np.sin(phi))  # 加速度

        x = x + xdot * self.dt + 1.0 / 2.0 * a * self.dt * self.dt
        xdot = xdot + a * self.dt
        dert_x = x - self.targetX
        phi = phi + action * self.dt

        self.state = np.array([dert_x, x, xdot, phi]).reshape(4)
        costs2 = 5 * dert_x ** 2 + 0.5 * (xdot ** 2) + 0.001 * (action ** 2)

        done = False
        reward = -costs2  # np.cos((dert_x**2)/0.5*np.pi) - np.clip(0.1 * (xdot**2), 0, 1)
        if(x < 0 or x > self.max_x):
            done = True
            reward = -1
        return self.state, reward, done, {}

    def _seed(self, seed=None):
        self.np_random, seed = random.seeding.np_random(seed)
        return [seed]

    def _render(self, model='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        screen_width = 600
        screen_height = 400
        polewidth = 2
        polelen = 400
        unit_lenth = self.max_x / polelen
        r = 20 #球半径
        phi = self.state[3]
        x = self.state[1]
        beam_leftpoint_x = screen_width / 2 - polelen / 2 * np.cos(phi)
        beam_leftpoint_y = screen_height / 2 - polelen / 2 * np.sin(phi)

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # 画球
            self.ball = rendering.make_circle(r)
            self.balltrans = rendering.Transform()
            self.ball.add_attr(self.balltrans)
            self.ball.set_color(0.8, 0.6, 0.4)

            # 杆
            l, r, t, b = -polelen / 2, polelen / 2, polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0, 0, 1)
            self.poletrans = rendering.Transform()
            pole.add_attr(self.poletrans)
            self.viewer.add_geom(self.ball)
            self.viewer.add_geom(pole)

        if self.state is None:
            return None

        touch_point_x = beam_leftpoint_x + x / unit_lenth * np.cos(phi)
        touch_point_y = beam_leftpoint_y + x / unit_lenth * np.sin(phi)
        circle_point_x = touch_point_x - r * np.sin(phi)
        circle_point_y = touch_point_y + r * np.cos(phi)
        self.balltrans.set_translation(circle_point_x, circle_point_y)

        self.poletrans.set_rotation(phi)
        self.poletrans.set_translation(300, 200)

        return self.viewer.render(return_rgb_array=model == 'rgb_array')

    def _close(self):
        if self.viewer:
            self.viewer.close()
